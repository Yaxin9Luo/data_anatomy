from typing import List, Optional
import inspect

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


NEUTRAL_PROMPTS = [
    "Continue the passage:",
    "Write the next paragraph:",
    "The following text discusses:",
    "Consider the following passage:",
    "In this article, we explore:",
    "Here is a paragraph about:",
    "An overview of the topic:",
    "A short description:",
    "The statement is:",
    "This section covers:",
    "We now present:",
    "Background:",
    "Introduction:",
    "Main text:",
    "Details:",
    "Key ideas:",
    "Context:",
    "Notes:",
    "Title:",
    "Abstract:",
    "Overview:",
    "Observation:",
    "Conclusion:",
    "The text continues:",
    "Opening lines:",
    "Body paragraph:",
    "Further explanation:",
    "Elaboration:",
    "Clarification:",
    "Summary:",
    "Rationale:",
    "Motivation:",
    "Discussion:",
    " ",
    "\n",
    "\n\n",
] * 10  
CATEGORY_STYLE_PROMPTS = {
    # CommonCrawl (raw-ish web; include light HTML/boilerplate cues)
    "commoncrawl": [
        "<html><head><title>Article</title></head><body><article><h1>",
        "<div class='content'><p>",
        "By Staff Writer — Updated:",
        "Breaking: ",
        "Related posts:",
        "Privacy Policy — ",
        "Contact us:",
    ],

    # C4 (cleaned web articles; no HTML, news/blog tone, headings)
    "c4": [
        "Title:",
        "Subtitle:",
        "Introduction",
        "Overview",
        "Key takeaways:",
        "Further reading:",
        "Author’s note:",
    ],

    # GitHub (code/files/readmes across languages)
    "github": [
        "```python\n# Filename: utils.py\n\"\"\"Module description:\"\"\"\n",
        "```javascript\n// file: app.js\n// Description:\n",
        "```cpp\n// utils.hpp\n//",
        "```go\n// Package docs:\n",
        "README.md\n# Project Title\n",
        "LICENSE\nMIT License\n",
        "def solve():\n    \"\"\"",
    ],

    # Wikipedia (markup, sections, neutral tone)
    "wikipedia": [
        "== Introduction ==",
        "{{Short description|}}",
        "== History ==",
        "== See also ==",
        "== References ==\n* ",
        "=== Background ===",
        "Infobox:",
    ],

    # Books (long-form narrative/exposition)
    "books": [
        "Chapter 1\n",
        "Prologue\n",
        "It was a",
        "The morning of",
        "He said,",
        "She remembered",
        "The following chapter explores",
    ],

    # arXiv (LaTeX/math/paper structure)
    "arxiv": [
        "\\documentclass{article}\n\\usepackage{amsmath}\n\\title{",
        "\\begin{abstract}\n",
        "\\section{Introduction}\n",
        "We prove the following theorem.",
        "Definition.",
        "Lemma.",
        "Proof.",
    ],

    # StackExchange (Q/A format, tags, accepted-answer vibe)
    "stackexchange": [
        "Title: How do I\n\nQuestion:\n",
        "Q:\nA:",
        "Problem statement:\n",
        "Accepted answer:\n",
        "Steps to reproduce:",
        "Tags: [python] [arrays]",
        "Comment:",
    ],
}


def load_hf_model(model_name: str, revision: Optional[str] = None):
    tok = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
        revision=revision,
    )

    # Ensure a valid pad token for batch padding with causal LMs (e.g., LLaMA)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
            tok.pad_token = "<|pad|>"
    tok.padding_side = "left"
    tok.truncation_side = "left"
    
    # Try loading on a single GPU first, fallback to device_map="auto" if OOM
    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True,
            revision=revision,
        )
    except torch.cuda.OutOfMemoryError:
        print("OOM on single GPU, using device_map='auto'")
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            revision=revision,
        )
    
    # If we added tokens, resize embeddings
    if hasattr(mdl, "get_input_embeddings") and hasattr(mdl, "resize_token_embeddings"):
        vocab_size = mdl.get_input_embeddings().weight.shape[0]
        if tok.vocab_size != vocab_size:
            mdl.resize_token_embeddings(len(tok))
    # Propagate pad_token_id to model config
    try:
        mdl.config.pad_token_id = tok.pad_token_id
    except Exception:
        pass
    try:
        mdl.generation_config.pad_token_id = tok.pad_token_id
    except Exception:
        pass
    if tok.eos_token_id is not None:
        try:
            mdl.generation_config.eos_token_id = tok.eos_token_id
        except Exception:
            try:
                mdl.config.eos_token_id = tok.eos_token_id
            except Exception:
                pass
    mdl.eval()
    return mdl, tok


def generate_texts(
    model_name: str,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    batch_size: int = 4,
    revision: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[str]:
    prompts = prompts or NEUTRAL_PROMPTS
    if not prompts:
        raise ValueError("No prompts provided for generation")
    model, tok = load_hf_model(model_name, revision=revision)

    out_texts: List[str] = []
    
    # Determine the device strategy
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        # Model is distributed across devices, need to move inputs to first device
        candidate_devices = []
        for dev in model.hf_device_map.values():
            if isinstance(dev, torch.device):
                if dev.type != "meta":
                    candidate_devices.append(dev)
            elif isinstance(dev, str):
                if dev.startswith("cuda") or dev.startswith("cpu"):
                    candidate_devices.append(torch.device(dev))
            elif isinstance(dev, int):
                candidate_devices.append(torch.device(f"cuda:{dev}"))
        if not candidate_devices:
            candidate_devices.append(torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        # Prefer CUDA if available in the map
        def _device_key(dev: torch.device) -> tuple:
            if dev.type == "cuda":
                # Prefer lower index CUDA devices first
                return (0, dev.index or 0)
            if dev.type == "cpu":
                return (1, 0)
            return (2, 0)

        candidate_devices.sort(key=_device_key)
        first_device = candidate_devices[0]
        use_device_map = True
    else:
        # Model is on a single device
        first_device = next(model.parameters()).device
        use_device_map = False
    
    # Optional torch.Generator for reproducible sampling across devices
    gen_obj = None
    if seed is not None:
        try:
            if first_device.type == "cuda":
                gen_obj = torch.Generator(device=first_device)
            else:
                gen_obj = torch.Generator()
            gen_obj.manual_seed(int(seed))
        except Exception:
            gen_obj = None

    # Detect whether this transformers version supports `generator` kwarg
    supports_generator = False
    try:
        sig = inspect.signature(model.generate)
        supports_generator = "generator" in sig.parameters
    except Exception:
        supports_generator = False

    for i in tqdm(range(0, len(prompts), batch_size), total=(len(prompts)+batch_size-1)//batch_size, desc="Generating"):
        batch = prompts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True)
        # Remove token_type_ids if present, as some models (like OLMo) don't use them
        if "token_type_ids" in enc:
            enc.pop("token_type_ids")
        
        # Move inputs to the appropriate device
        if use_device_map:
            # For distributed models, move to the primary device (Accelerate handles sharding)
            enc = enc.to(first_device)
        else:
            # For single-device models, move to model's device
            enc = enc.to(first_device)
            
        with torch.no_grad():
            # If `generator` kwarg is not supported, set and restore global RNG around generate
            use_local_rng = (seed is not None) and (not supports_generator)
            if use_local_rng:
                batch_seed = int(seed) + i
                cpu_state = torch.get_rng_state()
                cuda_states = None
                if torch.cuda.is_available():
                    try:
                        cuda_states = torch.cuda.get_rng_state_all()
                    except Exception:
                        cuda_states = None
                torch.manual_seed(batch_seed)
                if torch.cuda.is_available():
                    try:
                        torch.cuda.manual_seed_all(batch_seed)
                    except Exception:
                        pass
            try:
                gen_kwargs = dict(
                    enc,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=(getattr(model.config, "pad_token_id", None) or getattr(tok, "pad_token_id", None)),
                )
                if supports_generator and gen_obj is not None:
                    gen_kwargs["generator"] = gen_obj
                gen = model.generate(**gen_kwargs)
            except Exception as e:
                # Known issue: some OLMo/hf_olmo revisions error on None past_key_values.
                # Also handle older transformers that do not support `generator` kwarg.
                print(
                    f"Generation error: {e}\n"
                    "Retrying with use_cache=False and without generator if unsupported. "
                    "Consider pinning --hf_revision to a known-good checkpoint."
                )
                gen_kwargs = dict(
                    enc,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=(getattr(model.config, "pad_token_id", None) or getattr(tok, "pad_token_id", None)),
                    use_cache=False,
                )
                # Only pass generator if supported
                if supports_generator and gen_obj is not None:
                    gen_kwargs["generator"] = gen_obj
                try:
                    gen = model.generate(**gen_kwargs)
                except Exception as e2:
                    # Final fallback: remove generator entirely
                    if "generator" in gen_kwargs:
                        gen_kwargs.pop("generator", None)
                    gen = model.generate(**gen_kwargs)
            finally:
                if use_local_rng:
                    try:
                        torch.set_rng_state(cpu_state)
                        if cuda_states is not None:
                            # Restore per-device CUDA RNG states if available
                            try:
                                torch.cuda.set_rng_state_all(cuda_states)
                            except Exception:
                                # Best-effort: restore on current device only
                                try:
                                    torch.cuda.set_rng_state(cuda_states[0])
                                except Exception:
                                    pass
                    except Exception:
                        pass
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        # Strip the original prompt to keep only generations after the prompt
        for p, full in zip(batch, texts):
            if full.startswith(p):
                out_texts.append(full[len(p) :].strip())
            else:
                out_texts.append(full)
    return out_texts
