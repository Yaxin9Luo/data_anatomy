from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


DEFAULT_PROMPTS: List[str] = [
    "The following text discusses:",
    "Consider the following passage:",
    "In this article, we explore:",
    "Here is a paragraph about:",
    "The problem can be described as:",
    "An overview of the topic:",
    "A short description:",
    "The statement is:",
    "This section covers:",
    "We now present:",
] * 10  # 100 generic prompts


def load_hf_model(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    # Ensure a valid pad token for batch padding with causal LMs (e.g., LLaMA)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
    mdl = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
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
    mdl.eval()
    return mdl, tok


def generate_texts(
    model_name: str,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    batch_size: int = 4,
) -> List[str]:
    prompts = prompts or DEFAULT_PROMPTS
    model, tok = load_hf_model(model_name)

    out_texts: List[str] = []
    device = next(model.parameters()).device
    for i in tqdm(range(0, len(prompts), batch_size), total=(len(prompts)+batch_size-1)//batch_size, desc="Generating"):
        batch = prompts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=getattr(model.config, "pad_token_id", None) or getattr(tok, "pad_token_id", None),
            )
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        # Strip the original prompt to keep only generations after the prompt
        for p, full in zip(batch, texts):
            if full.startswith(p):
                out_texts.append(full[len(p) :].strip())
            else:
                out_texts.append(full)
    return out_texts
