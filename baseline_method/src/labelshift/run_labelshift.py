import argparse
import json
from pathlib import Path
from typing import Optional, List

import numpy as np
from tqdm import tqdm

from data_utils import build_balanced_splits
from classifier import train_tfidf_classifier
from generate import generate_texts, DEFAULT_PROMPTS
from prior import estimate_priors_least_squares, bootstrap_priors
from viz import plot_confusion_matrix, plot_priors_with_ci, plot_pbar_vs_ctpi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Label-shift mixture estimation via domain classifier + prior correction"
    )
    # Data
    p.add_argument("--local_samples_dir", type=str, default=str(Path(__file__).resolve().parents[2] / "data_samples"))
    p.add_argument("--merge_web", action="store_true", help="Merge CommonCrawl and C4 into a single Web class (6-way)")
    p.add_argument("--max_per_class", type=int, default=2000)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)

    # Classifier
    p.add_argument("--classifier", type=str, choices=["tfidf"], default="tfidf")
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--predict_batch_size", type=int, default=256)

    # Generations
    p.add_argument("--target_model", type=str, required=False, help="HF model name to sample from (e.g., huggyllama/llama-7b)")
    p.add_argument("--use_cached_generations", type=str, default=None, help="Path to JSONL with {text} per line to skip generation")
    p.add_argument("--num_prompts", type=int, default=100)
    p.add_argument("--prompts_file", type=str, default=None)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--gen_temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--gen_batch_size", type=int, default=4)

    # Bootstrap
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--n_boot", type=int, default=200)

    # Output
    p.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "out"))
    p.add_argument("--run_name", type=str, default="labelshift")
    return p.parse_args()


def load_prompts_from_file(path: str) -> List[str]:
    arr: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                arr.append(s)
    return arr


def read_jsonl_texts(path: str) -> List[str]:
    arr: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("text")
            if isinstance(t, str) and t.strip():
                arr.append(t)
    return arr


def main() -> None:
    args = parse_args()

    # 1) Build balanced dataset
    ds = build_balanced_splits(
        local_dir=args.local_samples_dir,
        merge_web=args.merge_web,
        max_per_class=args.max_per_class,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    K = len(ds.categories)
    print("Training TF-IDF classifier...")
    # 2) Train classifier (TF-IDF baseline)
    model, clf_metrics, C = train_tfidf_classifier(
        ds.train.texts,
        ds.train.labels,
        ds.val.texts,
        ds.val.labels,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    print(f"Classifier trained! Validation accuracy: {clf_metrics['val_acc']:.3f}")
    # 3) Gather model generations or load cached
    if args.use_cached_generations:
        gen_texts = read_jsonl_texts(args.use_cached_generations)
    else:
        if not args.target_model:
            raise ValueError("--target_model required unless --use_cached_generations is provided")
        if args.prompts_file:
            prompts = load_prompts_from_file(args.prompts_file)
            prompts = prompts[: args.num_prompts]
        else:
            prompts = DEFAULT_PROMPTS[: args.num_prompts]
        gen_texts = generate_texts(
            model_name=args.target_model,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.gen_temperature,
            top_p=args.top_p,
            batch_size=args.gen_batch_size,
        )

    # 4) Predict class probabilities on generations and average
    probs_chunks: List[np.ndarray] = []
    bs = max(1, int(args.predict_batch_size))
    for i in tqdm(range(0, len(gen_texts), bs), total=(len(gen_texts)+bs-1)//bs, desc="Predicting probs"):
        batch = gen_texts[i:i+bs]
        probs_chunks.append(model.predict_proba(batch))
    probs = np.concatenate(probs_chunks, axis=0) if probs_chunks else np.zeros((0, K))
    pbar = probs.mean(axis=0)

    # 5) Prior correction: solve for mixture pi
    pi = estimate_priors_least_squares(C, pbar)

    # 6) Optional bootstrap for CIs
    pi_mean = pi
    lo = np.zeros_like(pi)
    hi = np.zeros_like(pi)
    if args.bootstrap:
        # Manual bootstrap loop to expose progress bar
        rng = np.random.default_rng(args.seed)
        N = probs.shape[0]
        pis = []
        for _ in tqdm(range(args.n_boot), desc="Bootstrapping"):
            idx = rng.integers(0, N, size=N)
            pbar_b = probs[idx].mean(axis=0)
            pi_b = estimate_priors_least_squares(C, pbar_b)
            pis.append(pi_b)
        P = np.stack(pis, axis=0)
        pi_mean = P.mean(axis=0)
        lo = np.percentile(P, 2.5, axis=0)
        hi = np.percentile(P, 97.5, axis=0)

    # 7) Write outputs
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": {
            "local_samples_dir": args.local_samples_dir,
            "merge_web": args.merge_web,
            "classifier": args.classifier,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "target_model": args.target_model,
            "num_prompts": args.num_prompts,
            "max_new_tokens": args.max_new_tokens,
            "gen_temperature": args.gen_temperature,
            "top_p": args.top_p,
            "bootstrap": args.bootstrap,
            "n_boot": args.n_boot,
        },
        "categories": ds.categories,
        "val_metrics": clf_metrics,
        "confusion_matrix": C.tolist(),
        "pbar": pbar.tolist(),
        "priors": {
            "point": pi.tolist(),
            "mean": pi_mean.tolist(),
            "ci_lo": lo.tolist(),
            "ci_hi": hi.tolist(),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(out_dir / "summary.csv", "w", encoding="utf-8") as f:
        f.write("category,pi,ci_lo,ci_hi\n")
        for c, p, a, b in zip(ds.categories, pi_mean, lo, hi):
            f.write(f"{c},{p:.6f},{a:.6f},{b:.6f}\n")

    print(f"Wrote label-shift outputs to {out_dir}")

    # 8) Visualizations
    try:
        plot_confusion_matrix(C, ds.categories, str(out_dir / "confusion_matrix.png"))
        plot_priors_with_ci(ds.categories, pi_mean, lo, hi, str(out_dir / "priors.png"))
        Ctpi = (C.T @ pi_mean)
        plot_pbar_vs_ctpi(ds.categories, pbar, Ctpi, str(out_dir / "pbar_vs_ctpi.png"))
    except Exception as e:
        print(f"Warning: plotting failed: {e}")


if __name__ == "__main__":
    main()
