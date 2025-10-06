import argparse
import json
from pathlib import Path
from typing import Optional, List

import numpy as np
from tqdm import tqdm

from data_utils import build_balanced_splits
from classifier import train_tfidf_classifier, train_distilbert_classifier
from generate import generate_texts, NEUTRAL_PROMPTS
from prior import estimate_priors_least_squares, bootstrap_priors
from viz import plot_confusion_matrix, plot_priors_with_ci, plot_pbar_vs_ctpi
from sklearn.neighbors import NearestNeighbors
import csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Label-shift mixture estimation via domain classifier + prior correction"
    )
    # Data
    p.add_argument("--local_samples_dir", type=str, default=str(Path(__file__).resolve().parents[2] / "data_samples"))
    p.add_argument("--merge_web", action="store_true", help="Merge CommonCrawl and C4 into a single Web class (6-way)")
    p.add_argument("--max_per_class", type=int, default=5000)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)

    # Classifier
    p.add_argument("--classifier", type=str, choices=["tfidf", "distilbert"], default="distilbert")
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--predict_batch_size", type=int, default=256)
    p.add_argument("--clf_verbose", type=int, default=0)

    # HF classifier (DistilBERT) options
    p.add_argument("--hf_model_name", type=str, default="distilbert/distilbert-base-uncased")
    p.add_argument("--hf_epochs", type=int, default=3)
    p.add_argument("--hf_batch_size", type=int, default=64)
    p.add_argument("--hf_lr", type=float, default=2e-5)
    p.add_argument("--hf_weight_decay", type=float, default=0.01)
    p.add_argument("--hf_max_length", type=int, default=256)

    # Generations
    p.add_argument("--generator", type=str, choices=["hf"], default="hf", help="Generation engine to use.")
    p.add_argument("--target_model", type=str, required=False, help="HF model name (for --generator hf)")
    p.add_argument("--hf_revision", type=str, default=None, help="HF revision/commit/tag for --target_model (e.g., specific training step)")
    p.add_argument("--use_cached_generations", type=str, default=None, help="Path to JSONL with {text} per line to skip generation")
    p.add_argument("--num_prompts", type=int, default=400)
    p.add_argument("--prompts_file", type=str, default=None)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--gen_temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--gen_batch_size", type=int, default=8)

    # Bootstrap
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--n_boot", type=int, default=300)

    # Inspection options
    p.add_argument("--inspect", action="store_true", help="Dump per-sample predictions and NN neighbors (DistilBERT only)")
    p.add_argument("--nn_k", type=int, default=5, help="#neighbors per generated sample for inspection")
    p.add_argument("--nn_max_gens", type=int, default=20, help="Max generations to analyze for NN neighbors")
    p.add_argument("--snippet_len", type=int, default=200, help="Max chars for text snippets in CSV/JSONL")

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
    # Prepare output directory early to store training logs/curves
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training classifier: {args.classifier}...")
    # 2) Train classifier
    if args.classifier == "tfidf":
        model, clf_metrics, C = train_tfidf_classifier(
            ds.train.texts,
            ds.train.labels,
            ds.val.texts,
            ds.val.labels,
            seed=args.seed,
            n_jobs=args.n_jobs,
            verbose=args.clf_verbose,
        )
    else:
        model, clf_metrics, C = train_distilbert_classifier(
            ds.train.texts,
            ds.train.labels,
            ds.val.texts,
            ds.val.labels,
            model_name=args.hf_model_name,
            epochs=args.hf_epochs,
            batch_size=args.hf_batch_size,
            lr=args.hf_lr,
            weight_decay=args.hf_weight_decay,
            max_length=args.hf_max_length,
            seed=args.seed,
        )
    print(f"Classifier trained! Validation accuracy: {clf_metrics['val_acc']:.3f}")
    # 3) Gather model generations or load cached
    if args.use_cached_generations:
        gen_texts = read_jsonl_texts(args.use_cached_generations)
    else:
        # Get prompts
        if args.prompts_file:
            prompts = load_prompts_from_file(args.prompts_file)
            prompts = prompts[: args.num_prompts]
        else:
            print("Using neutral prompts !!!!!!!!!!!!!!!!!")
            prompts = NEUTRAL_PROMPTS[: args.num_prompts]

        # Generate texts using the selected engine
        if args.generator == "hf":
            if not args.target_model:
                raise ValueError("--target_model is required for --generator 'hf'")
            gen_texts = generate_texts(
                model_name=args.target_model,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.gen_temperature,
                top_p=args.top_p,
                batch_size=args.gen_batch_size,
                revision=args.hf_revision,
            )
        else:
            raise ValueError(f"Unknown generator: {args.generator}")

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

    # 7) Write outputs (out_dir already created)

    payload = {
        "config": {
            "local_samples_dir": args.local_samples_dir,
            "merge_web": args.merge_web,
            "classifier": args.classifier,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "target_model": args.target_model,
            "generator": args.generator,
            "num_prompts": args.num_prompts,
            "max_new_tokens": args.max_new_tokens,
            "gen_temperature": args.gen_temperature,
            "top_p": args.top_p,
            "hf_revision": args.hf_revision,
            "hf_model_name": args.hf_model_name,
            "hf_epochs": args.hf_epochs,
            "hf_batch_size": args.hf_batch_size,
            "hf_lr": args.hf_lr,
            "hf_weight_decay": args.hf_weight_decay,
            "hf_max_length": args.hf_max_length,
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

    # 8) Optional inspection: dump per-sample predictions and nearest neighbors
    if args.inspect:
        try:
            # Dump predictions for train/val
            def _topk_info(probs_row: np.ndarray, cats: List[str], k: int = 3) -> List[str]:
                idx = np.argsort(-probs_row)[:k]
                return [f"{cats[j]}:{probs_row[j]:.3f}" for j in idx]

            # Train predictions
            train_probs = model.predict_proba(ds.train.texts)
            train_pred = np.argmax(train_probs, axis=1)
            # Val predictions
            val_probs = model.predict_proba(ds.val.texts)
            val_pred = np.argmax(val_probs, axis=1)

            tv_path = out_dir / "train_val_predictions.csv"
            with open(tv_path, "w", encoding="utf-8", newline="") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(["split", "index", "true_class", "pred_class", "top1_conf", "top3", "text_snippet"])
                for i, (y, pp, txt) in enumerate(zip(ds.train.labels, train_probs, ds.train.texts)):
                    writer.writerow([
                        "train",
                        i,
                        ds.categories[y],
                        ds.categories[int(np.argmax(pp))],
                        f"{float(np.max(pp)):.3f}",
                        "|".join(_topk_info(pp, ds.categories)),
                        txt[: args.snippet_len].replace("\n", " ")
                    ])
                for i, (y, pp, txt) in enumerate(zip(ds.val.labels, val_probs, ds.val.texts)):
                    writer.writerow([
                        "val",
                        i,
                        ds.categories[y],
                        ds.categories[int(np.argmax(pp))],
                        f"{float(np.max(pp)):.3f}",
                        "|".join(_topk_info(pp, ds.categories)),
                        txt[: args.snippet_len].replace("\n", " ")
                    ])

            # Generated predictions
            gen_path = out_dir / "generated_predictions.csv"
            with open(gen_path, "w", encoding="utf-8", newline="") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(["gen_index", "pred_class", "top1_conf", "top3", "text_snippet"])
                for i, (pp, txt) in enumerate(zip(probs, gen_texts)):
                    writer.writerow([
                        i,
                        ds.categories[int(np.argmax(pp))],
                        f"{float(np.max(pp)):.3f}",
                        "|".join(_topk_info(pp, ds.categories)),
                        txt[: args.snippet_len].replace("\n", " ")
                    ])

            # Nearest neighbors (DistilBERT only)
            # Guard: embeddings available only for HFSequenceClassifier
            if hasattr(model, "embeddings"):
                # Train embeddings index
                train_emb = model.embeddings(ds.train.texts, batch_size=max(8, bs))
                nn = NearestNeighbors(n_neighbors=max(1, args.nn_k), metric="cosine", algorithm="brute")
                nn.fit(train_emb)
                # Gen subset
                m = min(max(1, args.nn_max_gens), len(gen_texts))
                sub_gen_texts = gen_texts[:m]
                gen_emb = model.embeddings(sub_gen_texts, batch_size=max(8, bs))
                dists, nbrs = nn.kneighbors(gen_emb, return_distance=True)

                # Write JSONL with neighbors
                import json as _json
                neigh_path = out_dir / "gen_neighbors.jsonl"
                with open(neigh_path, "w", encoding="utf-8") as fj:
                    for gi in range(m):
                        pp = probs[gi]
                        rec = {
                            "gen_index": gi,
                            "pred_class": ds.categories[int(np.argmax(pp))],
                            "top1_conf": float(np.max(pp)),
                            "text_snippet": sub_gen_texts[gi][: args.snippet_len],
                            "neighbors": []
                        }
                        for rk, (ti, dd) in enumerate(zip(nbrs[gi].tolist(), dists[gi].tolist())):
                            ti = int(ti)
                            rec["neighbors"].append({
                                "rank": rk + 1,
                                "train_index": ti,
                                "true_class": ds.categories[int(ds.train.labels[ti])],
                                "distance": float(dd),
                                "text_snippet": ds.train.texts[ti][: args.snippet_len]
                            })
                        fj.write(_json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                print("Inspector: skipping NN neighbors because embeddings() not available for this classifier.")

            print(f"Inspector outputs written to {out_dir}")
        except Exception as e:
            print(f"Warning: inspection failed: {e}")

    # 9) Visualizations
    try:
        plot_confusion_matrix(C, ds.categories, str(out_dir / "confusion_matrix.png"))
        plot_priors_with_ci(ds.categories, pi_mean, lo, hi, str(out_dir / "priors.png"))
        Ctpi = (C.T @ pi_mean)
        plot_pbar_vs_ctpi(ds.categories, pbar, Ctpi, str(out_dir / "pbar_vs_ctpi.png"))
    except Exception as e:
        print(f"Warning: plotting failed: {e}")


if __name__ == "__main__":
    main()
