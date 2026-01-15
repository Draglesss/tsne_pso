"""
Embedding benchmark: TSNE_PSO vs sklearn TSNE and a few baselines.

This script is designed to be:
- reproducible (fixed seeds)
- offline (uses bundled sklearn datasets / generators)
- reasonably fair (shared preprocessing + consistent metrics)

Metrics:
- runtime (seconds)
- trustworthiness (sklearn.manifold.trustworthiness)

Usage:
  python3 examples/benchmark.py --dataset digits
  python3 examples/benchmark.py --dataset swissroll --n-samples 2000
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class RunResult:
    name: str
    seconds: float
    trustworthiness: float
    seed: int
    config: Dict[str, Any]


@dataclass(frozen=True)
class AggregateResult:
    name: str
    seconds_mean: float
    seconds_std: float
    trust_mean: float
    trust_std: float
    best_trust: float
    best_seed: int
    best_config: Dict[str, Any]


def _load_dataset(name: str, seed: int, n_samples: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    # Offline + deterministic datasets only.
    if name == "digits":
        from sklearn.datasets import load_digits

        ds = load_digits()
        X = ds.data.astype(np.float64, copy=False)
        y = ds.target
        return X, y

    if name == "iris":
        from sklearn.datasets import load_iris

        ds = load_iris()
        X = ds.data.astype(np.float64, copy=False)
        y = ds.target
        return X, y

    if name == "swissroll":
        from sklearn.datasets import make_swiss_roll

        X, y = make_swiss_roll(n_samples=n_samples, random_state=seed)
        X = X.astype(np.float64, copy=False)
        return X, y

    raise ValueError(f"Unknown dataset: {name!r}")


def _subsample(X: np.ndarray, y: Optional[np.ndarray], seed: int, max_samples: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if max_samples <= 0 or X.shape[0] <= max_samples:
        return X, y
    rng = np.random.RandomState(seed)
    idx = rng.permutation(X.shape[0])[:max_samples]
    Xs = X[idx]
    ys = y[idx] if y is not None else None
    return Xs, ys


def _preprocess(X: np.ndarray, pca_dim: int, seed: int) -> np.ndarray:
    # Standardize features; optionally apply PCA for a controlled input dimension.
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    if pca_dim <= 0 or pca_dim >= Xs.shape[1]:
        return Xs

    from sklearn.decomposition import PCA

    return PCA(n_components=pca_dim, random_state=seed).fit_transform(Xs)


def _timeit(fn: Callable[[], np.ndarray]) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    Y = fn()
    t1 = time.perf_counter()
    return Y, (t1 - t0)


def _trustworthiness(X: np.ndarray, Y: np.ndarray, n_neighbors: int, metric: str) -> float:
    from sklearn.manifold import trustworthiness

    return float(trustworthiness(X, Y, n_neighbors=n_neighbors, metric=metric))


def _try_umap_model(seed: int) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    # Optional baseline. If not installed, skip.
    try:
        import umap  # type: ignore
    except Exception:
        return None

    def run(X: np.ndarray) -> np.ndarray:
        model = umap.UMAP(
            n_components=2,
            random_state=seed,
            n_neighbors=15,
            min_dist=0.1,
        )
        return model.fit_transform(X)

    return run


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["digits", "iris", "swissroll"], default="digits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=2000, help="Used for swissroll only.")
    p.add_argument("--max-samples", type=int, default=1000, help="Subsample to at most this many rows (0 disables).")

    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--learning-rate", type=float, default=200.0)
    p.add_argument("--max-iter", type=int, default=250)
    p.add_argument("--n-particles", type=int, default=50)
    p.add_argument("--n-jobs", type=int, default=4)

    p.add_argument("--pca-dim", type=int, default=50, help="Apply PCA to this many dims before models (0 disables).")
    p.add_argument("--trust-k", type=int, default=12, help="Neighborhood size for trustworthiness.")
    p.add_argument("--runs", type=int, default=1, help="Repeat each model and report the best trustworthiness.")
    p.add_argument(
        "--models",
        default="tsne_pso,sklearn_tsne,pca,isomap,spectral_embedding",
        help="Comma-separated list (subset) of models to run.",
    )
    p.add_argument(
        "--seeds",
        default="42,43,44",
        help="Comma-separated list of seeds. Results are aggregated across seeds for fairness.",
    )
    p.add_argument(
        "--time-budget-sec",
        type=float,
        default=20.0,
        help="Per-model time budget (seconds) for hyperparameter search, per seed.",
    )
    args = p.parse_args(argv)

    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap, SpectralEmbedding

    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    X_raw, y = _load_dataset(args.dataset, seed=int(seeds[0]), n_samples=args.n_samples)
    X_raw, y = _subsample(X_raw, y, seed=args.seed, max_samples=int(args.max_samples))
    X = _preprocess(X_raw, pca_dim=args.pca_dim, seed=args.seed)

    # Model runners (all return a 2D embedding).
    runners: Dict[str, Callable[[np.ndarray, Dict[str, Any], int], np.ndarray]] = {}
    grids: Dict[str, List[Dict[str, Any]]] = {}

    def run_tsne_pso(Xin: np.ndarray, cfg: Dict[str, Any], seed: int) -> np.ndarray:
        from tsne_pso import TSNE_PSO

        model = TSNE_PSO(
            n_components=2,
            perplexity=float(cfg["perplexity"]),
            n_particles=int(cfg["n_particles"]),
            max_iter=int(cfg["max_iter"]),
            learning_rate=float(cfg["learning_rate"]),
            random_state=int(seed),
            n_jobs=int(cfg["n_jobs"]),
            verbose=False,
        )
        return model.fit_transform(Xin)

    runners["tsne_pso"] = run_tsne_pso
    grids["tsne_pso"] = [
        {"perplexity": args.perplexity, "learning_rate": args.learning_rate, "max_iter": 150, "n_particles": 20, "n_jobs": args.n_jobs},
        {"perplexity": args.perplexity, "learning_rate": args.learning_rate, "max_iter": 250, "n_particles": 30, "n_jobs": args.n_jobs},
        {"perplexity": args.perplexity, "learning_rate": args.learning_rate, "max_iter": 250, "n_particles": 50, "n_jobs": args.n_jobs},
        {"perplexity": args.perplexity, "learning_rate": args.learning_rate, "max_iter": 350, "n_particles": 30, "n_jobs": args.n_jobs},
    ]

    def run_sklearn_tsne(Xin: np.ndarray, cfg: Dict[str, Any], seed: int) -> np.ndarray:
        model = TSNE(
            n_components=2,
            perplexity=float(cfg["perplexity"]),
            learning_rate=float(cfg["learning_rate"]),
            max_iter=int(cfg["max_iter"]),
            init=str(cfg["init"]),
            method=str(cfg["method"]),
            random_state=int(seed),
        )
        return model.fit_transform(Xin)

    runners["sklearn_tsne"] = run_sklearn_tsne
    grids["sklearn_tsne"] = [
        {"perplexity": args.perplexity, "learning_rate": args.learning_rate, "max_iter": 250, "init": "random", "method": "barnes_hut"},
        {"perplexity": args.perplexity, "learning_rate": args.learning_rate, "max_iter": 500, "init": "random", "method": "barnes_hut"},
        {"perplexity": args.perplexity, "learning_rate": args.learning_rate, "max_iter": 750, "init": "random", "method": "barnes_hut"},
    ]

    def run_pca_2d(Xin: np.ndarray, cfg: Dict[str, Any], seed: int) -> np.ndarray:
        return PCA(n_components=2, random_state=args.seed).fit_transform(Xin)

    runners["pca"] = run_pca_2d
    grids["pca"] = [{"_": 0}]

    def run_isomap(Xin: np.ndarray, cfg: Dict[str, Any], seed: int) -> np.ndarray:
        return Isomap(n_components=2, n_neighbors=int(cfg["n_neighbors"])).fit_transform(Xin)

    runners["isomap"] = run_isomap
    grids["isomap"] = [{"n_neighbors": 10}, {"n_neighbors": 15}, {"n_neighbors": 30}]

    def run_spectral(Xin: np.ndarray, cfg: Dict[str, Any], seed: int) -> np.ndarray:
        return SpectralEmbedding(n_components=2, random_state=seed).fit_transform(Xin)

    runners["spectral_embedding"] = run_spectral
    grids["spectral_embedding"] = [{"_": 0}]

    umap_runner = _try_umap_model(seed=args.seed)
    if umap_runner is not None:
        def run_umap(Xin: np.ndarray, cfg: Dict[str, Any], seed: int) -> np.ndarray:
            # Seed captured when the runner is built; keep config for uniformity.
            return umap_runner(Xin)

        runners["umap"] = run_umap
        grids["umap"] = [{"_": 0}]

    requested = [m.strip() for m in str(args.models).split(",") if m.strip()]
    unknown = [m for m in requested if m not in runners]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Available: {sorted(runners.keys())}")

    def budgeted_best_for_seed(
        *,
        name: str,
        runner: Callable[[np.ndarray, Dict[str, Any], int], np.ndarray],
        grid: List[Dict[str, Any]],
        seed: int,
        budget_sec: float,
    ) -> RunResult:
        best: Optional[RunResult] = None
        elapsed = 0.0
        # Run configs in order until the budget is exhausted. Always run at least one.
        for i, cfg in enumerate(grid):
            if i > 0 and elapsed >= budget_sec:
                break
            Y, secs = _timeit(lambda: runner(X, cfg, seed))
            elapsed += secs
            tw = _trustworthiness(X, Y, n_neighbors=int(args.trust_k), metric="euclidean")
            r = RunResult(name=name, seconds=secs, trustworthiness=tw, seed=seed, config=dict(cfg))
            if best is None or r.trustworthiness > best.trustworthiness:
                best = r
        assert best is not None
        return best

    # For each seed, each model gets the same time budget to select its best config.
    per_seed_results: List[RunResult] = []
    for seed in seeds:
        for name in requested:
            runner = runners[name]
            grid = grids.get(name, [{"_": 0}])
            best = budgeted_best_for_seed(
                name=name,
                runner=runner,
                grid=grid,
                seed=seed,
                budget_sec=float(args.time_budget_sec),
            )
            per_seed_results.append(best)

    aggregates: List[AggregateResult] = []
    for name in requested:
        rs = [r for r in per_seed_results if r.name == name]
        secs = np.array([r.seconds for r in rs], dtype=np.float64)
        tws = np.array([r.trustworthiness for r in rs], dtype=np.float64)
        best_idx = int(np.argmax(tws))
        aggregates.append(
            AggregateResult(
                name=name,
                seconds_mean=float(secs.mean()),
                seconds_std=float(secs.std(ddof=0)),
                trust_mean=float(tws.mean()),
                trust_std=float(tws.std(ddof=0)),
                best_trust=float(tws[best_idx]),
                best_seed=int(rs[best_idx].seed),
                best_config=dict(rs[best_idx].config),
            )
        )

    # Output as a markdown table.
    aggregates.sort(key=lambda r: (-r.trust_mean, r.seconds_mean))
    print()
    print(f"dataset={args.dataset} n={X.shape[0]} d={X.shape[1]} seed={args.seed} pca_dim={args.pca_dim}")
    print(f"seeds={seeds} pca_dim={args.pca_dim} trust_k={args.trust_k} budget_sec={args.time_budget_sec}")
    print()
    print("| model | seconds (mean±std) | trust (mean±std) | best trust | best config |")
    print("|---|---:|---:|---:|---|")
    for r in aggregates:
        secs = f"{r.seconds_mean:.3f}±{r.seconds_std:.3f}"
        tw = f"{r.trust_mean:.5f}±{r.trust_std:.5f}"
        best_cfg = ", ".join(f"{k}={v}" for k, v in r.best_config.items())
        print(f"| {r.name} | {secs} | {tw} | {r.best_trust:.5f} | {best_cfg} |")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


