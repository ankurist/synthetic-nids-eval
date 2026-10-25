#!/usr/bin/env python
"""Per-class synthetic data generators, fitted on training rows only.

Usage:
    python ids_synth.py smoke    --dataset cicids2017                 # short installation and timing check
    python ids_synth.py generate --dataset cicids2017 --seeds 0 1 2 3 4 --methods smote copula
    python ids_synth.py generate --dataset cicids2017 --seeds 0 1 2 3 4 --methods ctgan tvae --epochs 100
    python ids_synth.py generate --dataset cicids2017 --seeds 0 --methods ctgan tvae --epochs 50 300   # training-length ablation
    python ids_synth.py summary  --dataset cicids2017

Protocol:
  * each generator sees only the training rows of one class (never validation or test rows,
    never other classes); at most --max_train_rows rows are used for fitting;
  * per class, gen_multiplier x (real training rows of that class) rows are generated, so the
    regimes can draw 25/50/100% for augmentation (R1) or all rows (R3);
  * outputs are clipped to the class's real per-column range, and integer columns are rounded;
  * one Parquet file per (method, epochs, seed, class); completed files are skipped on re-run;
  * a quality record is logged per file (mean KS against real rows, exact-copy rate, run time).

Methods:
  smote   interpolation between a real row and one of its k nearest same-class neighbours
          (categorical values by neighbour majority)
  copula  Gaussian copula with empirical marginals (Synthetic Data Vault approach)
  ctgan   CTGAN, from the `ctgan` package
  tvae    TVAE, from the `ctgan` package
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time

import numpy as np
import pandas as pd
from scipy import stats

import ids_eval as E

ROOT = E.ROOT


def path(*p):
    return os.path.join(ROOT, *p)


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


# ---------------------------------------------------------------- helpers
def int_columns(X: pd.DataFrame, num: list[str]) -> list[str]:
    return [c for c in num if np.all(np.mod(X[c].values, 1) == 0)]


def auto_discrete(X: pd.DataFrame, num: list[str], cat: list[str], max_card: int = 10) -> list[str]:
    """Columns the GAN/VAE should treat as categories: true categoricals + integer columns with <= max_card values."""
    low = [c for c in num if X[c].nunique() <= max_card and np.all(np.mod(X[c].values, 1) == 0)]
    return list(cat) + low


def postprocess(S: pd.DataFrame, real: pd.DataFrame, num: list[str], cat: list[str]) -> pd.DataFrame:
    """Clip to the real class range, round integer columns, restore dtypes."""
    S = S.copy()
    ints = set(int_columns(real, num))
    for c in num:
        v = pd.to_numeric(S[c], errors="coerce").astype("float64")
        v = v.clip(real[c].min(), real[c].max())
        if c in ints:
            v = v.round()
        S[c] = v.fillna(real[c].median()).astype("float32")
    for c in cat:
        S[c] = S[c].astype(str)
    return S[num + cat]


# ---------------------------------------------------------------- SMOTE (per class, any n)
def gen_smote(X: pd.DataFrame, n: int, seed: int, num: list[str], cat: list[str], k: int = 5, **_) -> pd.DataFrame:
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    m = len(X)
    if m < 2:
        return X.sample(n, replace=True, random_state=seed).reset_index(drop=True)
    A = X[num].to_numpy("float64")
    lo, hi = A.min(0), A.max(0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    S = (A - lo) / span  # neighbour search in [0,1] space
    kk = min(k + 1, m)
    _, idx = NearestNeighbors(n_neighbors=kk).fit(S).kneighbors(S)
    base = rng.integers(0, m, n)
    nbr = idx[base, rng.integers(1, kk, n)]
    lam = rng.random((n, 1))
    out = pd.DataFrame(A[base] + lam * (A[nbr] - A[base]), columns=num)
    for c in cat:
        codes, uniq = pd.factorize(X[c].astype(str))
        neigh = codes[idx[base]]  # (n, kk) codes of the base row and its neighbours
        mode = stats.mode(neigh, axis=1, keepdims=False).mode
        out[c] = uniq[mode]
    return out


# ---------------------------------------------------------------- Gaussian copula (own, SDV-style)
def gen_copula(X: pd.DataFrame, n: int, seed: int, num: list[str], cat: list[str], **_) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = len(X)
    cols, Z, inv = [], [], {}
    const = {}
    for c in num:
        v = X[c].to_numpy("float64")
        if np.ptp(v) == 0:
            const[c] = v[0]
            continue
        Z.append(stats.norm.ppf(stats.rankdata(v) / (m + 1)))
        cols.append(c)
        inv[c] = ("num", np.sort(v))
    for c in cat:
        vc = X[c].astype(str).value_counts(normalize=True)
        if len(vc) == 1:
            const[c] = vc.index[0]
            continue
        cats, p = vc.index.to_numpy(), vc.values
        cum = np.cumsum(p)
        lo = cum - p
        pos = pd.Index(cats).get_indexer(X[c].astype(str))
        u = np.clip(lo[pos] + rng.random(m) * p[pos], 1e-6, 1 - 1e-6)  # random point inside the category's interval
        Z.append(stats.norm.ppf(u))
        cols.append(c)
        inv[c] = ("cat", (cats, cum))
    out = pd.DataFrame(index=range(n))
    if cols:
        C = np.corrcoef(np.column_stack(Z), rowvar=False)
        C = np.atleast_2d(np.nan_to_num(C, nan=0.0))
        w, V = np.linalg.eigh((C + C.T) / 2)  # make it a valid correlation matrix
        C = (V * np.clip(w, 1e-4, None)) @ V.T
        d = np.sqrt(np.diag(C))
        C = C / np.outer(d, d)
        L = np.linalg.cholesky(C + 1e-8 * np.eye(len(cols)))
        U = stats.norm.cdf(rng.standard_normal((n, len(cols))) @ L.T)
        for j, c in enumerate(cols):
            kind, info = inv[c]
            if kind == "num":
                out[c] = np.quantile(info, U[:, j])
            else:
                cats, cum = info
                out[c] = cats[np.minimum(np.searchsorted(cum, U[:, j]), len(cats) - 1)]
    for c, v in const.items():
        out[c] = v
    return out


# ---------------------------------------------------------------- CTGAN / TVAE (package `ctgan`)
def gen_deep(
    X: pd.DataFrame,
    n: int,
    seed: int,
    num: list[str],
    cat: list[str],
    method: str,
    epochs: int,
    min_steps: int = 1500,
    **_,
) -> pd.DataFrame:
    import torch
    from ctgan import CTGAN, TVAE

    torch.manual_seed(seed)
    np.random.seed(seed)
    m = len(X)
    bs = max(10, min(500, (m // 10) * 10))  # CTGAN needs batch size divisible by pac=10
    spe = max(m // bs, 1)
    ep = max(epochs, math.ceil(min_steps / spe))  # tiny classes get at least `min_steps` updates
    const = {c: X[c].iloc[0] for c in num + cat if X[c].nunique() == 1}  # constant-in-this-class columns
    num_f = [c for c in num if c not in const]
    cat_f = [c for c in cat if c not in const]
    disc = auto_discrete(X, num_f, cat_f)
    df = X[num_f + cat_f].copy()
    for c in cat_f:
        df[c] = df[c].astype(str)
    Cls = CTGAN if method == "ctgan" else TVAE
    try:  # ctgan >= 0.10: `enable_gpu` (auto-detects the GPU)
        model = Cls(epochs=ep, batch_size=bs, enable_gpu=True)
    except TypeError:  # older ctgan: `cuda`
        model = Cls(epochs=ep, batch_size=bs, cuda=torch.cuda.is_available())
    if hasattr(model, "set_random_state"):
        model.set_random_state(seed)
    model.fit(df, discrete_columns=disc)
    S = model.sample(n)
    for c, v in const.items():
        S[c] = v
    S.attrs["effective_epochs"] = ep
    return S


GENERATORS = {
    "smote": gen_smote,
    "copula": gen_copula,
    "ctgan": lambda *a, **k: gen_deep(*a, method="ctgan", **k),
    "tvae": lambda *a, **k: gen_deep(*a, method="tvae", **k),
}


# ---------------------------------------------------------------- quality line
def quality(real: pd.DataFrame, synth: pd.DataFrame, num: list[str], cat: list[str], seed: int) -> dict:
    r = real.sample(min(len(real), 5000), random_state=seed)
    s = synth.sample(min(len(synth), 5000), random_state=seed)
    ks = [stats.ks_2samp(r[c], s[c], method="asymp").statistic for c in num if r[c].nunique() > 1]
    key = num + cat
    hr = set(pd.util.hash_pandas_object(real[key].astype({c: "float32" for c in num}), index=False).tolist())
    hs = pd.util.hash_pandas_object(synth[key].astype({c: "float32" for c in num}), index=False)
    return dict(
        mean_ks=float(np.mean(ks)) if ks else float("nan"),
        max_ks=float(np.max(ks)) if ks else float("nan"),
        exact_copy_rate=float(hs.isin(hr).mean()),
    )


# ---------------------------------------------------------------- main loop
def generate(
    dataset: str,
    split: str,
    seeds: list[int],
    methods: list[str],
    epochs_list: list[int],
    classes: list[str] | None,
    gen_multiplier: float,
    max_train_rows: int,
    max_gen_rows: int,
    min_steps: int | None,
) -> None:
    log_path = path("synth", dataset, "" if split == "random" else split, "synth_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    for seed in seeds:
        tr, _, _, schema = E.load_split(dataset, seed, split)
        num, cat = schema["numeric"], list(schema["categorical"])
        num = [c for c in num if tr[c].nunique() > 1]  # train-constant columns carry no information
        for c in cat:
            tr[c] = tr[c].astype(str)
        todo = classes or schema["classes"]
        for method in methods:
            for ep in (epochs_list if method in ("ctgan", "tvae") else [0]):
                tag = method if ep == 0 else f"{method}_e{ep}"
                out_dir = path("synth", dataset, "" if split == "random" else split, f"seed{seed}", tag)
                os.makedirs(out_dir, exist_ok=True)
                for cls in todo:
                    f = os.path.join(out_dir, f"{safe(cls)}.parquet")
                    if os.path.exists(f):
                        print(f"[skip] {tag} seed{seed} {cls} (exists)")
                        continue
                    real = tr[tr["category"].astype(str) == cls]
                    if len(real) == 0:
                        continue
                    n_gen = int(min(max_gen_rows, max(1, round(gen_multiplier * len(real)))))
                    fit_rows = real if len(real) <= max_train_rows else real.sample(max_train_rows, random_state=seed)
                    t0 = time.time()
                    ms = min_steps if min_steps is not None else 15 * ep  # e100 -> 1500 steps (unchanged), e300 -> 4500
                    S = GENERATORS[method](fit_rows[num + cat], n_gen, seed, num, cat, epochs=ep, min_steps=ms)
                    eff = S.attrs.get("effective_epochs", ep)
                    S = postprocess(S, real, num, cat)
                    q = quality(real, S, num, cat, seed)
                    secs = time.time() - t0
                    S["category"] = cls
                    S["binary"] = np.int8(cls != "Benign")
                    S["synthetic"] = np.int8(1)
                    for c in cat:
                        S[c] = S[c].astype("category")
                    S.to_parquet(f, index=False)
                    row = dict(
                        dataset=dataset,
                        split=split,
                        seed=seed,
                        method=tag,
                        cls=cls,
                        n_real=len(real),
                        n_fit=len(fit_rows),
                        n_synth=len(S),
                        effective_epochs=eff,
                        seconds=round(secs, 1),
                        **q,
                    )
                    pd.DataFrame([row]).to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)
                    print(
                        f"[done] {tag:11s} seed{seed} {cls:15s} real {len(real):6d} -> synth {len(S):6d} | "
                        f"KS {q['mean_ks']:.3f} | copies {q['exact_copy_rate']:.3f} | {secs:6.1f}s"
                    )


def summary(dataset: str, split: str) -> None:
    lp = path("synth", dataset, "" if split == "random" else split, "synth_log.csv")
    if not os.path.exists(lp):
        raise SystemExit("no synth_log.csv yet")
    d = pd.read_csv(lp)
    g = (
        d.groupby(["method", "cls"])
        .agg(
            seeds=("seed", "nunique"),
            mean_ks=("mean_ks", "mean"),
            copies=("exact_copy_rate", "mean"),
            sec=("seconds", "mean"),
        )
        .round(3)
    )
    print(g.to_string())
    print("\nTotal generator time (hours):", round(d["seconds"].sum() / 3600, 2))


# ---------------------------------------------------------------- smoke test + time estimate
def smoke(dataset: str, split: str, max_train_rows: int, epochs: int, seeds: int, min_steps: int) -> None:
    tr, _, _, schema = E.load_split(dataset, 0, split)
    num = [c for c in schema["numeric"] if tr[c].nunique() > 1]
    cat = list(schema["categorical"])
    for c in cat:
        tr[c] = tr[c].astype(str)
    sizes = tr["category"].astype(str).value_counts()
    big = sizes.index[0]
    real = tr[tr["category"].astype(str) == big].sample(min(2000, sizes.iloc[0]), random_state=0)
    small = sizes.index[-1]
    realS = tr[tr["category"].astype(str) == small]
    print(f"dataset {dataset}: {len(num)} numeric + {len(cat)} categorical features; classes {dict(sizes)}")
    try:
        import torch

        print("torch", torch.__version__, "| GPU available:", torch.cuda.is_available())
    except ImportError:
        print("PyTorch is not installed; install it with: pip install torch ctgan")
        return
    per_row_epoch = {}
    for method in ("smote", "copula", "ctgan", "tvae"):
        for label, R in (("largest class (2000-row sample)", real), (f"smallest class '{small}'", realS)):
            t0 = time.time()
            S = GENERATORS[method](R[num + cat], 500, 0, num, cat, epochs=3, min_steps=0)
            S = postprocess(S, R, num, cat)
            dt = time.time() - t0
            q = quality(R, S, num, cat, 0)
            print(
                f"  {method:7s} | {label:38s} | {dt:6.1f}s | KS {q['mean_ks']:.3f} | cols ok: {list(S.columns) == num + cat}"
            )
            if method in ("ctgan", "tvae") and R is real:
                per_row_epoch[method] = dt / (len(R) * 3)
    if per_row_epoch:
        print("\nEstimated time for the planned CTGAN/TVAE runs (an upper bound: the short smoke run is dominated by")
        print("one-off setup time, which the estimate scales with the number of epochs).")
        capped = [min(n, max_train_rows) for n in sizes.values]
        for m, k in per_row_epoch.items():
            one = sum(max(r * epochs * k, 1) for r in capped)
            print(
                f"  {m}: ~{one/60:5.1f} min per seed at {epochs} epochs  ->  {one*seeds/3600:4.1f} h for {seeds} seeds"
            )
    print("\nSmoke test complete. Use `generate` for the full runs.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["smoke", "generate", "summary"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="random", choices=["random", "chrono", "pooled"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--methods", nargs="+", default=["smote", "copula"], choices=list(GENERATORS))
    ap.add_argument("--epochs", type=int, nargs="+", default=[100])
    ap.add_argument("--classes", nargs="+")
    ap.add_argument("--gen_multiplier", type=float, default=1.0)
    ap.add_argument("--max_train_rows", type=int, default=10000)
    ap.add_argument("--max_gen_rows", type=int, default=60000)
    ap.add_argument(
        "--min_steps", type=int, default=None, help="minimum optimiser steps for tiny classes (default: 15 x epochs)"
    )
    a = ap.parse_args()
    if a.cmd == "smoke":
        smoke(
            a.dataset,
            a.split,
            a.max_train_rows,
            a.epochs[0],
            len(a.seeds) if len(a.seeds) > 1 else 5,
            a.min_steps or 1500,
        )
    elif a.cmd == "generate":
        generate(
            a.dataset,
            a.split,
            a.seeds,
            a.methods,
            a.epochs,
            a.classes,
            a.gen_multiplier,
            a.max_train_rows,
            a.max_gen_rows,
            a.min_steps,
        )
    else:
        summary(a.dataset, a.split)
