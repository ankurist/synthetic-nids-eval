#!/usr/bin/env python
"""Fidelity and replication metrics for synthetic data.

Usage:
    python ids_fidelity.py run     --dataset cicids2017 --seeds 0 1 2 3 4 --out_tag _final
    python ids_fidelity.py summary --dataset cicids2017 --out_tag _final

Metrics are computed per (seed, method, class) and for the pooled group ALL.

Fidelity (synthetic rows of a class against real training rows of that class, min-max scaled
with the training range):
  ks_mean     mean two-sample Kolmogorov-Smirnov statistic over numeric columns
  wass_mean   mean one-dimensional Wasserstein distance over numeric columns
  corr_gap    mean absolute difference between real and synthetic Pearson correlations
  tv_cat      mean total-variation distance over categorical columns
  (the real-versus-synthetic probe AUC is computed by `ids_regimes.py probe`)

Replication (numeric columns, scaled space):
  dcr_syn_med, dcr_hold_med  median distance to the closest real training row, for synthetic rows
                             and for unseen real validation rows
  dcr_ratio                  dcr_syn_med / dcr_hold_med; values well below 1 indicate that synthetic
                             rows lie closer to training rows than unseen real rows do
  share_below_p5             share of synthetic rows closer than the 5th percentile of holdout distances
  exact_dup_share            share of synthetic rows at (numerically) zero distance from a training row
  mia_auc                    distance-based membership-inference AUC (0.5 = no measurable leakage)

The training reference set consists of the rows each generator was fitted on (at most
--max_train_rows per class, drawn with the same seed as in ids_synth.py). Distance-based
measures are weak privacy proxies and are reported as replication signals.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

import ids_eval as E
from ids_regimes import load_synth

ROOT = E.ROOT


def path(*p):
    return os.path.join(ROOT, *p)


def rtag(split: str) -> str:
    return "" if split == "random" else f"_{split}"


def scaled(prep: E.Prep, df: pd.DataFrame) -> np.ndarray:
    return np.clip(prep.scaler.transform(df[prep.num].values), 0, 1)


def nn_dist(ref: np.ndarray, q: np.ndarray) -> np.ndarray:
    return NearestNeighbors(n_neighbors=1).fit(ref).kneighbors(q)[0][:, 0]


def fidelity_row(R: np.ndarray, S: np.ndarray, Rc: pd.DataFrame, Sc: pd.DataFrame, cat: list[str]) -> dict:
    ks, ws = [], []
    for j in range(R.shape[1]):
        if np.ptp(R[:, j]) == 0 and np.ptp(S[:, j]) == 0:
            continue
        ks.append(stats.ks_2samp(R[:, j], S[:, j], method="asymp").statistic)
        ws.append(stats.wasserstein_distance(R[:, j], S[:, j]))
    keep = [j for j in range(R.shape[1]) if R[:, j].std() > 0]
    gap = np.nan
    if len(keep) > 1:
        with np.errstate(invalid="ignore", divide="ignore"):  # a column constant in the synthetic data -> 0/0
            cr = np.nan_to_num(np.corrcoef(R[:, keep], rowvar=False))
            cs = np.nan_to_num(
                np.corrcoef(S[:, keep], rowvar=False)
            )  # undefined correlation counts as 0 (dependence lost)
        iu = np.triu_indices(len(keep), 1)
        gap = float(np.mean(np.abs(cr[iu] - cs[iu])))
    tv = []
    for c in cat:
        p = Rc[c].astype(str).value_counts(normalize=True)
        q = Sc[c].astype(str).value_counts(normalize=True)
        idx = p.index.union(q.index)
        tv.append(0.5 * np.abs(p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).sum())
    return dict(
        ks_mean=float(np.mean(ks)) if ks else np.nan,
        wass_mean=float(np.mean(ws)) if ws else np.nan,
        corr_gap=gap,
        tv_cat=float(np.mean(tv)) if tv else np.nan,
    )


def privacy_row(R: np.ndarray, S: np.ndarray, V: np.ndarray, seed: int, min_nonmembers: int = 30) -> dict:
    out = dict(
        dcr_syn_med=np.nan,
        dcr_hold_med=np.nan,
        dcr_ratio=np.nan,
        share_below_p5=np.nan,
        exact_dup_share=np.nan,
        mia_auc=np.nan,
    )
    d_syn = nn_dist(R, S)
    out["dcr_syn_med"] = float(np.median(d_syn))
    out["exact_dup_share"] = float(np.mean(d_syn <= 1e-9))
    if len(V) >= min_nonmembers:
        d_hold = nn_dist(R, V)
        out["dcr_hold_med"] = float(np.median(d_hold))
        out["dcr_ratio"] = out["dcr_syn_med"] / out["dcr_hold_med"] if out["dcr_hold_med"] > 0 else np.nan
        out["share_below_p5"] = float(np.mean(d_syn < np.percentile(d_hold, 5)))
        # membership inference: members = rows the generator was trained on, non-members = unseen real rows
        rng = np.random.default_rng(seed)
        n = min(len(V), len(R))
        M = R[rng.choice(len(R), n, replace=False)]
        Vn = V[rng.choice(len(V), n, replace=False)]
        nn_s = NearestNeighbors(n_neighbors=1).fit(S)
        score = -np.r_[nn_s.kneighbors(M)[0][:, 0], nn_s.kneighbors(Vn)[0][:, 0]]
        out["mia_auc"] = float(roc_auc_score(np.r_[np.ones(n), np.zeros(n)], score))
    return out


def run(dataset: str, split: str, seeds: list[int], methods: list[str], max_rows: int, max_train_rows: int) -> None:
    out = path("results", dataset, "fidelity" + rtag(split) + OUT_TAG)
    os.makedirs(out, exist_ok=True)
    rows = []
    for seed in seeds:
        tr, va, _, schema = E.load_split(dataset, seed, split)
        classes, cat = schema["classes"], list(schema["categorical"])
        for c in cat:
            tr[c] = tr[c].astype(str)
            va[c] = va[c].astype(str)
        tr["category"] = tr["category"].astype(str)
        va["category"] = va["category"].astype(str)
        prep = E.Prep(schema).fit(tr)
        for m in methods:
            try:
                synth = load_synth(dataset, split, seed, m, classes)
            except FileNotFoundError as e:
                print("[skip]", e)
                continue
            for k in synth:
                for c in cat:
                    synth[k][c] = synth[k][c].astype(str)
            parts = {}
            for cls in classes:
                real = tr[tr["category"] == cls]
                fit = real if len(real) <= max_train_rows else real.sample(max_train_rows, random_state=seed)
                parts[cls] = (fit, synth[cls], va[va["category"] == cls])
            parts["ALL"] = tuple(
                pd.concat(
                    [
                        parts[c][i].sample(min(len(parts[c][i]), max(1, max_rows // len(classes))), random_state=seed)
                        for c in classes
                    ],
                    ignore_index=True,
                )
                for i in range(3)
            )
            for g, (fit, S, V) in parts.items():
                rs = fit.sample(min(len(fit), max_rows), random_state=seed)
                ss = S.sample(min(len(S), max_rows), random_state=seed)
                vs = V.sample(min(len(V), max_rows), random_state=seed) if len(V) else V
                R_, S_ = scaled(prep, rs), scaled(prep, ss)
                V_ = scaled(prep, vs) if len(vs) else np.zeros((0, R_.shape[1]))
                ref = scaled(
                    prep, fit.sample(min(len(fit), 20000), random_state=seed)
                )  # reference = generator's training rows
                row = dict(
                    dataset=dataset,
                    split=split,
                    seed=seed,
                    method=m,
                    group=g,
                    n_train=len(fit),
                    n_synth=len(S),
                    n_unseen=len(V),
                )
                row.update(fidelity_row(R_, S_, rs, ss, cat))
                row.update(privacy_row(ref, S_, V_, seed))
                rows.append(row)
            print(f"seed {seed} {m}: done")
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(out, "fidelity_privacy_per_seed.csv"), index=False)
    summary(dataset, split)


def summary(dataset: str, split: str) -> None:
    f = path("results", dataset, "fidelity" + rtag(split) + OUT_TAG, "fidelity_privacy_per_seed.csv")
    d = pd.read_csv(f)
    pd.set_option("display.width", 250)
    cols = ["ks_mean", "wass_mean", "corr_gap", "tv_cat", "dcr_ratio", "share_below_p5", "exact_dup_share", "mia_auc"]
    a = d[d["group"] == "ALL"].groupby("method")[cols].mean().round(4)
    print("\n=== POOLED over all classes (mean over seeds) ===\n", a.to_string())
    for col in ("dcr_ratio", "mia_auc", "corr_gap", "exact_dup_share"):
        t = d[d["group"] != "ALL"].pivot_table(index="group", columns="method", values=col, aggfunc="mean").round(3)
        print(f"\n--- {col} by class ---\n", t.to_string())
    print(
        "\nInterpretation: dcr_ratio << 1, share_below_p5 >> 0.05, exact_dup_share > 0 or mia_auc > 0.55 "
        "indicate the generator reproduces training rows."
    )


OUT_TAG = ""

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="random", choices=["random", "chrono", "pooled"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--methods", nargs="+", default=["smote", "copula", "ctgan_e100", "tvae_e100"])
    ap.add_argument("--max_rows", type=int, default=3000, help="rows per group used for distances (speed)")
    ap.add_argument("--max_train_rows", type=int, default=10000, help="must equal the value used in ids_synth.py")
    ap.add_argument("--out_tag", default="")
    a = ap.parse_args()
    OUT_TAG = a.out_tag
    if a.cmd == "run":
        run(a.dataset, a.split, a.seeds, a.methods, a.max_rows, a.max_train_rows)
    else:
        summary(a.dataset, a.split)
