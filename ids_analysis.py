#!/usr/bin/env python
"""Summary tables and diagnostic figures computed from saved result files.

Usage:
    python ids_analysis.py --root $IDS_ROOT --out_tag _final --outdir analysis_out
    python ids_analysis.py --indir <folder with the CSV files> --outdir analysis_out

Inputs per dataset: regimes_per_seed.csv (ids_regimes.py run), fidelity_privacy_per_seed.csv
(ids_fidelity.py run) and probe_per_seed.csv (ids_regimes.py probe).

Outputs (CSV tables and PNG figures):
  T1  binary results per regime with paired differences to R0 (95% t-interval over seeds)
  T2  multi-class augmentation (R1) against R0 and against duplication (R1_ROS)
  T3  fidelity and replication per generator
  T4  generator-level rank agreement between fidelity and utility (Kendall tau)
  T5  class-level correlation with cluster-bootstrap intervals
  F1  regime differences   F2  fidelity against utility   F3  replication against utility
The tables reported in the paper are reproduced by reproduce_tables.py.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

NAMES = {"smote": "SMOTE", "copula": "Copula", "ctgan_e100": "CTGAN", "tvae_e100": "TVAE", "real": "real"}
FILES = {
    "CIC": ("regimes_per_seed_icids.csv", "fidelity_privacy_per_seed_cicids.csv", "probe_per_seed_cicids.csv"),
    "UNSW": ("regimes_per_seed_uncw.csv", "fidelity_privacy_per_seed_unsw.csv", "probe_per_seed_unsw.csv"),
}
DIRS = {"CIC": "cicids2017", "UNSW": "unsw_nb15"}
RNG = np.random.default_rng(0)


def tci(x) -> float:
    x = np.asarray(pd.Series(x).dropna(), float)
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def pm(x, nd=3) -> str:
    x = pd.Series(x).dropna()
    return f"{x.mean():.{nd}f} ± {tci(x):.{nd}f}"


def load(ds: str, a) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if a.root:
        d = os.path.join(a.root, "results", DIRS[ds])
        paths = (
            os.path.join(d, "regimes" + a.out_tag, "regimes_per_seed.csv"),
            os.path.join(d, "fidelity" + a.out_tag, "fidelity_privacy_per_seed.csv"),
            os.path.join(d, "probe" + a.out_tag, "probe_per_seed.csv"),
        )
    else:
        paths = tuple(os.path.join(a.indir, f) for f in FILES[ds])
    r, f, p = (pd.read_csv(x) for x in paths)
    for t in (r, f, p):
        t["method"] = t["method"].map(lambda m: NAMES.get(m, m))
    return r, f, p


# ------------------------------------------------------------------ T1 / T2: regimes
def with_deltas(r: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    base = r[r["regime"] == "R0"].set_index(["seed", "model", "task"])
    idx = pd.MultiIndex.from_frame(r[["seed", "model", "task"]])
    r = r.copy()
    for c in cols:
        r["d_" + c] = r[c].values - base[c].reindex(idx).values
    return r


def t1(r: pd.DataFrame, ds: str) -> pd.DataFrame:
    cols = [c for c in ("macro_f1", "fpr", "rec", "pr_auc") if c in r.columns]
    b = with_deltas(r[r["task"] == "binary"], cols)
    rows = []
    for (m, rg, p, mo), g in b.groupby(["method", "regime", "param", "model"]):
        if rg == "R1" and p not in (0.25, 1.0):
            continue
        row = dict(dataset=ds, method=m, regime=rg, param=p, model=mo, n_seeds=g["seed"].nunique())
        for c in cols:
            row[c] = pm(g[c], 4)
            row["d_" + c] = pm(g["d_" + c], 4)
            row[c + "_mean"], row["d_" + c + "_mean"], row["d_" + c + "_ci"] = (
                g[c].mean(),
                g["d_" + c].mean(),
                tci(g["d_" + c]),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def t2(r: pd.DataFrame, ds: str) -> pd.DataFrame:
    mc = r[r["task"] == "multiclass"]
    rec = [c for c in mc.columns if c.startswith("recall_")]
    cols = ["macro_f1"] + rec
    mc = with_deltas(mc, cols)
    ros = mc[(mc["method"] == "real") & (mc["regime"] == "R1_ROS") & (mc["param"] == 1.0)].set_index(["seed", "model"])
    rows = []
    for m in ("real", "SMOTE", "Copula", "CTGAN", "TVAE"):
        sub = mc[
            (mc["regime"] == ("R1_ROS" if m == "real" else "R1"))
            & (mc["param"] == 1.0)
            & (mc["method"] == ("real" if m == "real" else m))
        ]
        for mo, g in sub.groupby("model"):
            g = g.set_index("seed")
            for c in cols:
                d_ros = g[c].values - np.array([ros.loc[(s, mo), c] for s in g.index])
                rows.append(
                    dict(
                        dataset=ds,
                        method="ROS (duplication)" if m == "real" else m,
                        model=mo,
                        metric=c,
                        delta_vs_R0=g["d_" + c].mean(),
                        ci_vs_R0=tci(g["d_" + c]),
                        delta_vs_ROS=(np.nan if m == "real" else d_ros.mean()),
                        ci_vs_ROS=(np.nan if m == "real" else tci(d_ros)),
                    )
                )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ T3: fidelity + privacy
def t3(f: pd.DataFrame, p: pd.DataFrame, ds: str) -> pd.DataFrame:
    a = f[f["group"] == "ALL"]
    cols = ["ks_mean", "wass_mean", "corr_gap", "tv_cat", "dcr_ratio", "share_below_p5", "exact_dup_share", "mia_auc"]
    pr = p[p["group"] != "ALL_ATTACKS"].groupby(["method", "seed"]).probe_auc.mean().rename("probe_auc").reset_index()
    rows = []
    for m, g in a.groupby("method"):
        row = dict(dataset=ds, method=m)
        for c in cols:
            row[c] = pm(g[c], 4) if g[c].notna().any() else ""
        row["probe_auc"] = pm(pr[pr["method"] == m]["probe_auc"], 3)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ T4 / T5: fidelity-utility rank agreement
def generator_table(r, f, p, ds) -> pd.DataFrame:
    r3 = (
        r[(r["regime"] == "R3") & (r["task"] == "binary")]
        .groupby("method")[["macro_f1", "pr_auc"]]
        .mean()
        .add_prefix("R3_")
    )
    fid = f[f["group"] == "ALL"].groupby("method")[["ks_mean", "wass_mean", "corr_gap", "mia_auc", "dcr_ratio"]].mean()
    pr = p[p["group"] != "ALL_ATTACKS"].groupby("method").probe_auc.mean()
    t = fid.join(pr).join(r3)
    t.insert(0, "dataset", ds)
    return t.reset_index()


def tau_table(G: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, g in G.groupby("dataset"):
        for fm in ("ks_mean", "wass_mean", "corr_gap", "probe_auc"):
            for um in ("R3_macro_f1", "R3_pr_auc"):
                tau = stats.kendalltau(-g[fm], g[um]).statistic
                rows.append(
                    dict(dataset=ds, fidelity_metric=fm, utility=um, kendall_tau=round(tau, 2), n_generators=len(g))
                )
    return pd.DataFrame(rows)


def class_units(r, f, p, ds) -> pd.DataFrame:
    f = f[(f["group"] != "ALL") & (f["n_unseen"] >= 30)]  # drop classes with too few unseen rows (e.g. Worms)
    fid = f.groupby(["method", "group"])[["ks_mean", "wass_mean", "corr_gap"]].mean()
    pr = p[p["group"] != "ALL_ATTACKS"].groupby(["method", "group"]).probe_auc.mean()
    mc = r[(r["regime"] == "R3") & (r["task"] == "multiclass")]
    rc = [c for c in mc.columns if c.startswith("recall_")]
    u = mc.groupby("method")[rc].mean().stack().rename("utility")
    u.index = u.index.set_names(["method", "group"])
    u = u.rename(lambda x: x.replace("recall_", ""), level="group")
    t = fid.join(pr).join(u).dropna().reset_index()
    t["dataset"] = ds
    return t


def boot_rho(sub: pd.DataFrame, fm: str, B: int = 4000) -> tuple[float, float]:
    groups = [g for _, g in sub.groupby(sub["dataset"] + "_" + sub["group"])]
    vals = []
    for _ in range(B):
        s = pd.concat([groups[i] for i in RNG.integers(0, len(groups), len(groups))])
        if s[fm].nunique() > 1 and s["utility"].nunique() > 1:
            vals.append(stats.spearmanr(-s[fm], s["utility"]).statistic)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def t5(U: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("CIC", "UNSW", "ALL"):
        sub = U if scope == "ALL" else U[U["dataset"] == scope]
        for fm in ("ks_mean", "wass_mean", "corr_gap", "probe_auc"):
            lo, hi = boot_rho(sub, fm)
            rows.append(
                dict(
                    scope=scope,
                    level="pooled over generators",
                    fidelity_metric=fm,
                    spearman=round(stats.spearmanr(-sub[fm], sub["utility"]).statistic, 2),
                    ci_lo=round(lo, 2),
                    ci_hi=round(hi, 2),
                    n=len(sub),
                )
            )
    for ds in ("CIC", "UNSW"):
        for m, g in U[U["dataset"] == ds].groupby("method"):
            for fm in ("ks_mean", "wass_mean", "corr_gap", "probe_auc"):
                rows.append(
                    dict(
                        scope=ds,
                        level=f"within {m}",
                        fidelity_metric=fm,
                        spearman=round(stats.spearmanr(-g[fm], g["utility"]).statistic, 2),
                        ci_lo=np.nan,
                        ci_hi=np.nan,
                        n=len(g),
                    )
                )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def figures(R: dict, G: pd.DataFrame, outdir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["SMOTE", "Copula", "TVAE", "CTGAN"]
    # F1: paired macro-F1 deltas of R2 / R2r / R3, averaged over RF and XGB, CI over seeds
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    for a, (ds, r) in zip(ax, R.items()):
        b = with_deltas(r[r["task"] == "binary"], ["macro_f1"])
        for k, rg in enumerate(("R2", "R2r", "R3")):
            vals, errs = [], []
            for m in order:
                g = b[(b["method"] == m) & (b["regime"] == rg)].groupby("seed")["d_macro_f1"].mean()
                vals.append(g.mean())
                errs.append(tci(g))
            a.bar(np.arange(4) + (k - 1) * 0.26, vals, 0.26, yerr=errs, capsize=2, label=rg)
        a.axhline(0, color="k", lw=0.8)
        a.set_xticks(range(4))
        a.set_xticklabels(order)
        a.set_title(ds)
        a.set_ylabel("change in macro-F1 vs real-only (R0)")
    ax[0].legend(title="training data", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "F1_regime_deltas.png"), dpi=200)
    plt.close(fig)
    # F2: fidelity metric vs utility (generator level)
    mets = [
        ("ks_mean", "mean KS"),
        ("wass_mean", "mean Wasserstein"),
        ("corr_gap", "correlation gap"),
        ("probe_auc", "real-vs-synthetic probe AUC"),
    ]
    fig, ax = plt.subplots(2, 4, figsize=(14, 6))
    for i, ds in enumerate(G["dataset"].unique()):
        g = G[G["dataset"] == ds]
        for j, (fm, lab) in enumerate(mets):
            a = ax[i, j]
            a.scatter(g[fm], g["R3_pr_auc"], s=45)
            for _, row in g.iterrows():
                a.annotate(
                    row["method"], (row[fm], row["R3_pr_auc"]), textcoords="offset points", xytext=(4, 4), fontsize=8
                )
            a.set_xlabel(lab)
            a.set_ylabel(f"{ds}: PR-AUC trained on synthetic only" if j == 0 else "")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "F2_fidelity_vs_utility.png"), dpi=200)
    plt.close(fig)
    # F3: privacy vs utility
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, ds in zip(ax, G["dataset"].unique()):
        g = G[G["dataset"] == ds]
        a.scatter(g["mia_auc"], g["R3_macro_f1"], s=45)
        for _, row in g.iterrows():
            a.annotate(
                row["method"],
                (row["mia_auc"], row["R3_macro_f1"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
        a.axvline(0.5, color="grey", ls=":")
        a.set_title(ds)
        a.set_xlabel("membership-inference AUC (0.5 = no leakage)")
        a.set_ylabel("macro-F1 trained on synthetic only")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "F3_privacy_vs_utility.png"), dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="project data folder (contains results/)")
    ap.add_argument("--indir", default=".", help="folder holding the six CSVs when --root is not used")
    ap.add_argument("--out_tag", default="_final")
    ap.add_argument("--outdir", default="analysis_out")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 300)
    R, T1, T2, T3, GEN, UN = {}, [], [], [], [], []
    for ds in ("CIC", "UNSW"):
        r, f, p = load(ds, a)
        R[ds] = r
        T1.append(t1(r, ds))
        T2.append(t2(r, ds))
        T3.append(t3(f, p, ds))
        GEN.append(generator_table(r, f, p, ds))
        UN.append(class_units(r, f, p, ds))
    T1, T2, T3, G, U = (
        pd.concat(T1),
        pd.concat(T2),
        pd.concat(T3),
        pd.concat(GEN, ignore_index=True),
        pd.concat(UN, ignore_index=True),
    )
    T4, T5 = tau_table(G), t5(U)
    for name, t in (
        ("T1_binary_regimes", T1),
        ("T2_multiclass_R1", T2),
        ("T3_fidelity_privacy", T3),
        ("T4_generator_level_rank_agreement", T4),
        ("T5_class_level_correlation", T5),
        ("generator_table", G),
    ):
        t.to_csv(os.path.join(a.outdir, name + ".csv"), index=False)
    figures(R, G, a.outdir)

    show = T1[(T1["regime"].isin(["R0", "R1", "R1_ROS", "R2", "R2r", "R3"]))]
    print("\n=== T1 binary (macro-F1 and paired change vs R0, mean ± 95% CI over seeds) ===")
    print(
        show[["dataset", "method", "regime", "param", "model", "macro_f1", "d_macro_f1", "fpr", "pr_auc"]].to_string(
            index=False
        )
    )
    print("\n=== T3 fidelity and privacy (pooled over classes) ===\n", T3.to_string(index=False))
    print(
        "\n=== T4 generator-level rank agreement (Kendall tau; 4 generators per dataset) ===\n",
        T4.to_string(index=False),
    )
    print("\n=== T5 class-level correlation ===\n", T5.to_string(index=False))
    print(f"\nWrote tables and figures to {a.outdir}/")


if __name__ == "__main__":
    main()
