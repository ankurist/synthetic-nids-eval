#!/usr/bin/env python
"""Control experiments for the provenance analysis and robustness checks.

Commands:
  rmix        within-class mixing control (RMIX): every class is half real and half synthetic,
              so provenance carries no label information
  provenance  scores held-out synthetic benign rows with the R2 detector and held-out synthetic
              attack rows with the R2r detector
  threshold   R0, R2 and R3 with a decision threshold tuned on the real validation split
              (maximum macro-F1), plus class-weighted real-only baselines
  mlp         MLP detector for R0, R2, R3 (binary) and R0, R1_ROS, R1 (multi-class)
  bootstrap   hierarchical bootstrap (seeds x test rows) of macro-F1 differences against R0,
              plus per-class F1 with test supports, from saved predictions
  floor       share of test rows whose feature vector occurs in training under a different label

Usage:
    python ids_extra.py rmix       --dataset unsw_nb15 --seeds 0 1 2 3 4 --xgb_gpu
    python ids_extra.py provenance --dataset unsw_nb15 --seeds 0 1 2 3 4 --xgb_gpu
    python ids_extra.py threshold  --dataset unsw_nb15 --seeds 0 1 2 3 4 --xgb_gpu
    python ids_extra.py mlp        --dataset unsw_nb15 --seeds 0 1 2
    python ids_extra.py bootstrap  --dataset unsw_nb15 --out_tag _final
    python ids_extra.py floor      --dataset unsw_nb15

Each command writes CSV files to results/<dataset>/extra<tag>/; re-runs replace matching rows.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

import ids_eval as E
from ids_regimes import load_synth, minority_classes, rtag

GENS = ["smote", "copula", "ctgan_e100", "tvae_e100"]


def path(*p):
    return os.path.join(E.ROOT, *p)


def outdir(dataset, split, tag):
    d = path("results", dataset, "extra" + rtag(split) + tag)
    os.makedirs(d, exist_ok=True)
    return d


def append(csv, rows, keys=("seed", "method", "regime", "model", "task")):
    """Safe append: re-reads the file, aligns columns by NAME, drops exact re-run duplicates (keeps the
    latest), and rewrites the whole file. Rows with different column sets can no longer be shifted."""
    new = pd.DataFrame(rows)
    if os.path.exists(csv):
        new = pd.concat([pd.read_csv(csv), new], ignore_index=True, sort=False)
    k = [c for c in keys if c in new.columns]
    if k:
        new = new.drop_duplicates(subset=k, keep="last")
    new.to_csv(csv, index=False)


def prepare(dataset, seed, split):
    tr, va, te, schema = E.load_split(dataset, seed, split)
    cat = list(schema["categorical"])
    for df in (tr, va, te):
        for c in cat:
            df[c] = df[c].astype(str)
        df["category"] = df["category"].astype(str)
    prep = E.Prep(schema).fit(tr)
    keep = prep.names + ["category", "binary"]
    return tr[keep], va, te, schema, prep, keep


def synth_for(dataset, split, seed, method, classes, cat, keep):
    s = load_synth(dataset, split, seed, method, classes)
    for k in s:
        for c in cat:
            s[k][c] = s[k][c].astype(str)
        s[k]["category"] = s[k]["category"].astype(str)
        s[k] = s[k][keep]
    return s


def binary_model(name, seed, weighted=False, ytr=None):
    if name == "mlp":
        return "MLP", MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=200, random_state=seed)
    n, clf = E.make_binary(name, seed)
    if weighted and name == "rf":
        clf.set_params(class_weight="balanced")
    return n, clf


def fit(clf, X, y, weighted=False):
    if weighted and not isinstance(clf, RandomForestClassifier):
        clf.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        clf.fit(X, y)
    return clf


def tuned_threshold(y_val, p_val):
    grid = np.unique(np.quantile(p_val, np.linspace(0.005, 0.995, 199)))
    scores = [f1_score(y_val, (p_val >= t).astype(int), average="macro", zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def gpu(flag):
    if flag:
        E.XGB_DEVICE = "cuda"


# ------------------------------------------------------------------ rmix
def cmd_rmix(a):
    out = outdir(a.dataset, a.split, a.out_tag)
    csv = os.path.join(out, "rmix_per_seed.csv")
    for seed in a.seeds:
        real, va, te, schema, prep, keep = prepare(a.dataset, seed, a.split)
        Xte = prep.transform(te)
        yte = te["binary"].values
        cte = te["category"].values
        classes = schema["classes"]
        cidx = {c: i for i, c in enumerate(classes)}
        rng = np.random.default_rng(seed + 77)
        for m in a.methods:
            try:
                syn = synth_for(a.dataset, a.split, seed, m, classes, list(schema["categorical"]), keep)
            except FileNotFoundError as e:
                print("[skip]", e)
                continue
            parts = []
            for c in classes:
                rc = real[real["category"] == c]
                n = len(rc)
                h = n // 2
                parts.append(rc.sample(n - h, random_state=int(rng.integers(1 << 30))))
                parts.append(syn[c].sample(h, replace=h > len(syn[c]), random_state=int(rng.integers(1 << 30))))
            mix = pd.concat(parts, ignore_index=True)
            Xtr = prep.transform(mix)
            rows = []
            for mo in a.models:
                n_, clf = E.make_binary(mo, seed)
                fit(clf, Xtr, mix["binary"].values)
                p = clf.predict_proba(Xte)[:, 1]
                rows.append(
                    dict(
                        seed=seed,
                        method=m,
                        regime="RMIX",
                        model=n_,
                        task="binary",
                        **E.binary_metrics(yte, (p >= 0.5).astype(int), p),
                    )
                )
                n_, clf = E.make_multi(mo, seed)
                fit(clf, Xtr, np.array([cidx[c] for c in mix["category"]]))
                pm = np.array(classes)[clf.predict(Xte).astype(int)]
                rows.append(
                    dict(
                        seed=seed,
                        method=m,
                        regime="RMIX",
                        model=n_,
                        task="multiclass",
                        **E.multiclass_metrics(cte, pm, classes),
                    )
                )
            append(csv, rows)
            b = [r for r in rows if r["task"] == "binary"]
            print(
                f"seed {seed} {m:11s} RMIX | "
                + " | ".join(f"{r['model']} macroF1 {r['macro_f1']:.4f} FPR {r['fpr']:.4f}" for r in b)
            )
    print("wrote", csv)


# ------------------------------------------------------------------ provenance
def cmd_provenance(a):
    out = outdir(a.dataset, a.split, a.out_tag)
    csv = os.path.join(out, "provenance_per_seed.csv")
    for seed in a.seeds:
        real, va, te, schema, prep, keep = prepare(a.dataset, seed, a.split)
        classes = schema["classes"]
        Xte = prep.transform(te)
        yte = te["binary"].values
        rng = np.random.default_rng(seed + 99)
        for m in a.methods:
            try:
                syn = synth_for(a.dataset, a.split, seed, m, classes, list(schema["categorical"]), keep)
            except FileNotFoundError as e:
                print("[skip]", e)
                continue
            # hold out 20% of every synthetic class; those rows are never used for fitting
            hold, used = {}, {}
            for c in classes:
                idx = rng.permutation(len(syn[c]))
                k = int(0.2 * len(idx))
                hold[c], used[c] = syn[c].iloc[idx[:k]], syn[c].iloc[idx[k:]]
            ben_real = real[real["category"] == "Benign"]
            att_real = real[real["category"] != "Benign"]
            syn_att_used = pd.concat([used[c] for c in classes if c != "Benign"])
            syn_att_hold = pd.concat([hold[c] for c in classes if c != "Benign"])
            designs = {"R2": pd.concat([ben_real, syn_att_used]), "R2r": pd.concat([used["Benign"], att_real])}
            rows = []
            for rg, mix in designs.items():
                Xtr = prep.transform(mix)
                for mo in a.models:
                    n_, clf = E.make_binary(mo, seed)
                    fit(clf, Xtr, mix["binary"].values)

                    def pr(df, _clf=clf, _prep=prep):
                        return _clf.predict_proba(_prep.transform(df))[:, 1]

                    p_te = clf.predict_proba(Xte)[:, 1]
                    row = dict(
                        seed=seed,
                        method=m,
                        regime=rg,
                        model=n_,
                        real_attack_recall=float(np.mean(p_te[yte == 1] >= 0.5)),
                        real_benign_fpr=float(np.mean(p_te[yte == 0] >= 0.5)),
                        heldout_syn_attack_recall=float(np.mean(pr(syn_att_hold) >= 0.5)),
                        heldout_syn_benign_flagged=float(np.mean(pr(hold["Benign"]) >= 0.5)),
                    )
                    rows.append(row)
            append(csv, rows, keys=("seed", "method", "regime", "model"))
            for r in rows:
                print(
                    f"seed {seed} {m:11s} {r['regime']:3s} {r['model']:3s} | real attacks caught {r['real_attack_recall']:.3f} | "
                    f"real benign flagged {r['real_benign_fpr']:.3f} | synthetic attacks caught {r['heldout_syn_attack_recall']:.3f} | "
                    f"synthetic BENIGN flagged as attack {r['heldout_syn_benign_flagged']:.3f}"
                )
    print("wrote", csv)
    print(
        "Interpretation: a high rate of held-out synthetic benign rows flagged by the R2 detector indicates reliance on provenance."
    )


# ------------------------------------------------------------------ threshold + class weights
def cmd_threshold(a):
    out = outdir(a.dataset, a.split, a.out_tag)
    csv = os.path.join(out, "threshold_per_seed.csv")
    for seed in a.seeds:
        real, va, te, schema, prep, keep = prepare(a.dataset, seed, a.split)
        classes = schema["classes"]
        cidx = {c: i for i, c in enumerate(classes)}
        Xva, Xte = prep.transform(va), prep.transform(te)
        yva, yte = va["binary"].values, te["binary"].values
        jobs = [("real", "R0", real, False), ("real", "R0_weighted", real, True)]
        for m in a.methods:
            try:
                syn = synth_for(a.dataset, a.split, seed, m, classes, list(schema["categorical"]), keep)
            except FileNotFoundError as e:
                print("[skip]", e)
                continue
            jobs.append(
                (
                    m,
                    "R2",
                    pd.concat([real[real["category"] == "Benign"]] + [syn[c] for c in classes if c != "Benign"]),
                    False,
                )
            )
            jobs.append((m, "R3", pd.concat([syn[c] for c in classes]), False))
        rows = []
        for m, rg, mix, w in jobs:
            Xtr = prep.transform(mix)
            for mo in a.models:
                n_, clf = binary_model(mo, seed, weighted=w)
                fit(clf, Xtr, mix["binary"].values, weighted=w)
                pv, pt = clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]
                t = tuned_threshold(yva, pv)
                d = E.binary_metrics(yte, (pt >= t).astype(int), pt)
                d0 = E.binary_metrics(yte, (pt >= 0.5).astype(int), pt)
                rows.append(
                    dict(
                        seed=seed,
                        method=m,
                        regime=rg,
                        model=n_,
                        task="binary",
                        threshold=t,
                        macro_f1_default=d0["macro_f1"],
                        fpr_default=d0["fpr"],
                        rec_default=d0["rec"],
                        macro_f1_tuned=d["macro_f1"],
                        fpr_tuned=d["fpr"],
                        rec_tuned=d["rec"],
                        pr_auc=d["pr_auc"],
                    )
                )
            if rg == "R0_weighted":
                for mo in a.models:
                    n_, clf = E.make_multi(mo, seed)
                    if mo == "rf":
                        clf.set_params(class_weight="balanced")
                    ym = np.array([cidx[c] for c in mix["category"]])
                    fit(clf, Xtr, ym, weighted=(mo != "rf"))
                    pm = np.array(classes)[clf.predict(Xte).astype(int)]
                    rows.append(
                        dict(
                            seed=seed,
                            method=m,
                            regime=rg,
                            model=n_,
                            task="multiclass",
                            **E.multiclass_metrics(te["category"].values, pm, classes),
                        )
                    )
        append(csv, rows)
        for r in rows:
            if r["task"] == "binary":
                print(
                    f"seed {seed} {r['method']:11s} {r['regime']:11s} {r['model']:3s} | macro-F1 default {r['macro_f1_default']:.4f} -> tuned {r['macro_f1_tuned']:.4f} (t={r['threshold']:.3f})"
                )
    print("wrote", csv)


# ------------------------------------------------------------------ MLP
def cmd_mlp(a):
    out = outdir(a.dataset, a.split, a.out_tag)
    csv = os.path.join(out, "mlp_per_seed.csv")
    for seed in a.seeds:
        real, va, te, schema, prep, keep = prepare(a.dataset, seed, a.split)
        classes = schema["classes"]
        cidx = {c: i for i, c in enumerate(classes)}
        Xte = prep.transform(te)
        yte, cte = te["binary"].values, te["category"].values
        minority = minority_classes(real, a.minority_frac)
        jobs = [
            ("real", "R0", real),
            (
                "real",
                "R1_ROS",
                pd.concat(
                    [real]
                    + [real[real["category"] == c].sample(frac=1.0, replace=True, random_state=seed) for c in minority]
                ),
            ),
        ]
        for m in a.methods:
            try:
                syn = synth_for(a.dataset, a.split, seed, m, classes, list(schema["categorical"]), keep)
            except FileNotFoundError as e:
                print("[skip]", e)
                continue
            n_c = real["category"].value_counts()
            jobs.append(
                (
                    m,
                    "R1",
                    pd.concat(
                        [real]
                        + [
                            syn[c].sample(int(n_c[c]), replace=n_c[c] > len(syn[c]), random_state=seed)
                            for c in minority
                        ]
                    ),
                )
            )
            jobs.append(
                (m, "R2", pd.concat([real[real["category"] == "Benign"]] + [syn[c] for c in classes if c != "Benign"]))
            )
            jobs.append((m, "R3", pd.concat([syn[c] for c in classes])))
        for m, rg, mix in jobs:
            t0 = time.time()
            Xtr = prep.transform(mix)
            rows = []
            if rg in ("R0", "R2", "R3"):
                clf = MLPClassifier(
                    hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=200, random_state=seed
                ).fit(Xtr, mix["binary"].values)
                p = clf.predict_proba(Xte)[:, 1]
                rows.append(
                    dict(
                        seed=seed,
                        method=m,
                        regime=rg,
                        model="MLP",
                        task="binary",
                        **E.binary_metrics(yte, (p >= 0.5).astype(int), p),
                    )
                )
            if rg in ("R0", "R1_ROS", "R1"):
                clf = MLPClassifier(
                    hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=200, random_state=seed
                ).fit(Xtr, np.array([cidx[c] for c in mix["category"]]))
                pm = np.array(classes)[clf.predict(Xte).astype(int)]
                rows.append(
                    dict(
                        seed=seed,
                        method=m,
                        regime=rg,
                        model="MLP",
                        task="multiclass",
                        **E.multiclass_metrics(cte, pm, classes),
                    )
                )
            append(csv, rows)
            print(
                f"seed {seed} {m:11s} {rg:6s} MLP | "
                + " | ".join(f"{r['task']} macroF1 {r['macro_f1']:.4f}" for r in rows)
                + f" | {time.time()-t0:.0f}s"
            )
    print("wrote", csv)


# ------------------------------------------------------------------ bootstrap + per-class F1 from saved predictions
def cmd_bootstrap(a):
    pdir = path("results", a.dataset, "regimes" + rtag(a.split) + a.out_tag, "preds")
    out = outdir(a.dataset, a.split, a.out_tag)
    files = sorted(f for f in os.listdir(pdir) if f.endswith(".npz"))
    load = lambda f: np.load(os.path.join(pdir, f), allow_pickle=True)
    seeds = sorted({int(f.split("seed")[-1].split(".")[0]) for f in files})
    rng = np.random.default_rng(0)
    rows = []
    keys = sorted({f.rsplit("__seed", 1)[0] for f in files})
    for key in keys:
        method, rest = key.split("__")
        if rest.startswith("R0_"):
            continue
        for mo in a.boot_models:
            pairs = []
            for s in seeds:
                fa, fb = f"{key}__seed{s}.npz", f"real__R0_0.0__seed{s}.npz"
                if fa in files and fb in files:
                    za, zb = load(fa), load(fb)
                    if f"{mo}_binary_pred" in za and f"{mo}_binary_pred" in zb:
                        pairs.append((za["y"], za[f"{mo}_binary_pred"], zb[f"{mo}_binary_pred"]))
            if not pairs:
                continue
            diffs = []
            for _ in range(a.B):
                y, pa, pb = pairs[rng.integers(len(pairs))]
                i = rng.integers(0, len(y), len(y))
                diffs.append(f1_score(y[i], pa[i], average="macro") - f1_score(y[i], pb[i], average="macro"))
            point = np.mean(
                [f1_score(y, pa, average="macro") - f1_score(y, pb, average="macro") for y, pa, pb in pairs]
            )
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            rows.append(
                dict(
                    method=method,
                    regime=rest,
                    model=mo,
                    task="binary",
                    n_seeds=len(pairs),
                    delta_macro_f1=point,
                    boot_lo=lo,
                    boot_hi=hi,
                )
            )
    pd.DataFrame(rows).to_csv(os.path.join(out, "bootstrap_binary_deltas.csv"), index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    # per-class F1 with supports
    pcs = []
    for key in [k for k in keys if k.startswith("real__R0_") or "R1_1.0" in k or "R1_ROS_1.0" in k]:
        for s in seeds:
            f = f"{key}__seed{s}.npz"
            if f not in files:
                continue
            z = load(f)
            for mo in a.boot_models:
                if f"{mo}_multi_pred" not in z:
                    continue
                y, p = z["cat"], z[f"{mo}_multi_pred"]
                labels = sorted(set(y))
                f1s = f1_score(y, p, labels=labels, average=None, zero_division=0)
                for c, v in zip(labels, f1s):
                    pcs.append(dict(key=key, seed=s, model=mo, cls=c, f1=v, support=int(np.sum(y == c))))
    pc = pd.DataFrame(pcs)
    if len(pc):
        pc.to_csv(os.path.join(out, "per_class_f1_per_seed.csv"), index=False)
        print(
            pc.groupby(["key", "model", "cls"])
            .agg(f1=("f1", "mean"), support=("support", "first"))
            .round(3)
            .to_string()
        )
    print("wrote", out)


# ------------------------------------------------------------------ label-conflict floor
def cmd_floor(a):
    tr, va, te, schema = E.load_split(a.dataset, a.seeds[0], a.split)
    cols = [c for c in schema["numeric"]] + list(schema["categorical"])
    h = lambda df: pd.util.hash_pandas_object(df[cols].astype({c: str for c in schema["categorical"]}), index=False)
    htr, hte = h(tr), h(te)
    lab = pd.DataFrame({"h": htr.values, "c": tr["category"].astype(str).values}).groupby("h")["c"].agg(set)
    te_lab = te["category"].astype(str).values
    in_train = hte.isin(lab.index).values
    conflict = np.array([(hh in lab.index) and (lab[hh] != {c}) for hh, c in zip(hte.values, te_lab)])
    res = dict(
        dataset=a.dataset,
        test_rows=len(te),
        share_test_rows_seen_in_train=float(in_train.mean()),
        share_test_rows_with_other_train_label=float(conflict.mean()),
    )
    out = outdir(a.dataset, a.split, a.out_tag)
    pd.DataFrame([res]).to_csv(os.path.join(out, "label_conflict_floor.csv"), index=False)
    print(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["rmix", "provenance", "threshold", "mlp", "bootstrap", "floor"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="random", choices=["random", "chrono", "pooled"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--methods", nargs="+", default=GENS)
    ap.add_argument("--models", nargs="+", default=["rf", "xgb"], choices=["rf", "xgb"])
    ap.add_argument("--boot_models", nargs="+", default=["RF", "XGB"])
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--minority_frac", type=float, default=0.03)
    ap.add_argument("--xgb_gpu", action="store_true")
    ap.add_argument("--out_tag", default="_final")
    a = ap.parse_args()
    gpu(a.xgb_gpu)
    {
        "rmix": cmd_rmix,
        "provenance": cmd_provenance,
        "threshold": cmd_threshold,
        "mlp": cmd_mlp,
        "bootstrap": cmd_bootstrap,
        "floor": cmd_floor,
    }[a.cmd](a)
