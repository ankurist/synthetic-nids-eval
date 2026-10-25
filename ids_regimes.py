#!/usr/bin/env python
"""Training regimes: real, synthetic and mixed training sets, always scored on the real test split.

Regimes (param = fraction of the real class size):
  R0      real rows only                                   reference
  R1      real + synthetic rows for minority classes       param in {0.25, 0.5, 1.0}
  R1_ROS  real + duplicated real minority rows              same sizes (duplication control, "ROS" in the paper)
  R2      real benign + synthetic attacks                  class-aligned provenance
  R2r     synthetic benign + real attacks                  mirror of R2
  R3      all classes synthetic                            train on synthetic, test on real (TSTR)
The real-versus-synthetic probe is run with the `probe` command; the within-class mixing
control (RMIX) and the provenance test are in ids_extra.py.

Usage:
    python ids_regimes.py run     --dataset cicids2017 --seeds 0 1 2 3 4 --methods smote copula ctgan_e100 tvae_e100
    python ids_regimes.py probe   --dataset cicids2017 --seeds 0 1 2 3 4 --methods smote copula ctgan_e100 tvae_e100
    python ids_regimes.py summary --dataset cicids2017

Runs are resumable: completed (seed, method, regime, param, model, task) rows are skipped.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import ids_eval as E
from ids_synth import safe

ROOT = E.ROOT


def path(*p):
    return os.path.join(ROOT, *p)


def rtag(split: str) -> str:
    return "" if split == "random" else f"_{split}"


OUT_TAG = ""  # set from --out_tag; appended to the results folder name


def load_synth(dataset: str, split: str, seed: int, method: str, classes: list[str]) -> dict[str, pd.DataFrame]:
    d = path("synth", dataset, "" if split == "random" else split, f"seed{seed}", method)
    out, missing = {}, []
    for c in classes:
        f = os.path.join(d, f"{safe(c)}.parquet")
        (out.__setitem__(c, pd.read_parquet(f)) if os.path.exists(f) else missing.append(c))
    if missing:
        raise FileNotFoundError(f"{method} seed{seed}: missing synthetic classes {missing} in {d}")
    return out


def minority_classes(tr: pd.DataFrame, frac: float) -> list[str]:
    share = tr["category"].astype(str).value_counts(normalize=True)
    return [c for c, s in share.items() if s < frac and c != "Benign"]


# ------------------------------------------------------------------ build the training mixture
def build(
    regime: str,
    param: float,
    real: pd.DataFrame,
    synth: dict | None,
    classes: list[str],
    minority: list[str],
    seed: int,
) -> tuple[pd.DataFrame, int, int]:
    rng = np.random.default_rng(seed * 1000 + int(param * 100))

    def take(df, n):
        return df.sample(n=int(n), replace=int(n) > len(df), random_state=int(rng.integers(1 << 30)))

    n_c = real["category"].value_counts()
    ben = real[real["category"] == "Benign"]
    if regime == "R0":
        parts, ns = [real], 0
    elif regime == "R1_ROS":
        extra = [take(real[real["category"] == c], round(param * n_c[c])) for c in minority]
        parts, ns = [real] + extra, 0
    elif regime == "R1":
        extra = [take(synth[c], round(param * n_c[c])) for c in minority]
        parts, ns = [real] + extra, sum(len(x) for x in extra)
    elif regime == "R2":
        sy = [synth[c] for c in classes if c != "Benign"]
        parts, ns = [ben] + sy, sum(len(x) for x in sy)
    elif regime == "R2r":
        parts, ns = [real[real["category"] != "Benign"], synth["Benign"]], len(synth["Benign"])
    elif regime == "R3":
        sy = [synth[c] for c in classes]
        parts, ns = sy, sum(len(x) for x in sy)
    else:
        raise ValueError(regime)
    cols = [c for c in real.columns]
    df = pd.concat([p[cols] for p in parts], ignore_index=True)
    return df, len(df) - ns, ns


# ------------------------------------------------------------------ run
def run(
    dataset: str,
    split: str,
    seeds: list[int],
    methods: list[str],
    models: list[str],
    tasks: list[str],
    regimes: list[str],
    fracs: list[float],
    minority_frac: float,
    minority_override: list[str] | None,
    mc_regimes: list[str],
) -> None:
    out = path("results", dataset, "regimes" + rtag(split) + OUT_TAG)
    os.makedirs(os.path.join(out, "preds"), exist_ok=True)
    csv = os.path.join(out, "regimes_per_seed.csv")
    done = set()
    if os.path.exists(csv):
        d0 = pd.read_csv(csv)
        done = set(zip(d0.seed, d0.method, d0.regime, d0.param.round(4), d0.model, d0.task))
    for seed in seeds:
        tr, va, te, schema = E.load_split(dataset, seed, split)
        classes = schema["classes"]
        cat_cols = list(schema["categorical"])
        for c in cat_cols:
            tr[c] = tr[c].astype(str)
            te[c] = te[c].astype(str)
        tr["category"] = tr["category"].astype(str)
        te["category"] = te["category"].astype(str)
        prep = E.Prep(schema).fit(tr)  # scaling ALWAYS fitted on real train
        Xte = prep.transform(te)
        yte, cte = te["binary"].values, te["category"].values
        minority = minority_override or minority_classes(tr, minority_frac)
        cidx = {c: i for i, c in enumerate(classes)}
        keep = prep.names + [
            "category",
            "binary",
        ]  # only columns the model uses (train-constant ones were never generated)
        real = tr[keep]
        print(f"\n=== seed {seed} | minority classes (share < {minority_frac}): {minority}")

        jobs = []
        if "R0" in regimes:
            jobs.append(("real", "R0", 0.0))
        if "R1_ROS" in regimes:
            jobs += [("real", "R1_ROS", f) for f in fracs]
        for m in methods:
            if "R1" in regimes:
                jobs += [(m, "R1", f) for f in fracs]
            jobs += [(m, r, 0.0) for r in ("R2", "R2r", "R3") if r in regimes]

        cache: dict[str, dict] = {}
        for method, regime, param in jobs:
            tks = [tk for tk in tasks if tk != "multiclass" or regime in mc_regimes]
            keys = [
                (seed, method, regime, round(param, 4), "RF" if mo == "rf" else "XGB", tk)
                for mo in models
                for tk in tks
            ]
            if all(k in done for k in keys):
                continue
            synth = None
            if method != "real":
                try:
                    if method not in cache:
                        cache.clear()
                        cache[method] = load_synth(dataset, split, seed, method, classes)
                    synth = cache[method]
                except FileNotFoundError as e:
                    print("  [skip]", e)
                    continue
                for c in cat_cols:
                    for k in synth:
                        synth[k][c] = synth[k][c].astype(str)
                for k in synth:
                    synth[k]["category"] = synth[k]["category"].astype(str)
            mix, n_real, n_syn = build(regime, param, real, synth, classes, minority, seed)
            Xtr = prep.transform(mix)
            ybin, ycat = mix["binary"].values, mix["category"].values
            rows, saved = [], {}
            for mo in models:
                if "binary" in tks:
                    name, clf = E.make_binary(mo, seed)
                    if clf is not None:
                        t0 = time.time()
                        clf.fit(Xtr, ybin)
                        proba = clf.predict_proba(Xte)[:, 1]
                        pred = (proba >= 0.5).astype(int)
                        saved[f"{name}_binary_pred"], saved[f"{name}_binary_proba"] = pred, proba
                        rows.append(
                            dict(
                                model=name,
                                task="binary",
                                seconds=round(time.time() - t0, 1),
                                **E.binary_metrics(yte, pred, proba),
                            )
                        )
                if "multiclass" in tks:
                    name, clf = E.make_multi(mo, seed)
                    if clf is not None:
                        t0 = time.time()
                        clf.fit(Xtr, np.array([cidx[c] for c in ycat]))
                        pm = np.array(classes)[clf.predict(Xte).astype(int)]
                        saved[f"{name}_multi_pred"] = pm
                        rows.append(
                            dict(
                                model=name,
                                task="multiclass",
                                seconds=round(time.time() - t0, 1),
                                **E.multiclass_metrics(cte, pm, classes),
                            )
                        )
            for r in rows:
                r.update(
                    dataset=dataset,
                    split=split,
                    seed=seed,
                    method=method,
                    regime=regime,
                    param=param,
                    n_train_real=n_real,
                    n_train_synth=n_syn,
                )
            np.savez_compressed(
                os.path.join(out, "preds", f"{method}__{regime}_{param}__seed{seed}.npz"), y=yte, cat=cte, **saved
            )
            new = pd.DataFrame(rows)  # align by column NAME, drop re-run duplicates
            if os.path.exists(csv):
                new = pd.concat([pd.read_csv(csv), new], ignore_index=True, sort=False)
            new.drop_duplicates(subset=["seed", "method", "regime", "param", "model", "task"], keep="last").to_csv(
                csv, index=False
            )
            b = [r for r in rows if r["task"] == "binary"]
            msg = " | ".join(
                f"{r['model']} macroF1 {r['macro_f1']:.4f} FPR {r.get('fpr', float('nan')):.4f}" for r in b
            ) or " | ".join(f"{r['model']} mc-macroF1 {r['macro_f1']:.4f}" for r in rows)
            print(f"  {method:11s} {regime:7s} p={param:<4} train real {n_real:6d} synth {n_syn:6d} | {msg}")


# ------------------------------------------------------------------ R4 provenance probe
def probe(dataset: str, split: str, seeds: list[int], methods: list[str], max_rows: int = 5000) -> None:
    out = path("results", dataset, "probe" + rtag(split) + OUT_TAG)
    os.makedirs(out, exist_ok=True)
    rows = []
    for seed in seeds:
        tr, _, _, schema = E.load_split(dataset, seed, split)
        classes = schema["classes"]
        for c in schema["categorical"]:
            tr[c] = tr[c].astype(str)
        tr["category"] = tr["category"].astype(str)
        prep = E.Prep(schema).fit(tr)
        for m in methods:
            try:
                synth = load_synth(dataset, split, seed, m, classes)
            except FileNotFoundError as e:
                print("[skip]", e)
                continue
            for c in schema["categorical"]:
                for k in synth:
                    synth[k][c] = synth[k][c].astype(str)
            groups = {c: (tr[tr["category"] == c], synth[c]) for c in classes}
            groups["ALL_ATTACKS"] = (
                tr[tr["category"] != "Benign"],
                pd.concat([synth[c] for c in classes if c != "Benign"], ignore_index=True),
            )
            for g, (R, S) in groups.items():
                n = min(len(R), len(S), max_rows)
                if n < 20:
                    continue
                R, S = R.sample(n, random_state=seed), S.sample(n, random_state=seed)
                X = np.vstack([prep.transform(R), prep.transform(S)])
                y = np.r_[np.zeros(n), np.ones(n)]
                a, b, ya, yb = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
                clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=2).fit(a, ya)
                auc = roc_auc_score(yb, clf.predict_proba(b)[:, 1])
                top = prep.names[int(np.argmax(clf.feature_importances_))]
                rows.append(
                    dict(
                        dataset=dataset,
                        split=split,
                        seed=seed,
                        method=m,
                        group=g,
                        n_each=n,
                        probe_auc=auc,
                        top_feature=top,
                    )
                )
            print(f"seed {seed} {m}: probe done")
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(out, "probe_per_seed.csv"), index=False)
    t = d.pivot_table(index="group", columns="method", values="probe_auc", aggfunc="mean").round(3)
    print("\nMean probe AUC (0.5 = indistinguishable from real, 1.0 = trivially detectable;")
    print("values below 0.5 indicate near-copies of real rows):\n", t.to_string())


# ------------------------------------------------------------------ threshold-free metrics from the saved predictions
def add_threshold_free(d: pd.DataFrame, out: str) -> pd.DataFrame:
    """Attack recall at a FIXED false-positive rate (1 % and 5 %), computed from the saved probabilities.
    Unlike macro-F1 at threshold 0.5 it does not move when a regime merely shifts the operating point."""
    from sklearn.metrics import roc_curve

    d = d.copy()
    d["tpr_fpr01"], d["tpr_fpr05"] = np.nan, np.nan
    cache = {}
    for i, r in d[d["task"] == "binary"].iterrows():
        f = os.path.join(out, "preds", f"{r['method']}__{r['regime']}_{float(r['param'])}__seed{int(r['seed'])}.npz")
        if not os.path.exists(f):
            continue
        z = cache.setdefault(f, np.load(f, allow_pickle=True))
        key = f"{r['model']}_binary_proba"
        if key not in z:
            continue
        fpr, tpr, _ = roc_curve(z["y"], z[key])
        for col, tgt in (("tpr_fpr01", 0.01), ("tpr_fpr05", 0.05)):
            d.at[i, col] = tpr[max(np.searchsorted(fpr, tgt, side="right") - 1, 0)]
    return d


# ------------------------------------------------------------------ summary with paired deltas vs R0
def summary(dataset: str, split: str) -> None:
    out = path("results", dataset, "regimes" + rtag(split) + OUT_TAG)
    d = pd.read_csv(os.path.join(out, "regimes_per_seed.csv"))
    d = add_threshold_free(d, out)
    base = d[d["regime"] == "R0"].set_index(["seed", "model", "task"])
    metr = [
        c
        for c in d.columns
        if c in ("macro_f1", "bal_acc", "acc", "fpr", "rec", "prec", "pr_auc", "roc_auc", "tpr_fpr01", "tpr_fpr05")
        or c.startswith("recall_")
    ]
    for c in metr:
        b = d.set_index(["seed", "model", "task"]).index.map(base[c])
        d["d_" + c] = d[c].values - np.asarray(b, dtype=float)  # paired: same seed, same model, same task
    keycols = ("method", "regime", "param", "model", "task")
    s = E.summarise(d, group=keycols)
    s.to_csv(os.path.join(out, "regimes_summary_with_deltas.csv"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    for task, cols in (
        (
            "binary",
            [
                "macro_f1_mean",
                "macro_f1_ci95",
                "d_macro_f1_mean",
                "d_macro_f1_ci95",
                "fpr_mean",
                "d_fpr_mean",
                "rec_mean",
                "pr_auc_mean",
                "d_pr_auc_mean",
                "tpr_fpr05_mean",
                "d_tpr_fpr05_mean",
            ],
        ),
    ):
        t = s.xs(task, level="task")
        print("\n=== BINARY (delta = regime minus R0, paired over seeds) ===")
        print(t[[c for c in cols if c in t.columns]].to_string())
    t = s.xs("multiclass", level="task")
    dc = ["d_macro_f1_mean"] + sorted(c for c in t.columns if c.startswith("d_recall_") and c.endswith("_mean"))
    print("\n=== MULTICLASS (paired deltas vs R0; per-class recall) ===")
    print(t[[c for c in dc if c in t.columns]].round(3).to_string())
    print(f"\nsaved {os.path.join(out, 'regimes_summary_with_deltas.csv')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "probe", "summary"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="random", choices=["random", "chrono", "pooled"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--methods", nargs="+", default=["smote", "copula", "ctgan_e100", "tvae_e100"])
    ap.add_argument("--models", nargs="+", default=["rf", "xgb"], choices=["rf", "xgb"])
    ap.add_argument("--tasks", nargs="+", default=["binary", "multiclass"], choices=["binary", "multiclass"])
    ap.add_argument("--regimes", nargs="+", default=["R0", "R1", "R1_ROS", "R2", "R2r", "R3"])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument(
        "--minority_frac",
        type=float,
        default=0.03,
        help="classes with a smaller share of the training set are 'minority' (default 3%%)",
    )
    ap.add_argument("--minority", nargs="+", help="override the minority class list")
    ap.add_argument(
        "--mc_regimes",
        nargs="+",
        default=["R0", "R1", "R1_ROS", "R3"],
        help="regimes that also get the multiclass task (saves time; binary always runs)",
    )
    ap.add_argument("--xgb_gpu", action="store_true", help="run XGBoost on a CUDA-capable GPU")
    ap.add_argument("--out_tag", default="", help="suffix for the results folder, e.g. _final")
    a = ap.parse_args()
    OUT_TAG = a.out_tag
    if a.xgb_gpu:
        try:
            import torch

            ok = torch.cuda.is_available()
        except ImportError:
            ok = False
        if not ok:
            raise SystemExit("--xgb_gpu needs a GPU runtime (Runtime > Change runtime type > T4 GPU).")
        E.XGB_DEVICE = "cuda"
        print("XGBoost: running on GPU")
    if a.cmd == "run":
        run(
            a.dataset,
            a.split,
            a.seeds,
            a.methods,
            a.models,
            a.tasks,
            a.regimes,
            a.fracs,
            a.minority_frac,
            a.minority,
            a.mc_regimes,
        )
    elif a.cmd == "probe":
        probe(a.dataset, a.split, a.seeds, a.methods)
    else:
        summary(a.dataset, a.split)
