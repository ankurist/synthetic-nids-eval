#!/usr/bin/env python
"""Shared evaluation utilities and the real-only baseline (regime R0).

Usage:
    python ids_eval.py r0       --dataset cicids2017 --seeds 0 1 2 3 4 --models rf xgb
    python ids_eval.py diagnose --dataset unsw_nb15

`Prep` (train-only scaling and encoding), `binary_metrics` and `multiclass_metrics` are
imported by the other modules, so every regime is scored identically. All reported metrics
are computed from saved prediction files, so tables can be regenerated without retraining.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

try:
    import xgboost as xgb
except ImportError:
    xgb = None

ROOT = os.environ.get("IDS_ROOT", "ids_project")
XGB_DEVICE = os.environ.get("IDS_XGB_DEVICE", "cpu")  # set to "cuda" to run XGBoost on a GPU


def path(*p):
    return os.path.join(ROOT, *p)


# ---------------------------------------------------------------- data access
def load_split(dataset: str, seed: int, split: str = "random"):
    d = path("splits", dataset, "" if split == "random" else split, f"seed{seed}")
    tr, va, te = (pd.read_parquet(os.path.join(d, f"{n}.parquet")) for n in ("train", "val", "test"))
    schema = json.load(open(path("processed", dataset, "schema.json")))
    return tr, va, te, schema


class Prep:
    """Scaler / encoder fitted on TRAIN ONLY. Test is clipped to [0,1]."""

    def __init__(self, schema: dict):
        self.schema = schema

    def fit(self, train: pd.DataFrame) -> "Prep":
        self.num = [c for c in self.schema["numeric"] if train[c].nunique() > 1]
        self.cat = list(self.schema["categorical"])
        self.scaler = MinMaxScaler().fit(train[self.num].values)
        self.enc = None
        if self.cat:
            self.enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(
                train[self.cat].astype(str)
            )
        self.names = self.num + self.cat
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = np.clip(self.scaler.transform(df[self.num].values), 0, 1).astype("float32")
        if self.enc is not None:
            X = np.hstack([X, self.enc.transform(df[self.cat].astype(str)).astype("float32")])
        return X


# -------------------------------------------------------------------- metrics
def binary_metrics(y, pred, proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        acc=accuracy_score(y, pred),
        bal_acc=balanced_accuracy_score(y, pred),
        prec=precision_score(y, pred, zero_division=0),
        rec=recall_score(y, pred, zero_division=0),
        f1_attack=f1_score(y, pred, zero_division=0),
        macro_f1=f1_score(y, pred, average="macro", zero_division=0),
        fpr=fp / max(fp + tn, 1),
        roc_auc=roc_auc_score(y, proba),
        pr_auc=average_precision_score(y, proba),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        baseline_always_attack_acc=float(np.mean(y == 1)),
        baseline_always_benign_acc=float(np.mean(y == 0)),
    )


def multiclass_metrics(y, pred, classes) -> dict:
    cm = confusion_matrix(y, pred, labels=classes)
    rec = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    out = dict(
        acc=accuracy_score(y, pred),
        bal_acc=balanced_accuracy_score(y, pred),
        macro_f1=f1_score(y, pred, average="macro", labels=classes, zero_division=0),
    )
    out.update({f"recall_{c}": r for c, r in zip(classes, rec)})
    return out


def ci95(x: pd.Series) -> float:
    """Half-width of the 95% t-interval over seeds."""
    x = x.dropna()
    if len(x) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def summarise(rows: pd.DataFrame, group=("model", "task")) -> pd.DataFrame:
    num = [
        c
        for c in rows.columns
        if c not in group and c not in ("seed", "regime", "dataset") and pd.api.types.is_numeric_dtype(rows[c])
    ]
    g = rows.groupby(list(group))[num]
    mean, half = g.mean(), g.agg(ci95)
    out = mean.add_suffix("_mean").join(half.add_suffix("_ci95"))
    return out.round(4)


# ------------------------------------------------------------------------ R0
def make_binary(name, seed):
    if name == "rf":
        return "RF", RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=2)
    if name == "xgb":
        if xgb is None:
            print("xgboost is not installed; skipping XGBoost (pip install xgboost)")
            return None, None
        return "XGB", xgb.XGBClassifier(
            n_estimators=100, random_state=seed, n_jobs=2, tree_method="hist", eval_metric="logloss", device=XGB_DEVICE
        )
    raise ValueError(name)


def make_multi(name, seed):
    if name == "rf":
        return "RF", RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=2)
    if name == "xgb" and xgb is not None:
        return "XGB", xgb.XGBClassifier(
            n_estimators=100, random_state=seed, n_jobs=2, tree_method="hist", device=XGB_DEVICE
        )
    return None, None


def run_r0(dataset: str, seeds: list[int], models: list[str], split: str = "random") -> pd.DataFrame:
    out = path("results", dataset, "R0" if split == "random" else f"R0_{split}")
    os.makedirs(os.path.join(out, "preds"), exist_ok=True)
    rows = []
    for seed in seeds:
        tr, va, te, schema = load_split(dataset, seed, split)
        classes = schema["classes"]
        prep = Prep(schema).fit(tr)
        Xtr, Xte = prep.transform(tr), prep.transform(te)
        ytr, yte = tr["binary"].values, te["binary"].values
        ctr, cte = tr["category"].astype(str).values, te["category"].astype(str).values
        cidx = {c: i for i, c in enumerate(classes)}
        for m in models:
            name, clf = make_binary(m, seed)
            if clf is None:
                continue
            clf.fit(Xtr, ytr)
            proba = clf.predict_proba(Xte)[:, 1]
            pred = (proba >= 0.5).astype(int)
            np.savez_compressed(
                os.path.join(out, "preds", f"{name}_binary_seed{seed}.npz"), y=yte, pred=pred, proba=proba, cat=cte
            )
            rows.append(dict(model=name, task="binary", seed=seed, **binary_metrics(yte, pred, proba)))
            name, clf = make_multi(m, seed)
            if clf is None:
                continue
            clf.fit(Xtr, np.array([cidx[c] for c in ctr]))
            pm = np.array(classes)[clf.predict(Xte).astype(int)]
            np.savez_compressed(os.path.join(out, "preds", f"{name}_multi_seed{seed}.npz"), y=cte, pred=pm)
            pd.DataFrame(confusion_matrix(cte, pm, labels=classes), index=classes, columns=classes).to_csv(
                os.path.join(out, f"{name}_multi_confusion_seed{seed}.csv")
            )
            rows.append(dict(model=name, task="multiclass", seed=seed, **multiclass_metrics(cte, pm, classes)))
        print(f"seed {seed} done")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "R0_per_seed.csv"), index=False)
    summ = summarise(res)
    summ.to_csv(os.path.join(out, "R0_summary_mean_ci95.csv"))
    show = [
        c
        for c in summ.columns
        if any(k in c for k in ("acc_mean", "macro_f1", "fpr", "pr_auc", "baseline_always_attack"))
    ]
    print(summ[show].T.to_string())
    return res


# ------------------------------------------------------------- diagnose
def diagnose(dataset: str, seed: int = 0, split: str = "random") -> None:
    """Three checks that expose shortcut features and partition shift.
    1) feature list   2) adversarial validation train-vs-test   3) top features of the real task
    4) sanity: same model on a RANDOM re-split of train+test pooled."""
    from sklearn.model_selection import train_test_split

    tr, va, te, schema = load_split(dataset, seed, split)
    prep = Prep(schema).fit(tr)
    print(f"\n[1] {len(prep.names)} features used ({len(prep.num)} numeric, {len(prep.cat)} categorical):")
    print(prep.names)
    sus = [
        n
        for n in prep.names
        if n.lower() in ("id", "index", "idx", "row_id") or "unnamed" in n.lower() or "\ufeff" in n or "\u00ef" in n
    ]
    if sus:
        print("WARNING: identifier-like feature(s) detected:", sus)
    Xtr, Xte = prep.transform(tr), prep.transform(te)

    # 2) adversarial validation: can a model tell train rows from test rows?
    Xa = np.vstack([Xtr[:20000], Xte[:20000]])
    ya = np.r_[np.zeros(min(len(Xtr), 20000)), np.ones(min(len(Xte), 20000))]
    a_tr, a_te, y_tr, y_te = train_test_split(Xa, ya, test_size=0.3, random_state=0, stratify=ya)
    rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=2).fit(a_tr, y_tr)
    auc = roc_auc_score(y_te, rf.predict_proba(a_te)[:, 1])
    top = np.argsort(rf.feature_importances_)[::-1][:5]
    print(f"\n[2] train-vs-test classifier AUC = {auc:.3f}  (0.5 = identical distributions, 1.0 = trivially different)")
    print(
        "    features that separate train from test:",
        [(prep.names[i], round(float(rf.feature_importances_[i]), 3)) for i in top],
    )

    # 3) what does the real classifier rely on?
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=2).fit(Xtr, tr["binary"].values)
    top = np.argsort(clf.feature_importances_)[::-1][:8]
    print(
        "\n[3] top features for attack-vs-benign:",
        [(prep.names[i], round(float(clf.feature_importances_[i]), 3)) for i in top],
    )
    acc_off = accuracy_score(te["binary"].values, clf.predict(Xte))
    print(f"    accuracy on the official/locked TEST split = {acc_off:.4f}")

    # 4) pooled random re-split
    pool = pd.concat([tr, va, te], ignore_index=True)
    p_tr, p_te = train_test_split(pool, test_size=0.3, random_state=0, stratify=pool["category"])
    pp = Prep(schema).fit(p_tr)
    c2 = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=2).fit(
        pp.transform(p_tr), p_tr["binary"].values
    )
    acc_rand = accuracy_score(p_te["binary"].values, c2.predict(pp.transform(p_te)))
    print(f"\n[4] same model, pooled train+test randomly re-split: accuracy = {acc_rand:.4f}")
    print(
        "\nHow to read it: if [4] is far higher than the official-test accuracy, the gap comes from "
        "train/test shift (look at the features named in [2]); if [4] is also poor, suspect the features or labels."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["r0", "diagnose"])
    ap.add_argument("--split", default="random", choices=["random", "chrono", "pooled"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="+", default=["rf", "xgb"], choices=["rf", "xgb"])
    a = ap.parse_args()
    if a.cmd == "r0":
        run_r0(a.dataset, a.seeds, a.models, a.split)
    else:
        diagnose(a.dataset, a.seeds[0], a.split)
