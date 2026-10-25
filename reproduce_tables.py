#!/usr/bin/env python
"""Reproduce every table and figure data set reported in the paper from the per-seed result files.

Usage:
    python reproduce_tables.py --results results --out tables

No model is retrained: all values are computed from the CSV and JSON files in --results, which
are the outputs of ids_data.py, ids_synth.py, ids_regimes.py, ids_fidelity.py and ids_extra.py.
One CSV per table or figure is written to --out, together with summary.txt.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

DATASETS = {"CIC-IDS2017": "cicids2017", "UNSW-NB15": "unsw_nb15"}
GENERATORS = {"SMOTE": "smote", "Copula": "copula", "TVAE": "tvae_e100", "CTGAN": "ctgan_e100"}
DETECTORS = ("RF", "XGB")
EQUIVALENCE_MARGIN = 0.01  # macro-F1, declared before testing
KEYS = ["seed", "method", "regime", "model", "task"]


# --------------------------------------------------------------------------- statistics
def t_interval(x) -> float:
    """Half-width of the 95% t-interval of the mean."""
    x = np.asarray(x, float)
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def holm(p) -> np.ndarray:
    """Holm step-down adjustment of a family of p-values."""
    p = np.asarray(p, float)
    order, m, adj, running = np.argsort(p), len(p), np.empty(len(p)), 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Differences a - b paired by seed."""
    a, b = a.sort_index(), b.sort_index()
    return (a - b.loc[a.index]).to_numpy()


def tost_p(d: np.ndarray, margin: float = EQUIVALENCE_MARGIN) -> float:
    """Two one-sided t-tests for equivalence of the mean paired difference to zero within +-margin."""
    n, se, mu = len(d), d.std(ddof=1) / np.sqrt(len(d)), d.mean()
    return float(max(1 - stats.t.cdf((mu + margin) / se, n - 1), stats.t.cdf((mu - margin) / se, n - 1)))


def ci90(d: np.ndarray) -> tuple[float, float]:
    se = d.std(ddof=1) / np.sqrt(len(d))
    h = stats.t.ppf(0.95, len(d) - 1) * se
    return float(d.mean() - h), float(d.mean() + h)


# --------------------------------------------------------------------------- data access
class Results:
    def __init__(self, root: str, name: str):
        d = os.path.join(root, name)
        rd = lambda f, keys=KEYS: pd.read_csv(os.path.join(d, f)).drop_duplicates(subset=[k for k in keys if k])
        self.regimes = rd("regimes_per_seed.csv", KEYS + ["param"])
        self.ablation = rd("regimes_epoch_ablation.csv", KEYS + ["param"])
        self.fidelity = pd.read_csv(os.path.join(d, "fidelity_privacy_per_seed.csv"))
        self.probe = pd.read_csv(os.path.join(d, "probe_per_seed.csv"))
        self.genlog = pd.read_csv(os.path.join(d, "generator_log.csv"))
        self.rmix = rd("rmix_per_seed.csv")
        self.prov = rd("provenance_per_seed.csv", KEYS[:4])
        self.thr = rd("threshold_per_seed.csv")
        self.mlp = rd("mlp_per_seed.csv")
        self.boot = pd.read_csv(os.path.join(d, "bootstrap_binary_deltas.csv"))
        self.pcf1 = pd.read_csv(os.path.join(d, "per_class_f1_per_seed.csv"))
        self.floor = pd.read_csv(os.path.join(d, "label_conflict_floor.csv")).iloc[0].to_dict()
        self.card = json.load(open(os.path.join(d, "dataset_card.json")))
        self.manifest = json.load(open(os.path.join(d, "split_manifest_seed0.json")))
        chrono = os.path.join(d, "regimes_chronological_split.csv")
        self.chrono = pd.read_csv(chrono) if os.path.exists(chrono) else None

    def s(self, method, regime, model, metric="macro_f1", task="binary", param=None, df=None) -> pd.Series:
        r = self.regimes if df is None else df
        g = r[(r.task == task) & (r.method == method) & (r.regime == regime) & (r.model == model)]
        if param is not None:
            g = g[np.isclose(g.param, param)]
        return g.set_index("seed")[metric].sort_index()

    def base(self, model, metric="macro_f1", task="binary") -> pd.Series:
        return self.s("real", "R0", model, metric, task)

    def delta(self, method, regime, model, metric="macro_f1", task="binary", param=None) -> np.ndarray:
        return paired(self.s(method, regime, model, metric, task, param), self.base(model, metric, task))


def load(root: str) -> dict:
    return {short: Results(root, name) for short, name in (("CIC", "cicids2017"), ("UNSW", "unsw_nb15"))}


# --------------------------------------------------------------------------- tables
def table1_cleaning(R: dict) -> pd.DataFrame:
    cc, uc = R["CIC"].card, R["UNSW"].card
    mc, mu = R["CIC"].manifest, R["UNSW"].manifest
    files = cc["files"]
    blank = files["Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"]["dropped_nan_inf_by_label"]["nan"]
    read = sum(v["rows_read"] for v in files.values())
    nonfin = sum(v["dropped_nan_inf"] for v in files.values())
    dups = sum(v["dropped_duplicates"] for v in files.values()) + cc["dups_removed_across_files"]
    tr, te = uc["files"]["UNSW_NB15_training-set.csv"], uc["files"]["UNSW_NB15_testing-set.csv"]
    overlap = uc["test_rows_removed_because_identical_to_train"]
    tr_conf = tr["rows_after_dedup"] - (mu["train"]["n"] + mu["val"]["n"])
    te_conf = te["rows_after_dedup"] - overlap - mu["test"]["n"]
    assert tr_conf + te_conf == uc["rows_conflicting_benign_vs_attack"], "UNSW-NB15 cleaning flow does not reconcile"
    n = lambda m: m["train"]["n"] + m["val"]["n"] + m["test"]["n"]
    rows = [
        ("rows read", read, f"{tr['rows_read']} / {te['rows_read']}"),
        ("blank rows removed", blank, "0 / 0"),
        ("non-finite rows removed", nonfin - blank, "0 / 0"),
        ("exact duplicates removed", dups, f"{tr['dropped_duplicates']} / {te['dropped_duplicates']}"),
        (
            "exact duplicate share (%)",
            round(100 * dups / (read - nonfin), 1),
            f"{100 * tr['dropped_duplicates'] / tr['rows_read']:.1f} / {100 * te['dropped_duplicates'] / te['rows_read']:.1f}",
        ),
        ("test rows identical to train removed", 0, f"0 / {overlap}"),
        ("benign-vs-attack conflicts removed", cc["rows_conflicting_benign_vs_attack"], f"{tr_conf} / {te_conf}"),
        ("rows with any label conflict", cc["rows_conflicting_any_category"], uc["rows_conflicting_any_category"]),
        (
            "attack-vs-attack conflicts kept",
            0,
            uc["rows_conflicting_any_category"] - uc["rows_conflicting_benign_vs_attack"],
        ),
        ("rows used (per seed / partitions)", n(mc), f"{mu['train']['n'] + mu['val']['n']} / {mu['test']['n']}"),
        (
            "train / val / test",
            f"{mc['train']['n']} / {mc['val']['n']} / {mc['test']['n']}",
            f"{mu['train']['n']} / {mu['val']['n']} / {mu['test']['n']}",
        ),
        (
            "test attack share (%)",
            round(100 * (1 - mc["test"]["by_class"]["Benign"] / mc["test"]["n"]), 1),
            round(100 * (1 - mu["test"]["by_class"]["Benign"] / mu["test"]["n"]), 1),
        ),
    ]
    return pd.DataFrame(rows, columns=["step", "CIC-IDS2017", "UNSW-NB15 (train / test partition)"])


def table3_per_class(R: dict) -> pd.DataFrame:
    rows = []
    for ds, r in R.items():
        for cls, n_tr in r.manifest["train"]["by_class"].items():
            f1 = {
                mo: r.pcf1[(r.pcf1.key == "real__R0_0.0") & (r.pcf1.model == mo) & (r.pcf1.cls == cls)].f1.mean()
                for mo in DETECTORS
            }
            f = r.fidelity
            p = r.probe
            rows.append(
                dict(
                    dataset=ds,
                    cls=cls,
                    train=n_tr,
                    val=r.manifest["val"]["by_class"][cls],
                    test=r.manifest["test"]["by_class"][cls],
                    f1_rf=f1["RF"],
                    f1_xgb=f1["XGB"],
                    smote_copies_pct=100 * f[(f.method == "smote") & (f.group == cls)].exact_dup_share.mean(),
                    smote_probe_auc=p[(p.method == "smote") & (p.group == cls)].probe_auc.mean(),
                )
            )
    return pd.DataFrame(rows)


def table4_operating_point(R: dict) -> pd.DataFrame:
    rows = []
    for mo in DETECTORS:
        for label, method, regime in [("R0 real only", "real", "R0")] + [
            (f"{rg} {g}", m, rg) for rg in ("R2", "R3") for g, m in GENERATORS.items()
        ]:
            row = dict(detector=mo, training=label)
            for ds, r in R.items():
                for col in ("rec", "fpr", "roc_auc", "pr_auc"):
                    row[f"{ds}_{col}"] = r.s(method, regime, mo, col).mean()
            rows.append(row)
    return pd.DataFrame(rows)


def fig2_regime_effects(R: dict) -> pd.DataFrame:
    rows = []
    for ds, r in R.items():
        for mo in DETECTORS:
            for g, m in GENERATORS.items():
                for rg in ("R2", "R2r", "R3"):
                    d = r.delta(m, rg, mo)
                    rows.append(
                        dict(
                            dataset=ds, detector=mo, generator=g, regime=rg, delta_macro_f1=d.mean(), ci95=t_interval(d)
                        )
                    )
    return pd.DataFrame(rows)


def table5_provenance(R: dict) -> pd.DataFrame:
    rows = []
    for ds, r in R.items():
        for g, m in GENERATORS.items():
            for mo in DETECTORS:
                mix = (
                    r.rmix[(r.rmix.method == m) & (r.rmix.model == mo) & (r.rmix.task == "binary")]
                    .set_index("seed")
                    .macro_f1
                )
                pv = r.prov

                def sel(rg, pv=pv, m=m, mo=mo):
                    return pv[(pv.method == m) & (pv.regime == rg) & (pv.model == mo)]

                rows.append(
                    dict(
                        dataset=ds,
                        generator=g,
                        detector=mo,
                        delta_r2=r.delta(m, "R2", mo).mean(),
                        delta_rmix=paired(mix, r.base(mo)).mean(),
                        syn_benign_flagged_by_r2_pct=100 * sel("R2").heldout_syn_benign_flagged.mean(),
                        real_benign_flagged_by_r2_pct=100 * sel("R2").real_benign_fpr.mean(),
                        syn_attacks_caught_by_r2r_pct=100 * sel("R2r").heldout_syn_attack_recall.mean(),
                    )
                )
    return pd.DataFrame(rows)


def table6_thresholds(R: dict) -> pd.DataFrame:
    rows = []
    labels = [("R0 real only", "real", "R0"), ("R0 class-weighted", "real", "R0_weighted")] + [
        (f"{rg} {g}", m, rg) for rg in ("R2", "R3") for g, m in GENERATORS.items()
    ]
    for label, m, rg in labels:
        row = dict(training=label)
        for ds, r in R.items():
            t = r.thr[(r.thr.method == m) & (r.thr.regime == rg) & (r.thr.task == "binary")]
            for mo in DETECTORS:
                tm = t[t.model == mo]
                row[f"{ds}_{mo}_default"] = tm.macro_f1_default.mean()
                row[f"{ds}_{mo}_tuned"] = tm.macro_f1_tuned.mean()
                row[f"{ds}_{mo}_threshold"] = tm.threshold.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def table7_augmentation(R: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    def family(r, metric):
        keys = [(m, mo) for m in ("smote", "copula", "ctgan_e100", "tvae_e100") for mo in DETECTORS]
        diffs = {
            k: paired(
                r.s(k[0], "R1", k[1], metric, "multiclass", 1.0), r.s("real", "R1_ROS", k[1], metric, "multiclass", 1.0)
            )
            for k in keys
        }
        p_t = dict(zip(keys, holm([stats.ttest_1samp(diffs[k], 0).pvalue for k in keys])))
        return keys, diffs, p_t

    def equivalence(r):
        keys, diffs, _ = family(r, "macro_f1")
        p_eq = dict(zip(keys, holm([tost_p(diffs[k]) for k in keys])))
        out = {}
        for k in keys:
            lo, hi = ci90(diffs[k])
            out[k] = (
                "equivalent"
                if p_eq[k] < 0.05
                else (
                    "worse" if hi < -EQUIVALENCE_MARGIN else ("better" if lo > EQUIVALENCE_MARGIN else "inconclusive")
                )
            )
        return out, p_eq

    r = R["UNSW"]
    keys, d_f1, p_f1 = family(r, "macro_f1")
    _, d_sh, p_sh = family(r, "recall_Shellcode")
    eq, p_eq = equivalence(r)
    rows = []
    for mo in DETECTORS:
        d = paired(r.s("real", "R1_ROS", mo, "macro_f1", "multiclass", 1.0), r.base(mo, "macro_f1", "multiclass"))
        rows.append(dict(method="Duplication", detector=mo, delta_vs_r0=d.mean(), ci_vs_r0=t_interval(d)))
    names = {"smote": "SMOTE", "copula": "Copula", "ctgan_e100": "CTGAN", "tvae_e100": "TVAE"}
    for m in ("smote", "copula", "ctgan_e100", "tvae_e100"):
        for mo in DETECTORS:
            d0 = r.delta(m, "R1", mo, "macro_f1", "multiclass", 1.0)
            rows.append(
                dict(
                    method=names[m],
                    detector=mo,
                    delta_vs_r0=d0.mean(),
                    ci_vs_r0=t_interval(d0),
                    delta_vs_dup=d_f1[(m, mo)].mean(),
                    ci_vs_dup=t_interval(d_f1[(m, mo)]),
                    p_holm=p_f1[(m, mo)],
                    equivalence=eq[(m, mo)],
                    p_tost_holm=p_eq[(m, mo)],
                    shellcode_delta_vs_dup=d_sh[(m, mo)].mean(),
                    shellcode_ci=t_interval(d_sh[(m, mo)]),
                    shellcode_p_holm=p_sh[(m, mo)],
                )
            )
    t = r.thr
    for mo in DETECTORS:
        cw = t[(t.regime == "R0_weighted") & (t.model == mo) & (t.task == "multiclass")].set_index("seed")
        ros = r.s("real", "R1_ROS", mo, "macro_f1", "multiclass", 1.0)
        d0, d1 = paired(cw.macro_f1, r.base(mo, "macro_f1", "multiclass")), paired(cw.macro_f1, ros)
        dsh = paired(cw.recall_Shellcode, r.s("real", "R1_ROS", mo, "recall_Shellcode", "multiclass", 1.0))
        rows.append(
            dict(
                method="Class weights",
                detector=mo,
                delta_vs_r0=d0.mean(),
                ci_vs_r0=t_interval(d0),
                delta_vs_dup=d1.mean(),
                ci_vs_dup=t_interval(d1),
                shellcode_delta_vs_dup=dsh.mean(),
                shellcode_ci=t_interval(dsh),
            )
        )
    cic_eq, cic_p = equivalence(R["CIC"])
    cic = pd.DataFrame(
        [
            dict(generator=names[m], detector=mo, equivalence=cic_eq[(m, mo)], p_tost_holm=cic_p[(m, mo)])
            for m, mo in cic_eq
        ]
    )
    return pd.DataFrame(rows), cic


def fig3_augmentation_levels(R: dict) -> pd.DataFrame:
    r, rows = R["UNSW"], []
    for mo in DETECTORS:
        for name, m, rg in [("Duplication", "real", "R1_ROS")] + [(g, mm, "R1") for g, mm in GENERATORS.items()]:
            for p in (0.25, 0.5, 1.0):
                d = paired(r.s(m, rg, mo, "macro_f1", "multiclass", p), r.base(mo, "macro_f1", "multiclass"))
                rows.append(
                    dict(detector=mo, method=name, added_pct=int(100 * p), delta_macro_f1=d.mean(), ci95=t_interval(d))
                )
    return pd.DataFrame(rows)


def table8_fidelity(R: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, taus = [], []
    for ds, r in R.items():
        f, p = r.fidelity, r.probe

        def per_class(m, col, f=f):
            return f[(f.method == m) & (f.group != "ALL")].groupby("seed")[col].mean()

        gen = {}
        for g, m in GENERATORS.items():
            pr = p[(p.method == m) & (p.group != "ALL_ATTACKS")].groupby("seed").probe_auc.mean()
            pooled = f[(f.method == m) & (f.group == "ALL")]
            row = dict(
                dataset=ds,
                generator=g,
                ks=per_class(m, "ks_mean").mean(),
                w1=per_class(m, "wass_mean").mean(),
                corr_gap=per_class(m, "corr_gap").mean(),
                probe_auc=pr.mean(),
                probe_ci=t_interval(pr),
                dist_ratio=pooled.dcr_ratio.mean(),
                copies_pct=100 * f[(f.method == m) & (f.group != "ALL")].groupby("group").exact_dup_share.mean().mean(),
                mia_auc=pooled.mia_auc.mean(),
                mia_ci=t_interval(pooled.mia_auc),
            )
            row["r3_macro_f1"] = np.mean([r.s(m, "R3", mo).mean() for mo in DETECTORS])
            row["r3_pr_auc"] = np.mean([r.s(m, "R3", mo, "pr_auc").mean() for mo in DETECTORS])
            rows.append(row)
            gen[g] = row
        G = pd.DataFrame(gen).T
        for metric in ("ks", "w1", "corr_gap", "probe_auc"):
            for util in ("r3_macro_f1", "r3_pr_auc"):
                taus.append(
                    dict(
                        dataset=ds,
                        fidelity=metric,
                        utility=util,
                        kendall_tau=stats.kendalltau(-G[metric].astype(float), G[util].astype(float)).statistic,
                    )
                )
    return pd.DataFrame(rows), pd.DataFrame(taus)


def fig4_ks_vs_utility(R: dict) -> pd.DataFrame:
    configs = ["smote", "copula", "ctgan_e50", "ctgan_e100", "ctgan_e300", "tvae_e50", "tvae_e100", "tvae_e300"]
    rows = []
    for ds, r in R.items():
        both = pd.concat([r.regimes, r.ablation])
        r3 = both[(both.seed == 0) & (both.regime == "R3") & (both.task == "binary")]
        for m in configs:
            g = r3[r3.method == m]
            ks = r.genlog[(r.genlog.seed == 0) & (r.genlog.method == m)].mean_ks.mean()
            rows.append(
                dict(
                    dataset=ds, config=m, mean_per_class_ks=ks, r3_macro_f1=g.macro_f1.mean(), r3_pr_auc=g.pr_auc.mean()
                )
            )
    out = pd.DataFrame(rows)
    out["spearman_macro_f1"] = (
        out.groupby("dataset")
        .apply(lambda g: stats.spearmanr(-g.mean_per_class_ks, g.r3_macro_f1).statistic)
        .reindex(out.dataset)
        .values
    )
    out["spearman_pr_auc"] = (
        out.groupby("dataset")
        .apply(lambda g: stats.spearmanr(-g.mean_per_class_ks, g.r3_pr_auc).statistic)
        .reindex(out.dataset)
        .values
    )
    return out


def robustness(R: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for ds, r in R.items():
        both = pd.concat([r.ablation, r.regimes[r.regimes.seed == 0]])
        for gen in ("ctgan", "tvae"):
            for rg in ("R2", "R3"):
                row = dict(dataset=ds, generator=gen, regime=rg)
                for mo in DETECTORS:
                    for ep in (50, 100, 300):
                        g = both[
                            (both.task == "binary")
                            & (both.method == f"{gen}_e{ep}")
                            & (both.regime == rg)
                            & (both.model == mo)
                        ]
                        row[f"{mo}_{ep}"] = float(g.macro_f1.iloc[0])
                rows.append(row)
    chrono_rows = []
    r = R["CIC"]
    for m, name in (("smote", "SMOTE"), ("copula", "Copula")):
        for rg in ("R2", "R3"):
            row = dict(generator=name, regime=rg)
            for mo in DETECTORS:
                for label, df in (("chrono", r.chrono), ("random", r.regimes[r.regimes.seed.isin([0, 1, 2])])):
                    b = df[df.task == "binary"]
                    base = b[b.regime == "R0"].set_index(["seed", "model"]).macro_f1
                    v = b[(b.method == m) & (b.regime == rg) & (b.model == mo)]
                    row[f"{mo}_{label}"] = float(np.mean([x.macro_f1 - base.loc[(x.seed, mo)] for x in v.itertuples()]))
            chrono_rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(chrono_rows)


def summary_numbers(R: dict) -> dict:
    out = {}
    mx = 0.0
    for _ds, r in R.items():
        for m in GENERATORS.values():
            for mo in DETECTORS:
                mix = (
                    r.rmix[(r.rmix.method == m) & (r.rmix.model == mo) & (r.rmix.task == "binary")]
                    .set_index("seed")
                    .macro_f1
                )
                mx = max(mx, abs(paired(mix, r.base(mo)).mean()))
    out["rmix_max_abs_delta"] = mx
    agree = total = 0
    largest = 0.0
    for _ds, r in R.items():
        for _, row in r.boot.iterrows():
            regime, p = row.regime.rsplit("_", 1)
            d = r.delta(row.method, regime, row.model, param=float(p) if regime in ("R1", "R1_ROS") else None)
            same = (abs(d.mean()) > t_interval(d)) == ((row.boot_lo > 0) or (row.boot_hi < 0))
            agree, total = agree + same, total + 1
            if not same:
                largest = max(largest, abs(d.mean()))
    out["bootstrap_agreement"] = f"{agree}/{total}"
    out["bootstrap_largest_disagreement"] = largest
    out["unsw_test_rows_with_other_training_label_pct"] = (
        100 * R["UNSW"].floor["share_test_rows_with_other_train_label"]
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = load(a.results)
    t7, t7_cic = table7_augmentation(R)
    t8, t8_tau = table8_fidelity(R)
    ab, ch = robustness(R)
    outputs = {
        "table1_cleaning_flow": table1_cleaning(R),
        "table3_per_class": table3_per_class(R),
        "table4_operating_point": table4_operating_point(R),
        "fig2_regime_effects": fig2_regime_effects(R),
        "table5_provenance": table5_provenance(R),
        "table6_thresholds": table6_thresholds(R),
        "table7_augmentation_unsw": t7,
        "table7_equivalence_cic": t7_cic,
        "fig3_augmentation_levels": fig3_augmentation_levels(R),
        "table8_fidelity_replication": t8,
        "table8_rank_agreement": t8_tau,
        "fig4_ks_vs_utility": fig4_ks_vs_utility(R),
        "robustness_training_length": ab,
        "robustness_chronological_split": ch,
    }
    for name, df in outputs.items():
        df.to_csv(os.path.join(a.out, f"{name}.csv"), index=False)
    summary = summary_numbers(R)
    with open(os.path.join(a.out, "summary.txt"), "w") as fh:
        for k, v in summary.items():
            fh.write(f"{k}: {v}\n")
    print(f"Wrote {len(outputs)} tables and summary.txt to {a.out}/")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
