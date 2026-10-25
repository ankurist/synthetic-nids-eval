#!/usr/bin/env python
"""Dataset acquisition, inspection, cleaning and splitting.

Usage:
    python ids_data.py acquire --dataset cicids2017 --mode path --src <CSV or ZIP files>
    python ids_data.py acquire --dataset unsw_nb15  --mode path --src UNSW_NB15_training-set.csv UNSW_NB15_testing-set.csv
    python ids_data.py inspect --dataset cicids2017
    python ids_data.py clean   --dataset cicids2017
    python ids_data.py split   --dataset cicids2017 --seeds 0 1 2 3 4
    python ids_data.py all     --dataset unsw_nb15                  # clean + split

Other flow datasets (e.g. NF-UNSW-NB15-v2) are supported through --label_col,
--benign_value and --drop_cols, e.g.
    python ids_data.py all --dataset nf_unsw_v2 --label_col Attack --benign_value Benign \
                           --drop_cols Label IPV4_SRC_ADDR IPV4_DST_ADDR

Cleaning protocol:
  * identifier columns (flow IDs, addresses, source ports, timestamps, row ids) are dropped;
  * exact duplicates (all features and the label) are removed before any split;
  * labels are mapped by an explicit function that raises on unknown labels
    ('Web Attack' is matched before 'Brute Force');
  * rows whose feature vectors occur with both benign and attack labels are dropped;
    attack-versus-attack conflicts are kept and counted;
  * for datasets with official partitions, test rows identical to training rows are removed;
  * splits are stratified on the multi-class category and stored with SHA-256 manifests;
    scalers are fitted later, on training rows only (see ids_eval.py);
  * every count is recorded in dataset_card.json;
  * files are read in chunks as float32, with per-file de-duplication and class caps.

Acquisition modes: `path` (local files, default); `drive` and `upload` are optional
Google Colab helpers. All outputs are written under $IDS_ROOT (default ./ids_project).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.environ.get("IDS_ROOT", "ids_project")


def path(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def norm_col(c) -> str:
    """Strip BOM (real or latin-1-mis-decoded), collapse whitespace."""
    c = str(c).replace("\ufeff", "").replace("\u00ef\u00bb\u00bf", "")
    return re.sub(r"\s+", " ", c).strip()


# names that are row counters / identifiers in some dataset versions -> never a feature
INDEX_NAMES = {"id", "index", "unnamed: 0", "unnamed: 0.1", "no.", "no", "row_id", "rowid", "#", "idx"}


# --------------------------------------------------------------------------
# 1. Label mappers (explicit, fail loudly)
# --------------------------------------------------------------------------
def map_cicids(label: str) -> str:
    l = label.strip()
    if l.upper() == "BENIGN":
        return "Benign"
    if "Web Attack" in l:  # MUST come before Brute Force
        return "Web Attack"
    if "DoS" in l or "DDoS" in l:
        return "DoS/DDoS"
    if "PortScan" in l:
        return "PortScan"
    if "Patator" in l or "Brute Force" in l:
        return "Brute Force"
    if l in ("Bot", "Infiltration", "Heartbleed"):
        return "Rare Attack"
    raise ValueError(f"CIC-IDS2017: unmapped label {label!r}")


_UNSW = {
    "normal": "Benign",
    "benign": "Benign",
    "nan": "Benign",
    "": "Benign",
    "generic": "Generic",
    "exploits": "Exploits",
    "fuzzers": "Fuzzers",
    "dos": "DoS",
    "reconnaissance": "Reconnaissance",
    "analysis": "Analysis",
    "backdoor": "Backdoor",
    "backdoors": "Backdoor",
    "shellcode": "Shellcode",
    "worms": "Worms",
}


def map_unsw(label: str) -> str:
    key = label.strip().lower()
    if key not in _UNSW:
        raise ValueError(f"UNSW-NB15: unmapped attack_cat {label!r}")
    return _UNSW[key]


def make_generic_mapper(benign_value: str):
    b = benign_value.strip().lower()

    def _m(label: str) -> str:
        return "Benign" if label.strip().lower() == b else label.strip()

    return _m


# --------------------------------------------------------------------------
# 2. Dataset specifications
# --------------------------------------------------------------------------
# 'drop' = shortcut / identifier columns (compared lower-case, whitespace-normalised)
CICIDS_CAPS = {
    "Benign": 50000,
    "DoS/DDoS": 30000,
    "PortScan": 20000,
    "Brute Force": 10000,
    "Rare Attack": 2000,
    "Web Attack": 10000,
}

SPECS = {
    "cicids2017": dict(
        label_col="label",
        mapper=map_cicids,
        cat_cols=[],
        drop=[
            "flow id",
            "source ip",
            "src ip",
            "destination ip",
            "dst ip",
            "source port",
            "src port",
            "unnamed: 0",
            "simillarhttp",
            "fwd header length.1",
        ],
        time_col="timestamp",
        official_split=False,
        caps=CICIDS_CAPS,
    ),
    "unsw_nb15": dict(
        label_col="attack_cat",
        mapper=map_unsw,
        cat_cols=["proto", "service", "state"],
        drop=["id", "label"],
        time_col=None,
        official_split=True,
        caps=None,
    ),
}


def get_spec(dataset: str, args=None) -> dict:
    if dataset in SPECS and not (args and getattr(args, "label_col", None)):
        spec = dict(SPECS[dataset])
    else:  # generic dataset
        if not (args and args.label_col):
            raise SystemExit("Generic dataset: pass --label_col (and --benign_value).")
        spec = dict(
            label_col=args.label_col.lower(),
            mapper=make_generic_mapper(args.benign_value or "Benign"),
            cat_cols=[c.lower() for c in (args.cat_cols or [])],
            drop=[
                "flow id",
                "source ip",
                "src ip",
                "destination ip",
                "dst ip",
                "source port",
                "src port",
                "unnamed: 0",
                "ipv4_src_addr",
                "ipv4_dst_addr",
                "l4_src_port",
            ],
            time_col=(args.time_col.lower() if args.time_col else None),
            official_split=False,
            caps=None,
        )
    if args is not None:
        if getattr(args, "drop_cols", None):
            spec["drop"] = list(spec["drop"]) + [c.lower() for c in args.drop_cols]
        if getattr(args, "cat_cols", None):
            spec["cat_cols"] = [c.lower() for c in args.cat_cols]
        if getattr(args, "caps", None):
            spec["caps"] = json.loads(args.caps)
        if getattr(args, "no_caps", False):
            spec["caps"] = None
    return spec


# --------------------------------------------------------------------------
# 3. ACQUIRE  (upload / drive / path)
# --------------------------------------------------------------------------
def _safe_extract(zpath: str, dest: str) -> None:
    with zipfile.ZipFile(zpath) as z:
        for m in z.namelist():
            target = os.path.abspath(os.path.join(dest, m))
            if not target.startswith(os.path.abspath(dest) + os.sep):
                raise RuntimeError(f"Unsafe path in zip: {m}")
        z.extractall(dest)


def acquire(dataset: str, mode: str, src: list[str] | None = None, fresh: bool = False) -> None:
    dest = path("raw", dataset)
    if fresh and os.path.isdir(dest):
        print(f"--fresh: deleting old raw files in {dest}")
        shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)
    if mode == "upload":
        try:
            from google.colab import files  # type: ignore
        except ImportError:
            raise SystemExit(
                "--mode upload works only inside Google Colab. " "Use --mode path --src <file-or-folder> locally."
            ) from None
        print("Select the CSV or ZIP files of this dataset (for large files, use --mode path or --mode drive).")
        up = files.upload()
        for name in up:
            shutil.move(name, os.path.join(dest, os.path.basename(name)))
    elif mode in ("drive", "path"):
        if mode == "drive":
            try:
                from google.colab import drive  # type: ignore

                if not os.path.isdir("/content/drive/MyDrive"):
                    drive.mount("/content/drive")
            except ImportError:
                pass  # not in Colab; --src may still be a normal path
        if not src:
            raise SystemExit("--src is required for --mode drive/path")
        for s in src:
            s = os.path.expanduser(s)
            if os.path.isdir(s):
                for f in glob.glob(os.path.join(s, "**", "*"), recursive=True):
                    if os.path.isfile(f) and f.lower().endswith((".csv", ".zip", ".gz")):
                        shutil.copy2(f, dest)
            elif os.path.isfile(s):
                shutil.copy2(s, dest)
            else:
                raise SystemExit(f"Not found: {s}")
    else:
        raise SystemExit("mode must be upload | drive | path")
    for z in glob.glob(os.path.join(dest, "*.zip")):
        print("extracting", os.path.basename(z))
        _safe_extract(z, dest)
    print(f"\nFiles now in {dest}:")
    for f in sorted(glob.glob(os.path.join(dest, "**", "*.csv"), recursive=True)):
        print(f"  {os.path.relpath(f, dest):60s} {os.path.getsize(f)/1e6:9.1f} MB")


def _csv_files(dataset: str) -> list[str]:
    files = sorted(glob.glob(path("raw", dataset, "**", "*.csv"), recursive=True))
    if not files:
        raise SystemExit(f"No CSV files in {path('raw', dataset)}. Run `acquire` first.")
    return files


# --------------------------------------------------------------------------
# 4. INSPECT
# --------------------------------------------------------------------------
def inspect(dataset: str, nrows: int = 200_000) -> None:
    for f in _csv_files(dataset):
        h = pd.read_csv(f, nrows=nrows, encoding="latin-1", low_memory=False)
        h.columns = [norm_col(c) for c in h.columns]
        print(f"\n=== {os.path.basename(f)}  ({h.shape[1]} columns, first {len(h):,} rows)")
        print("columns:", list(h.columns) if h.shape[1] <= 90 else list(h.columns)[:8] + ["..."] + list(h.columns)[-4:])
        print("index-like columns that will be dropped:", [c for c in h.columns if c.lower() in INDEX_NAMES])
        for c in h.columns:
            if c.lower() in ("label", "attack", "attack_cat", "class", "attack_type"):
                print(f"label-like column {c!r}:")
                print(h[c].astype(str).str.strip().value_counts().head(20).to_string())
        obj = [c for c in h.columns if h[c].dtype == object or str(h[c].dtype) in ("str", "string")]
        print("non-numeric columns:", obj[:15])


# --------------------------------------------------------------------------
# 5. CLEAN
# --------------------------------------------------------------------------
def _clean_file(f: str, spec: dict, chunksize: int = 150_000):
    """Read one CSV in chunks -> float32, drop inf/NaN, strip labels. Returns
    (dataframe after exact de-dup, stats). Peak RAM ~ one file."""
    stats = dict(rows_read=0, dropped_nan_inf=0, dropped_duplicates=0, dropped_nan_inf_by_label={})
    parts = []
    for ch in pd.read_csv(f, chunksize=chunksize, encoding="latin-1", low_memory=False):
        ch.columns = [norm_col(c) for c in ch.columns]
        ch = ch.loc[:, ~ch.columns.duplicated()]
        low = {c.lower(): c for c in ch.columns}
        if spec["label_col"] not in low:
            raise SystemExit(
                f"{os.path.basename(f)}: label column {spec['label_col']!r} not found. "
                f"Columns: {list(ch.columns)[:12]} ... Use `inspect`, then --label_col."
            )
        lab = low[spec["label_col"]]
        drop = [low[d] for d in (set(spec["drop"]) | INDEX_NAMES) if d in low and low[d] != lab]
        ts = low.get(spec["time_col"]) if spec.get("time_col") else None
        cats = [low[c] for c in spec["cat_cols"] if c in low]
        ch = ch.drop(columns=drop).rename(columns={lab: "Label"})
        if ts:
            ch = ch.rename(columns={ts: "_ts"})
        feats = [c for c in ch.columns if c not in ("Label", "_ts") and c not in cats]
        ch[feats] = ch[feats].apply(pd.to_numeric, errors="coerce").astype("float64")
        # auto-detect row counters (1,2,3,... in file order): a counter is a shortcut, not a feature
        counters = [
            c for c in feats if len(ch) > 50 and ch[c].notna().all() and bool((np.diff(ch[c].values) == 1).all())
        ]
        if counters:
            stats.setdefault(
                "dropped_counter_like_columns", sorted(set(stats.get("dropped_counter_like_columns", []) + counters))
            )
            ch = ch.drop(columns=counters)
            feats = [c for c in feats if c not in counters]
        ch[feats] = ch[feats].astype("float32")
        ch[feats] = ch[feats].replace([np.inf, -np.inf], np.nan)
        for c in cats:
            ch[c] = ch[c].astype(str).str.strip()
        ch["Label"] = ch["Label"].astype(str).str.strip()
        n0 = len(ch)
        bad = ch[feats].isna().any(axis=1)
        for l, n in ch.loc[bad, "Label"].value_counts().items():  # audit: WHICH labels are lost?
            stats["dropped_nan_inf_by_label"][l] = stats["dropped_nan_inf_by_label"].get(l, 0) + int(n)
        ch = ch.loc[~bad]
        stats["rows_read"] += n0
        stats["dropped_nan_inf"] += n0 - len(ch)
        parts.append(ch)
    d = pd.concat(parts, ignore_index=True)
    del parts
    feats = [c for c in d.columns if c not in ("Label", "_ts")]
    n0 = len(d)
    d = d.drop_duplicates(subset=feats + ["Label"]).reset_index(drop=True)
    stats["dropped_duplicates"] = n0 - len(d)
    m = {l: spec["mapper"](l) for l in d["Label"].unique()}
    d["category"] = d["Label"].map(m)
    stats["columns"] = sorted(c for c in d.columns if c not in ("Label", "_ts", "category"))
    return d, stats


def _row_hash(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return pd.util.hash_pandas_object(df[cols], index=False)


def clean(dataset: str, spec: dict, seed: int = 0, min_class: int = 30, conflicts: str = "binary") -> None:
    out = path("processed", dataset)
    os.makedirs(out, exist_ok=True)
    kept, card = [], dict(dataset=dataset, files={}, natural_counts_after_dedup={})
    colsets: dict[str, list[str]] = {}
    shutil.rmtree(path("splits", dataset), ignore_errors=True)  # old splits would be stale
    if os.path.isdir(path("results", dataset)):
        print(f"NOTE: results in {path('results', dataset)} were computed on previously cleaned data; re-run them.")
    caps = spec["caps"]
    for f in _csv_files(dataset):
        base = os.path.basename(f)
        if spec["official_split"] and not re.search(r"train|test", base, re.I):
            print("skipping (not an official train/test file):", base)
            continue
        d, st = _clean_file(f, spec)
        d["source_file"] = base
        if spec["official_split"]:
            d["_split"] = "train" if re.search(r"train", base, re.I) else "test"
        st["rows_after_dedup"] = len(d)
        colsets[base] = st.pop("columns")
        card["files"][base] = st
        for c, n in d["category"].value_counts().items():
            card["natural_counts_after_dedup"][c] = card["natural_counts_after_dedup"].get(c, 0) + int(n)
        if caps and not spec["official_split"]:  # per-file cap bounds memory
            d = pd.concat(
                [g.sample(n=min(caps.get(c, len(g)), len(g)), random_state=seed) for c, g in d.groupby("category")],
                ignore_index=True,
            )
        kept.append(d)
        print(
            f"{base}: read {st['rows_read']:,} | nan/inf {st['dropped_nan_inf']:,} | "
            f"dups {st['dropped_duplicates']:,} | kept {len(d):,}"
        )
    ref_name, ref = next(iter(colsets.items()))
    odd = {b: sorted(set(c) ^ set(ref)) for b, c in colsets.items() if set(c) != set(ref)}
    if odd:
        raise SystemExit(
            "STOP: files do not have the same columns (a foreign or toy file in raw/?). "
            f"Reference file {ref_name}. Differences (symmetric): "
            + "; ".join(f"{b}: {d[:6]}..." for b, d in odd.items())
            + "\nDelete the odd file(s) or re-run `acquire --fresh`."
        )
    df = pd.concat(kept, ignore_index=True)
    del kept
    cat_cols = [c for c in df.columns if c in spec["cat_cols"] or c.lower() in spec["cat_cols"]]
    key = [c for c in df.columns if c not in ("_ts", "source_file", "_split", "category")]

    if spec["official_split"]:
        # de-dup inside each split, then remove test rows that also occur in train
        parts = []
        for _sp, g in df.groupby("_split"):
            parts.append(g.drop_duplicates(subset=key))
        df = pd.concat(parts, ignore_index=True)
        trh = set(_row_hash(df[df["_split"] == "train"], key).tolist())
        h = _row_hash(df, key)
        overlap = (df["_split"] == "test") & h.isin(trh)
        card["test_rows_removed_because_identical_to_train"] = int(overlap.sum())
        df = df[~overlap].reset_index(drop=True)
    else:
        n0 = len(df)
        df = df.drop_duplicates(subset=key).reset_index(drop=True)
        card["dups_removed_across_files"] = n0 - len(df)

    # identical feature vectors with DIFFERENT labels.  Two kinds:
    #  binary conflict  = same features labelled benign AND attack  -> true label noise, always wrong for one copy
    #  category conflict = same features, different attack names     -> ambiguity inside the attack family
    fkey = [c for c in key if c != "Label"]
    h = _row_hash(df, fkey)
    n_cat = df.groupby(h)["category"].transform("nunique")
    n_bin = (df["category"] != "Benign").groupby(h).transform("nunique")
    conf_cat, conf_bin = n_cat > 1, n_bin > 1
    card["rows_conflicting_any_category"] = int(conf_cat.sum())
    card["rows_conflicting_benign_vs_attack"] = int(conf_bin.sum())
    card["conflicting_rows_by_category"] = {
        str(k): int(v) for k, v in df.loc[conf_cat, "category"].value_counts().items()
    }
    card["class_counts_before_conflict_drop"] = {str(k): int(v) for k, v in df["category"].value_counts().items()}
    drop_mask = {"binary": conf_bin, "all": conf_cat, "keep": conf_cat & False}[conflicts]
    card["conflict_policy"] = conflicts
    card["conflicting_rows_dropped"] = int(drop_mask.sum())
    if drop_mask.any():
        df = df.loc[~drop_mask].reset_index(drop=True)

    # merge tiny classes so that stratified splitting is possible; log it
    vc = df["category"].value_counts()
    tiny = [c for c, n in vc.items() if n < min_class and c != "Benign"]
    if tiny:
        print(f"WARNING: merging classes with < {min_class} rows into 'Rare Attack': {tiny}")
        df.loc[df["category"].isin(tiny), "category"] = "Rare Attack"
    card["merged_into_rare_attack"] = tiny

    num_now = [
        c for c in df.columns if c not in ("Label", "category", "_ts", "source_file", "_split") and c not in cat_cols
    ]
    assert not df[num_now].isna().any().any(), "NaN in feature columns after cleaning"
    df["binary"] = (df["category"] != "Benign").astype("int8")
    df["row_id"] = np.arange(len(df), dtype="int64")
    for c in cat_cols + ["category", "source_file"] + (["_split"] if "_split" in df else []):
        df[c] = df[c].astype("category")
    feats = [
        c
        for c in df.columns
        if c not in ("Label", "category", "binary", "row_id", "source_file", "_ts", "_split") and c not in cat_cols
    ]
    classes = ["Benign"] + sorted(c for c in df["category"].unique() if c != "Benign")
    schema = dict(numeric=feats, categorical=cat_cols, classes=classes, has_timestamp="_ts" in df.columns)
    json.dump(schema, open(os.path.join(out, "schema.json"), "w"), indent=1)
    df.to_parquet(os.path.join(out, "clean.parquet"), index=False)
    card.update(
        n_rows=len(df),
        n_numeric_features=len(feats),
        n_categorical_features=len(cat_cols),
        class_counts_in_clean_file=df["category"].value_counts().to_dict(),
        attack_share=float(df["binary"].mean()),
        caps_used=caps,
    )
    json.dump(card, open(os.path.join(out, "dataset_card.json"), "w"), indent=1, default=str)
    print("\nFinal class counts:\n" + df["category"].value_counts().to_string())
    print(f"features: {len(feats)} numeric + {len(cat_cols)} categorical | wrote {out}/clean.parquet")


def parse_cic_time(ts: pd.Series) -> pd.Series:
    """CIC-IDS2017 timestamps: mixed formats (with/without seconds), day-first, and NO AM/PM marker.
    The capture ran about 08:00-17:00, so hours 1-7 are afternoon (e.g. '3:56' = 15:56): +12h."""
    t = pd.to_datetime(ts, errors="coerce", dayfirst=True, format="mixed")
    h = t.dt.hour
    return t.where(~((h >= 1) & (h < 8)), t + pd.Timedelta(hours=12))


# --------------------------------------------------------------------------
# 6. SPLIT  (locked, stratified on category, 5 seeds)
# --------------------------------------------------------------------------
def make_splits(
    dataset: str,
    spec: dict,
    seeds: list[int],
    test_frac=0.2,
    val_frac=0.1,
    split_mode: str = "random",
    min_class: int = 30,
) -> None:
    df = pd.read_parquet(path("processed", dataset, "clean.parquet"))
    if split_mode == "chrono" and "_ts" not in df.columns:
        raise SystemExit(
            "chrono split needs a Timestamp column (use the GeneratedLabelledFlows "
            "version of CIC-IDS2017). Otherwise use --split_mode random."
        )
    for seed in seeds:
        if spec["official_split"] and split_mode != "pooled":
            tr_all = df[df["_split"] == "train"]
            te = df[df["_split"] == "test"]
        else:
            caps = spec["caps"] or {}
            sub = pd.concat(
                [
                    g.sample(n=min(caps.get(c, len(g)), len(g)), random_state=seed)
                    for c, g in df.groupby("category", observed=True)
                ],
                ignore_index=True,
            )
            if split_mode in ("random", "pooled"):
                tr_all, te = train_test_split(sub, test_size=test_frac, random_state=seed, stratify=sub["category"])
            else:  # chronological INSIDE each class: earliest 80% train, latest 20% test
                ts = parse_cic_time(sub["_ts"].astype(str))
                if ts.isna().mean() > 0.01:
                    raise SystemExit(
                        "Timestamps could not be parsed. Examples: " + str(sub["_ts"][ts.isna()].head(5).tolist())
                    )
                sub = sub.assign(_t=ts).sort_values("_t")
                tr_idx, te_idx = [], []
                for _c, g in sub.groupby("category", observed=True):
                    k = int(round(len(g) * (1 - test_frac)))
                    tr_idx += list(g.index[:k])
                    te_idx += list(g.index[k:])
                tr_all, te = sub.loc[tr_idx], sub.loc[te_idx]
        tr, va = train_test_split(tr_all, test_size=val_frac, random_state=seed, stratify=tr_all["category"])
        d = path("splits", dataset, "" if split_mode == "random" else split_mode, f"seed{seed}")
        os.makedirs(d, exist_ok=True)
        man = dict(seed=seed, split_mode=split_mode, official=bool(spec["official_split"] and split_mode != "pooled"))
        for name, part in (("train", tr), ("val", va), ("test", te)):
            part = part.drop(columns=[c for c in ("_t", "_split", "_ts", "source_file") if c in part.columns])
            part.reset_index(drop=True).to_parquet(os.path.join(d, f"{name}.parquet"), index=False)
            man[name] = dict(
                n=len(part),
                by_class={str(k): int(v) for k, v in part["category"].value_counts().items()},
                sha256=hashlib.sha256(np.sort(part["row_id"].values).tobytes()).hexdigest(),
            )
        # leakage guard: no row_id in two splits
        ids = [
            set(pd.read_parquet(os.path.join(d, f"{n}.parquet"), columns=["row_id"])["row_id"])
            for n in ("train", "val", "test")
        ]
        assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "SPLIT OVERLAP!"
        json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=1)
        print(
            f"seed {seed}: train {man['train']['n']:,} | val {man['val']['n']:,} | test {man['test']['n']:,}"
            f" | attack share in test {np.mean(te['binary']):.3f}"
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["acquire", "inspect", "clean", "split", "all"])
    ap.add_argument("--dataset", required=True, help="cicids2017 | unsw_nb15 | any new name (generic)")
    ap.add_argument("--mode", choices=["upload", "drive", "path"], default="path")
    ap.add_argument("--src", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument(
        "--split_mode",
        choices=["random", "chrono", "pooled"],
        default="random",
        help="random = official partition if the dataset has one, else stratified random; "
        "pooled = pool everything and re-split randomly; chrono = per-class time order",
    )
    ap.add_argument("--min_class", type=int, default=30)
    ap.add_argument("--caps", help="JSON, e.g. '{\"Benign\": 50000}'")
    ap.add_argument("--no_caps", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="acquire: delete old raw files of this dataset first")
    ap.add_argument(
        "--conflicts",
        choices=["binary", "all", "keep"],
        default="binary",
        help="identical features, different labels: drop benign-vs-attack contradictions (default), "
        "drop any contradiction, or keep all",
    )
    ap.add_argument("--label_col")
    ap.add_argument("--benign_value")
    ap.add_argument("--drop_cols", nargs="*")
    ap.add_argument("--cat_cols", nargs="*")
    ap.add_argument("--time_col")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "acquire":
        acquire(a.dataset, a.mode, a.src, a.fresh)
        return
    if a.cmd == "inspect":
        inspect(a.dataset)
        return
    spec = get_spec(a.dataset, a)
    if a.cmd in ("clean", "all"):
        clean(a.dataset, spec, min_class=a.min_class, conflicts=a.conflicts)
    if a.cmd in ("split", "all"):
        make_splits(a.dataset, spec, a.seeds, split_mode=a.split_mode, min_class=a.min_class)


if __name__ == "__main__":
    main()
