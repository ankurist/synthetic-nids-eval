#!/usr/bin/env python
"""Generate small synthetic CSV files that mimic the schemas of CIC-IDS2017 and UNSW-NB15.

Used by the test suite to exercise the full pipeline in seconds, without the real datasets.

Usage:
    python tests/make_toy_data.py [output_dir]        # default: toy_raw
Creates <output_dir>/cicids2017/*.csv and <output_dir>/unsw_nb15/*.csv.
"""

import os
import numpy as np, pandas as pd

rng = np.random.default_rng(0)


def blob(n, shift, k=10):
    return rng.normal(shift, 1.0, size=(n, k)).round(3)


def cicids(outdir):
    os.makedirs(outdir, exist_ok=True)
    classes = {
        "BENIGN": 3000,
        "DoS Hulk": 800,
        "DDoS": 800,
        "PortScan": 700,
        "FTP-Patator": 300,
        "SSH-Patator": 300,
        "Web Attack \x96 Brute Force": 150,
        "Web Attack \x96 XSS": 60,
        "Bot": 120,
        "Infiltration": 40,
    }
    for day, (a, b) in zip(["Mon", "Tue", "Wed"], [(0, 4), (4, 8), (8, 10)]):
        frames = []
        for i, (lab, n) in enumerate(list(classes.items())):
            if not (a <= i < b or lab == "BENIGN"):
                continue
            X = blob(n // 3 if lab == "BENIGN" else n, shift=i * 0.7)
            d = pd.DataFrame(X, columns=[f" Feat {j}" for j in range(10)])
            d[" Destination Port"] = rng.integers(1, 65000, len(d))
            d["Flow Bytes/s"] = np.abs(X[:, 0]) * 100
            d[" Label"] = lab
            d["Flow ID"] = [f"id{i}_{k}" for k in range(len(d))]
            d[" Source IP"] = "10.0.0.1"
            tt = pd.date_range("2017-07-03 14:00", periods=len(d), freq="7s")  # afternoon, no AM/PM in file
            d[" Timestamp"] = [
                x.strftime("%d/%m/%Y %H:%M:%S") if k % 2 else x.strftime("%-d/%-m/%Y %-I:%M") for k, x in enumerate(tt)
            ]  # MIXED formats like the real files
            frames.append(d)
        d = pd.concat(frames, ignore_index=True)
        d = pd.concat([d, d.sample(120, random_state=1)])  # exact duplicates
        c = d[d[" Label"] != "BENIGN"].sample(8, random_state=3).copy()
        c[" Label"] = "BENIGN"
        d = pd.concat([d, c])  # benign-vs-attack label conflicts
        d.loc[d.sample(10, random_state=2).index, "Flow Bytes/s"] = np.inf  # inf rows
        d.to_csv(f"{outdir}/{day}-WorkingHours.pcap_ISCX.csv", index=False, encoding="latin-1")


def unsw(outdir):
    os.makedirs(outdir, exist_ok=True)
    cats = {
        "Normal": 2000,
        "Generic": 900,
        "Exploits": 600,
        "Fuzzers": 400,
        "DoS": 200,
        "Reconnaissance": 250,
        "Analysis": 60,
        "Backdoor": 50,
        "Shellcode": 60,
        "Worms": 35,
    }
    for name, scale in (("training-set", 1.0), ("testing-set", 0.5)):
        frames = []
        for i, (c, n) in enumerate(cats.items()):
            n = max(int(n * scale), 6)
            d = pd.DataFrame(blob(n, i * 0.6, 8), columns=[f"f{j}" for j in range(8)])
            d["dur"] = np.abs(d["f0"])
            d["proto"] = rng.choice(["tcp", "udp", "icmp"] if c != "Normal" else ["tcp", "udp"], n)
            d["service"] = rng.choice(["-", "http", "dns"], n)
            d["state"] = rng.choice(["FIN", "INT", "CON"], n)
            d["attack_cat"] = c
            d["label"] = int(c != "Normal")
            frames.append(d)
        d = pd.concat(frames, ignore_index=True)
        if name == "testing-set":  # leak some train rows into test to exercise the overlap guard
            tr = pd.read_csv(f"{outdir}/UNSW_NB15_training-set.csv").drop(columns=["id"]).sample(25, random_state=0)
            d = pd.concat([d, tr], ignore_index=True)
        d.insert(0, "id", range(1, len(d) + 1))
        d.to_csv(f"{outdir}/UNSW_NB15_{name}.csv", index=False)


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "toy_raw"
    cicids(os.path.join(root, "cicids2017"))
    unsw(os.path.join(root, "unsw_nb15"))
    print(f"Wrote {root}/cicids2017 and {root}/unsw_nb15")
