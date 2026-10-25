"""End-to-end tests.

test_pipeline_on_toy_data runs cleaning, splitting, generation, training regimes and the control
experiments on small synthetic CSV files (SMOTE and Gaussian copula only, so no GPU is needed).
test_reproduce_tables checks that the paper's tables are regenerated from the bundled results.

Run with:  python -m pytest -q tests
"""

import os
import subprocess
import sys

import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(args, env):
    out = subprocess.run([sys.executable, *args], cwd=REPO, env=env, capture_output=True, text=True)
    assert out.returncode == 0, f"{' '.join(args)} failed:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}"
    return out.stdout


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("ids_root")
    e = dict(os.environ, IDS_ROOT=str(root), PYTHONPATH=REPO)
    run(["tests/make_toy_data.py", str(root / "raw_toy")], e)
    return e, root


@pytest.mark.parametrize("dataset", ["cicids2017", "unsw_nb15"])
def test_pipeline_on_toy_data(env, dataset):
    e, root = env
    src = str(root / "raw_toy" / dataset)
    run(["ids_data.py", "acquire", "--dataset", dataset, "--mode", "path", "--src", src], e)
    run(["ids_data.py", "all", "--dataset", dataset, "--seeds", "0"], e)
    run(["ids_synth.py", "generate", "--dataset", dataset, "--seeds", "0", "--methods", "smote", "copula"], e)
    run(
        [
            "ids_regimes.py",
            "run",
            "--dataset",
            dataset,
            "--seeds",
            "0",
            "--methods",
            "smote",
            "copula",
            "--models",
            "rf",
            "--regimes",
            "R0",
            "R1",
            "R1_ROS",
            "R2",
            "R2r",
            "R3",
            "--fracs",
            "1.0",
            "--out_tag",
            "_test",
        ],
        e,
    )
    res = pd.read_csv(root / "results" / dataset / "regimes_test" / "regimes_per_seed.csv")
    assert {"R0", "R1", "R1_ROS", "R2", "R2r", "R3"} <= set(res.regime)
    assert res["macro_f1"].between(0, 1).all()
    assert not res.duplicated(subset=["seed", "method", "regime", "param", "model", "task"]).any()
    for cmd in ("rmix", "provenance"):
        run(
            [
                "ids_extra.py",
                cmd,
                "--dataset",
                dataset,
                "--seeds",
                "0",
                "--methods",
                "smote",
                "copula",
                "--models",
                "rf",
                "--out_tag",
                "_test",
            ],
            e,
        )
    extra = root / "results" / dataset / "extra_test"
    prov = pd.read_csv(extra / "provenance_per_seed.csv")
    assert prov["heldout_syn_benign_flagged"].between(0, 1).all()
    assert len(pd.read_csv(extra / "rmix_per_seed.csv")) > 0


def test_reproduce_tables(tmp_path):
    out = tmp_path / "tables"
    run(["reproduce_tables.py", "--results", "results", "--out", str(out)], dict(os.environ))
    summary = (out / "summary.txt").read_text()
    assert "bootstrap_agreement: 100/108" in summary
    t7 = pd.read_csv(out / "table7_equivalence_cic.csv")
    assert (t7.equivalence == "equivalent").all(), "all generator-detector pairs are equivalent on CIC-IDS2017"
    t5 = pd.read_csv(out / "table5_provenance.csv")
    assert t5.delta_rmix.abs().max() < 0.008
