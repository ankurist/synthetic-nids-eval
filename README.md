# A Duplicate-Controlled Evaluation of Tabular Generators for Network Intrusion Detection

Code, per-seed results and figure sources accompanying the paper of the same title.

The study evaluates whether synthetic tabular data from four generators (SMOTE, a Gaussian
copula, CTGAN and TVAE) improves network intrusion detection on CIC-IDS2017 and UNSW-NB15,
under a protocol that removes exact duplicates and train–test overlaps before splitting,
fits all preprocessing on training rows only and scores every detector on real test data.
Training regimes separate generator quality from data provenance:

| Regime | Training data | Purpose |
|---|---|---|
| R0 | real rows only | reference |
| R1 | real + synthetic minority rows (25, 50, 100% of class size) | augmentation |
| ROS (`R1_ROS` in code) | real + duplicated real minority rows | duplication control |
| R2 | real benign + synthetic attacks | class-aligned provenance |
| R2r | synthetic benign + real attacks | mirror of R2 |
| R3 | all classes synthetic | train on synthetic, test on real |
| RMIX | every class half real, half synthetic | provenance uninformative |

## Repository structure

```
ids_data.py          acquisition, inspection, cleaning, locked splits (SHA-256 manifests)
ids_eval.py          shared preprocessing and metrics; real-only baseline
ids_synth.py         per-class generators (SMOTE, Gaussian copula, CTGAN, TVAE)
ids_regimes.py       training regimes R0-R3 and the real-versus-synthetic probe
ids_fidelity.py      fidelity (KS, Wasserstein, correlation gap) and replication metrics
ids_extra.py         controls: RMIX, provenance test, tuned thresholds, class weights, MLP,
                     bootstrap over test rows, label-conflict floor
ids_analysis.py      summary tables and diagnostic figures
reproduce_tables.py  regenerates every table and figure data set of the paper from results/
results/             per-seed results, dataset cards and split manifests (both datasets)
tables/              output of reproduce_tables.py
figures/             draw.io sources of Figs. 1-4
tests/               end-to-end tests on small synthetic CSV files
```

## Installation

Python 3.10 or later.

```
pip install -r requirements.txt          # add -r requirements-dev.txt for the tests
```

PyTorch and `ctgan` are needed only for CTGAN and TVAE. A GPU is optional; XGBoost uses one
when `--xgb_gpu` is passed.

## Reproducing the paper's tables (no training required)

```
python reproduce_tables.py --results results --out tables
```

| Paper | Output file |
|---|---|
| Table 1 (cleaning flow) | `table1_cleaning_flow.csv` |
| Table 3 (per-class supports, F1, replication) | `table3_per_class.csv` |
| Table 4 (operating point and ranking) | `table4_operating_point.csv` |
| Fig. 2 (regime effects) | `fig2_regime_effects.csv` |
| Table 5 (provenance controls) | `table5_provenance.csv` |
| Table 6 (default vs tuned thresholds) | `table6_thresholds.csv` |
| Table 7 (augmentation vs duplication) | `table7_augmentation_unsw.csv`, `table7_equivalence_cic.csv` |
| Fig. 3 (augmentation levels) | `fig3_augmentation_levels.csv` |
| Table 8 (fidelity, replication, rank agreement) | `table8_fidelity_replication.csv`, `table8_rank_agreement.csv` |
| Fig. 4 (KS vs utility) | `fig4_ks_vs_utility.csv` |
| Robustness checks (Sect. 4.7) | `robustness_training_length.csv`, `robustness_chronological_split.csv` |
| Summary statistics in the text | `summary.txt` |

Equivalence to duplication is tested with two one-sided t-tests at a margin of ±0.01
macro-F1; paired comparisons are Holm-adjusted within each family of eight
generator–detector pairs.

## Running the full pipeline

The datasets are not redistributed. Obtain them from their providers:

* CIC-IDS2017 (labelled flows): https://www.unb.ca/cic/datasets/ids-2017.html
* UNSW-NB15 (official training and testing sets): https://research.unsw.edu.au/projects/unsw-nb15-dataset

Commands for one dataset (`D` = `cicids2017` or `unsw_nb15`); outputs go to `$IDS_ROOT`
(default `./ids_project`):

```
python ids_data.py acquire   --dataset D --mode path --src <downloaded files>
python ids_data.py all       --dataset D --seeds 0 1 2 3 4
python ids_synth.py generate --dataset D --seeds 0 1 2 3 4 --methods smote copula ctgan tvae --epochs 100
python ids_regimes.py run    --dataset D --seeds 0 1 2 3 4 --methods smote copula ctgan_e100 tvae_e100 --out_tag _final
python ids_regimes.py probe  --dataset D --seeds 0 1 2 3 4 --methods smote copula ctgan_e100 tvae_e100 --out_tag _final
python ids_fidelity.py run   --dataset D --seeds 0 1 2 3 4 --out_tag _final
for c in rmix provenance threshold bootstrap floor; do python ids_extra.py $c --dataset D; done
python ids_extra.py mlp --dataset D --seeds 0 1 2
```

Training-length ablation (seed 0): `ids_synth.py generate --methods ctgan tvae --epochs 50 300`,
then `ids_regimes.py run --methods ctgan_e50 ctgan_e300 tvae_e50 tvae_e300 --regimes R0 R2 R3 --out_tag _ablation`.
Chronological split (CIC-IDS2017): `ids_data.py split --split_mode chrono`, then pass `--split chrono`.

Every random component is seeded; splits are stored with SHA-256 manifests of their row
identifiers (seed 0 included in `results/`) and regenerate identically from the seed. All runs
are resumable, and re-running a seed reproduces its metrics exactly. On a single GPU, fitting
CTGAN and TVAE takes about 13–18 minutes per seed; SMOTE and the copula take under a minute.

## Tests

```
python -m pytest -q tests
```

The tests run the full pipeline on small synthetic CSV files (SMOTE and the copula, no GPU
needed) and check that `reproduce_tables.py` regenerates the paper's summary statistics.

## License

MIT (see `LICENSE`). The datasets remain subject to their providers' terms of use, which
require citing the papers listed on their pages.
