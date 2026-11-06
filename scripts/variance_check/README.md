# Replication across machines

The cross-machine experiment asks whether the conclusions survive a re-run outside the controlled environment. Per-run
values do not transfer between arithmetic paths, while the distributional statistics do (Section 3.6 and Appendix H of
the article).

| Script | What it answers |
|---|---|
| `cpu_probe.py` | Which reproducibility cell is this machine in? Reports the torch CPU tier and a weight fingerprint |
| `probe_table.py` | Groups probed machines into cells and flags tier/cell disagreements |
| `compare_and_decompose.py <TAG>` | Bit-exactness, KS, Spearman, misranking and seed x machine variance split against the reference |
| `distribution_robustness.py <TAG>` | Center, spread and shape agreement, percentile shifts and bootstrap CIs against the reference |
| `paired_equivalence.py` | Seed-paired differences, Wilcoxon and TOST equivalence against the reference |
| `dispatcher_controls.py` | Do the effects of the two CPU dispatchers separate when varied one at a time? Needs an Intel machine with AVX-512; its results are in `dispatcher_controls.csv` |
| `kaggle/` | Runs the probe and a full sweep in public cloud sessions |

## Results of each machine

Results from another machine live in `output_data_<TAG>/`, selected with the `CPC18_OUTPUT_TAG` environment variable. Of
the analysis scripts, only `07_analyze_ablation.py` follows the tag and writes to `analysis_results_<TAG>/`. The others
(`02_analyze.py`, `05_make_paper_figures.py`, `06_analyze_decomp.py`, `08_power_analysis.py`, `09_export_csv.py`,
`10_paper_tables.py`) write to `analysis_results/` whatever the tag, so they should be run without one, on the results of
the reference machine.

The tags used for the article are the identifiers of `machines.csv` in the data deposit, listed in the order of the
machine table of Appendix H: `M01` to `M06`, `C02`, and `M01-forced-AVX2`-style names for the forced-tier runs. The
reference machine, `M00`, uses `output_data/` without a tag. For example, from the root of the repository (`07` needs both
ablation grids):

```bash
CPC18_OUTPUT_TAG=M03 python src/04_train_ablation.py --study b1 --workers 6
CPC18_OUTPUT_TAG=M03 python src/04_train_ablation.py --study b3 --workers 6
CPC18_OUTPUT_TAG=M03 python src/07_analyze_ablation.py
```

The cloud sweep (`kaggle/run_amd_sweep.sh`) merges the results of its sessions into
`scripts/variance_check/amd_sweep/canonical/`; a symbolic link `output_data_C02` pointing there makes them readable by
the analysis scripts.

## Running from the data deposit

These scripts also work with the data deposit alone: for the machines of the article, the results come from `trials.csv`
when `output_data_<TAG>/` is not there, and `probe_table.py` reads `probes.csv` when there are no probe files, which are
not part of the repository. `compare_and_decompose.py`, `distribution_robustness.py` and `paired_equivalence.py` take the
reference from the flat CSVs, so `src/09_export_csv.py` runs first.

## PyTorch packages

On Linux, `pip` installs the standard torch package, which also carries CUDA; the reference machine used it, run on the
CPU. The other machines of the study, and `cpu_probe.py`, which stops on a CUDA build, use the CPU-only package:

```sh
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

The reference machine reproduces bit for bit the trials of the two other `AVX512` machines, which use the CPU-only
package.
