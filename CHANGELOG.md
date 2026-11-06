# Changelog

All notable changes to this project are documented in this file. The format follows the spirit of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/): an `[Unreleased]` section plus
versioned, dated entries in reverse-chronological order (SemVer), with changes listed as
flat bullets prefixed by type (`Bugfix:`, `Feature:`, …) rather than grouped into
Added/Changed/Fixed sections. Each release links to its git tag.

Note: the version of the code is the one in [CITATION.cff](CITATION.cff).

Versioning note: minor releases mark visible feature jumps; patch releases cover bugfixes and
maintenance updates, including docs polish that does not change behavior. Versions prior to 1.0.0
were reconstructed from the git history, with each tag on the last commit of its version.

## [1.0.0](https://github.com/enheragu/cpc18-learning-playground/releases/tag/v1.0.0) - 2026-11-06

First release archived in Zenodo, so that the article *Variance-Aware Evaluation of Behavioral Choice Prediction Models: A Replication and Extension on the Choice Prediction Competition 2018* can cite a fixed version of this code. The results themselves are archived in a separate Zenodo data deposit.

- Feature: cross-machine study. Part of the experiments run again on other machines and CPU tiers, and are compared with the reference machine, with controls on what selects the CPU kernels.
- Feature: data deposit. Builds the public per-trial results of the article from the training logs, with a README that documents every file. Two exports of the same data are byte-identical.
- Feature: analysis from the deposit. Without the training logs, the analysis scripts read the deposit, so every figure and table of the article can be regenerated from the Zenodo files alone.
- Feature: the tables and the numbers quoted in the text of the article that no other script produces are printed from the deposit.
- Feature: audits of the Gaussian assumption and of other parametric families for the lower tail of the results.
- Analysis: the tables read the 10th percentile from a fitted lognormal and add the misranking of the leaderboards, and the power analysis also gives the precision of Monte Carlo.
- Analysis: the MNIST submodule moves to v1.0.1, so every simulation starts from a fixed seed and a tie is not a switched order. The numbers of the article do not change.
- Figures: trials needed to reach a target, which replaces a survival figure that showed another quantity, transfer across machines, choice of the family of the target, and one dot per run of the six models.
- Bugfix: the figures and the variance components only ran on the machine of the author, and the figures script stopped on a clean clone.
- Bugfix: the sampling figures have a fixed seed, so two runs give the same result.
- Bugfix: the diagonal of the factorial pairs each initialization seed with the order seed at the same position of its pool, as the article says, and not the sorted seeds.
- Bugfix: the standard error of the ICC used a wrong formula, and its table also gives the exact interval.
- Bugfix: an incomplete ablation grid stops the analysis instead of giving a plausible ANOVA.
- Bugfix: the re-runs failed on a clean clone when saving their results, and the script that fetches the reference code of BEAST-GB did not work.
- Training: trials that end in the same millisecond no longer overwrite each other, and the trainers and the determinism audit end with an error when something fails.
- Data: provenance and CC BY 4.0 licence of the features of the CPC18 problems.
- Tests: ordering-probability helper and reading the results from the deposit.
- Cleanup: the personal queue and presentation scripts leave the repository.
- Docs: shorter README that opens with what the study shows and how to regenerate the article, with the experiments, the models and the cross-machine study in their own READMEs.
- License: MIT license and citation metadata.

## [0.1.0](https://github.com/enheragu/cpc18-learning-playground/releases/tag/v0.1.0) - 2026-07-21
- Feature: training of the experiments of the article and of an exploratory regime on the raw lottery parameters, all from a shared pool of seeds.
- Feature: analysis. Misranking, variance decomposition, ablation ANOVA, power analysis and determinism audit, with flat CSV exports.
- Feature: figures and tables of the article. The figures and the variance components still depend on the machine of the author.
- Feature: export of the data shown in MLV Tools.
- Data: the BEAST-derived features of the CPC18 problems, with the script that regenerates them.
- Tests: ANOVA contrasts, misranking against its analytic value and same-seed reproducibility.
- Docs: README with the experiments, the scripts and the setup, and pinned versions of the dependencies.


# Useful notes :)

> To commit a new release: update version in `CITATION.cff` and add the new section in the changelog with the tag URL (tag name is known in advance). Commit, tag, push.
> ```sh
> git add <files>
> git commit -m "Release vN.M.P"
> git tag -a vN.M.P -m "vN.M.P"
> git push origin main
> git push --tags
> ```
> Zenodo archives GitHub releases, not tags, so publish a release only for the versions to archive.
