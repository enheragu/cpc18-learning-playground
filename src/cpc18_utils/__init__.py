import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# Runs on other machines: CPC18_OUTPUT_TAG is any label of the machine or
# configuration. When set, training logs go to output_data_<tag>/ instead of
# output_data/, so the results of one machine never mix with those of
# another. Unset, the reference output_data/ is used.
_tag = os.environ.get("CPC18_OUTPUT_TAG", "")
_output_dirname = f"output_data_{_tag}" if _tag else "output_data"
output_path = os.path.normpath(os.path.join(_HERE, "..", "..", _output_dirname))
data_path = os.path.normpath(os.path.join(_HERE, "..", "..", "data"))

# Training logs of the stable protocol (`--protocol stable` of the trainers, see
# protocol.py): output_data_stable/, or output_data_stable_<tag>/ with a tag, so
# that they never mix with the logs of the original protocol. Created on first use.
_stable_dirname = f"output_data_stable_{_tag}" if _tag else "output_data_stable"
stable_output_path = os.path.normpath(os.path.join(_HERE, "..", "..", _stable_dirname))

# Analysis outputs that follow the tag: analysis_results_<tag>/ when a tag is
# set, analysis_results/ otherwise. Only the scripts that import this path
# (07_analyze_ablation.py) follow the tag; the others write to
# analysis_results/ whatever the tag, so they are run without one.
_analysis_dirname = f"analysis_results_{_tag}" if _tag else "analysis_results"
analysis_path = os.path.normpath(os.path.join(_HERE, "..", "..", _analysis_dirname))

# Data deposit of the article (output of 16_export_zenodo.py, the files archived
# in Zenodo), read when the training logs are not available (see training_logs.py).
deposit_path = os.path.normpath(os.path.join(_HERE, "..", "..", "zenodo_export"))

os.makedirs(output_path, exist_ok=True)

from .data import load_cpc18_aggregate, CPC18Dataset  # noqa: E402
