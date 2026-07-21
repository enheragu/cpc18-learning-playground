import os

_HERE = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.normpath(os.path.join(_HERE, "..", "..", "output_data"))
data_path = os.path.normpath(os.path.join(_HERE, "..", "..", "data"))

os.makedirs(output_path, exist_ok=True)

from .data import load_cpc18_aggregate, CPC18Dataset  # noqa: E402
