"""Shared by the experiment scripts: where results go and how numbers are recorded."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def save_metrics(experiment, values):
    """Merge one experiment's numbers into results/metrics.json."""
    path = OUT / "metrics.json"
    metrics = json.loads(path.read_text()) if path.exists() else {}
    metrics[experiment] = values
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
