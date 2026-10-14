# Reproduction Protocol

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q
python src/experiment.py
python src/agent_selection.py
```

The base benchmark (`src/experiment.py`) uses scikit-learn's public breast-cancer
dataset only as a methodological proxy; it is not a clinical validation.

The action-selection benchmark (`src/agent_selection.py`) uses the public
scikit-learn `digits`, `wine`, and `iris` datasets as proxies for a finite
autonomous-agent action set. The default seed is `20260915`; metrics are written
to `data/agent_selection_metrics.json`.

Both runs are deterministic for a fixed seed, ensemble size, `alpha`, and `beta`.
Exact trailing floating-point digits can vary with the installed NumPy /
scikit-learn version even though the values are mathematically identical; pin the
versions in `requirements.txt` if bit-identical output is required.
Before submitting as a high-stakes AI paper, add public time-series data, repeated
seeds, stronger baselines, and dataset/license provenance.
