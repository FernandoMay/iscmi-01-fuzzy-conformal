# Reproduction Protocol

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q
python src/experiment.py
```

The benchmark uses scikit-learn's public breast-cancer dataset only as a methodological proxy; it is not a clinical validation. The seed is `20260915`, data are split chronologically only where temporal data exist, and all metrics are generated into `data/metrics.json`. Before submitting as a high-stakes AI paper, add public time-series data, repeated seeds, stronger baselines, and dataset/license provenance.
