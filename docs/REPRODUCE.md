# Reproducing Section VI

## 1) Install dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Add the simulation script
Place your simulation script at repo root as:
- `atheer_sim.py`  (not included in this zip)

## 3) Run the experiment
```bash
python atheer_sim.py
```

## 4) Expected outputs
Your run should create:
- Raw results: `atheer_simulation_results_*.csv`
- Figures: `figure_success_rate_ci_*.png`, `figure_p95_latency_ci_*.png`
- Summary tables: `agg_long_*.csv` or `agg_wide_*.csv`
- Failure breakdown: `failure_breakdown_*.csv`

If you only have raw logs, you can generate paper-ready tables via:
```bash
python tools/build_paper_tables.py --raw <PATH_TO_RAW_CSV> --out paper_artifacts
```
