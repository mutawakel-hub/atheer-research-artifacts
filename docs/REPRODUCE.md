# Reproducing Section VI

This guide reproduces the Section VI simulation artifact directly from this repository.

## 1) Install dependencies

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

## 2) Run the simulation

Run from the repository root:

```bash
python atheer_sim.py
```

## 3) Expected outputs

The run creates timestamped files under `outputs/`:

- Raw results CSV:
  - `outputs/atheer_simulation_results_YYYYMMDD_HHMMSS.csv`
- Figures:
  - `outputs/figure_success_rate_ci_YYYYMMDD_HHMMSS.png`
  - `outputs/figure_p95_latency_ci_YYYYMMDD_HHMMSS.png`
- Summary tables:
  - `outputs/agg_long_YYYYMMDD_HHMMSS.csv`
  - `outputs/agg_wide_YYYYMMDD_HHMMSS.csv`
  - `outputs/table_wide_YYYYMMDD_HHMMSS.tex`
- Failure breakdown:
  - `outputs/failure_breakdown_YYYYMMDD_HHMMSS.csv`

## 4) Optional: build paper tables from an existing raw CSV

If you already have a raw simulation CSV, generate paper-ready tables with:

```bash
python tools/build_paper_tables.py --raw outputs/atheer_simulation_results_YYYYMMDD_HHMMSS.csv --out paper_artifacts
```

This writes table artifacts into `paper_artifacts/`.

## Troubleshooting

If you run on a headless server (no GUI), force a non-interactive Matplotlib backend:

```bash
# Linux/macOS:
export MPLBACKEND=Agg

# Windows PowerShell:
# setx MPLBACKEND Agg
```
