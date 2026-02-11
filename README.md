# Atheer Simulation Evaluation Artifact (Section VI)

This repository is a **reproducibility artifact** for the simulation-based evaluation (Section VI).
It is designed to reproduce:
- **Fig. 6** — Transaction Success Rate vs Load (Mean ± 95% CI)
- **Fig. 7** — P95 End-to-End Latency vs Load (Mean ± 95% CI)
- **Table IV** — Aggregated Performance Summary (Mean ± 95% CI)
- **Table V** — Failure Breakdown at 50 TPS (%)

> NOTE: You will add the main simulation script **`atheer_sim.py`** (not included in this zip).

## Quick start
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
pip install -r requirements.txt
python atheer_sim.py
```

## Paper config
- `configs/paper.yml` contains the **S1 vs S2 (Network-only)** setup.
- If you change paper parameters, update `configs/paper.yml` and regenerate outputs.

## Outputs
Expected output files (names may include timestamps):
- Raw logs: `atheer_simulation_results_*.csv`
- Figures: `figure_success_rate_ci_*.png`, `figure_p95_latency_ci_*.png`
- Summary tables: `agg_long_*.csv` or `agg_wide_*.csv`
- Failure breakdown: `failure_breakdown_*.csv`

Optional: If you only have raw logs, create paper-ready tables:
```bash
python tools/build_paper_tables.py --raw <RAW_CSV_PATH> --out paper_artifacts
```

## Reproducibility
See `docs/REPRODUCE.md`.

## Citation
GitHub reads `CITATION.cff` to show “Cite this repository”.

## License
MIT (see `LICENSE`).
