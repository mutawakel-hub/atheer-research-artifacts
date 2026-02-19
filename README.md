# Atheer Simulation Evaluation Artifact (Section VI)

This repository is a **reproducibility artifact** for the simulation-based evaluation in **Section VI** of the paper.

It reproduces (from the same simulation run):

- **Fig. 6** — Transaction Success Rate vs Load (Mean ± 95% CI)
- **Fig. 7** — P95 End-to-End Latency vs Load (Mean ± 95% CI)
- **Table IV** — Aggregated Performance Summary (Mean ± 95% CI)
- **Table V** — Failure Breakdown at 50 TPS (%)

> Scope note: This artifact currently evaluates **S1 vs S2 (Network-only)**:
> - S1: Public Internet
> - S2: Private APN (Atheer)
>
> The parameters used are defined in `atheer_sim.py` (see `SCENARIOS`, `LOAD_POINTS_TPS`, etc.).

---

## Repository layout

- `atheer_sim.py` — main Discrete-Event Simulation (SimPy) + plotting + table export
- `requirements.txt` — Python dependencies
- `tools/build_paper_tables.py` — optional helper to build paper tables from a raw CSV
- `configs/paper.yml` — **documentation mirror** of the paper scenario setup (the current code does **not** read YAML)

---

## Requirements

- Python 3.10+ recommended
- Packages in `requirements.txt`

---

## Quick start

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
python atheer_sim.py
````

If you run on a headless server (no GUI), you can force a non-interactive Matplotlib backend:

```bash
# Linux/macOS:
export MPLBACKEND=Agg

# Windows PowerShell:
# setx MPLBACKEND Agg
```

---

## Outputs

Running `python atheer_sim.py` creates an `outputs/` folder and writes files with a timestamp suffix:

* Raw per-transaction CSV:

  * `outputs/atheer_simulation_results_YYYYMMDD_HHMMSS.csv`

* Figures:

  * `outputs/figure_success_rate_ci_YYYYMMDD_HHMMSS.png`
  * `outputs/figure_p95_latency_ci_YYYYMMDD_HHMMSS.png`

* Summary tables:

  * `outputs/agg_long_YYYYMMDD_HHMMSS.csv`
  * `outputs/agg_wide_YYYYMMDD_HHMMSS.csv`
  * `outputs/table_wide_YYYYMMDD_HHMMSS.tex`

* Failure breakdown (at max load, e.g., 50 TPS):

  * `outputs/failure_breakdown_YYYYMMDD_HHMMSS.csv`

---

## Optional: build paper tables from an existing raw CSV

If you already have a raw CSV and want paper-ready summaries:

```bash
python tools/build_paper_tables.py \
  --raw outputs/atheer_simulation_results_YYYYMMDD_HHMMSS.csv \
  --out paper_artifacts
```

This generates (in `paper_artifacts/`):

* `table_summary_long_*.csv`
* `table_failure_breakdown_*.csv`

---

## Reproducibility

See: `docs/REPRODUCE.md`

---

## Citation

GitHub reads `CITATION.cff` to show **“Cite this repository”**.

---

## License

MIT (see `LICENSE`)
