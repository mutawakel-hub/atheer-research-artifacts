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
