# Atheer Simulation Evaluation Artifact

This repository is a **reproducibility artifact** for the simulation-based evaluation of the "Atheer" system as described in **Section VI** of the research paper.


This repository enables researchers and developers to reproduce and confirm the published results, ensuring scientific transparency and reliability.
**Now simulates a full 4-layer End-to-End (E2E) system:**

1. **Edge Layer (SDK):** Local NFC tap and HCE cryptogram generation.
2. **Network Layer (Transport):** Uplink/Downlink via S1 (Public Internet) or S2 (Private APN), with packet loss and latency.
3. **Processing Layer (Atheer Switch):** Python-simulated switch with modeled Redis (idempotency) and PostgreSQL (transaction storage) micro-latencies.
4. **Integration Layer (Core Bank):** Bank processing via API adapters.

## 📄 Reproducible Results

This repository reproduces (from the same simulation run) the following figures and tables from the research paper:

*   **Fig. 6** — Transaction Success Rate vs Load (Mean ± 95% CI)
*   **Fig. 7** — P95 End-to-End Latency vs Load (Mean ± 95% CI)
*   **Table IV** — Aggregated Performance Summary (Mean ± 95% CI)
*   **Table V** — Failure Breakdown at Max Load (500 TPS) (%)


> **Scope Note:**
> - This artifact now evaluates the **full E2E system** (Edge, Network, Switch, Bank).
> - S1: Public Internet
> - S2: Private APN (Atheer)
> - The simulation parameters are defined in `atheer_sim.py` and mirrored in `configs/paper.yml` for documentation.

## 📂 Repository Layout

*   `atheer_sim.py` — Main Discrete-Event Simulation (SimPy) file + plotting + table export.
*   `requirements.txt` — List of required Python dependencies.
*   `tools/build_paper_tables.py` — Optional helper tool to build paper tables from a raw CSV file.
*   `configs/paper.yml` — Configuration file for the paper scenario setup. The simulation reads these parameters directly at runtime.
*   `docs/` — Contains additional documentation, including `REPRODUCE.md`, `MODEL_ASSUMPTIONS.md`, and `PARAMETERS.md`.

## ⚙️ Requirements

*   Python 3.10+ (recommended)
*   Dependencies listed in `requirements.txt`:
    *   `simpy>=4.1`
    *   `numpy>=1.24`
    *   `pandas>=2.0`
    *   `matplotlib>=3.7`
    *   `jinja2>=3.1`
    *   `pyyaml>=6.0`

## 🚀 Quick Start

To reproduce the results, follow these steps:

1.  **Install Dependencies:**

    ```shell
    python -m venv .venv

    # For Windows:
    .venv\Scripts\activate

    # For Linux/macOS:
    source .venv/bin/activate

    pip install -r requirements.txt
    ```

2.  **Run the Simulation:**

    Run the simulation from the repository root:

    ```shell
    python atheer_sim.py
    ```

    If you are running on a headless server (no GUI), you can force a non-interactive Matplotlib backend:

    ```shell
    # For Linux/macOS:
    export MPLBACKEND=Agg

    # For Windows PowerShell:
    # setx MPLBACKEND Agg
    ```


## 📊 Expected Outputs & KPIs

Running `python atheer_sim.py` will create an `outputs/` folder and write timestamped files within it:

* **Raw per-transaction CSV file:**
    * `outputs/atheer_simulation_results_YYYYMMDD_HHMMSS.csv`
* **Figures (Plots):**
    * `outputs/figure_success_rate_ci_YYYYMMDD_HHMMSS.{png,pdf,svg}`
    * `outputs/figure_p95_latency_ci_YYYYMMDD_HHMMSS.{png,pdf,svg}`
* **Summary Tables:**
    * `outputs/agg_long_YYYYMMDD_HHMMSS.csv`
    * `outputs/agg_wide_YYYYMMDD_HHMMSS.csv`
    * `outputs/table_wide_YYYYMMDD_HHMMSS.tex`
* **Failure Breakdown (at max load, e.g., 500 TPS):**
    * `outputs/failure_breakdown_YYYYMMDD_HHMMSS.csv`

**Key Performance Indicators (KPIs):**
- **Switch Overhead:** Total time spent inside the Atheer Switch (Redis + PostgreSQL + Logic). Must prove it's < 20ms.
- **End-to-End Latency:** Total time from NFC tap to Merchant Screen response.
- **Success Rate under Load:** Including transactions dropped due to network timeouts vs. successfully processed E2E.


## 📝 Optional: Build Paper Tables from an Existing Raw CSV

If you already have a raw simulation CSV and wish to generate paper-ready tables (including Switch Overhead), use the following command:

```shell
python tools/build_paper_tables.py \
    --raw outputs/atheer_simulation_results_YYYYMMDD_HHMMSS.csv \
    --out paper_artifacts
```

This will write the table artifacts into the `paper_artifacts/` folder, including the "Switch Overhead (Mean±CI)" column for IEEE tables.


## 🐛 Troubleshooting

* **Matplotlib issue on headless servers:** If you encounter issues when running the simulation on a headless server, ensure that the Matplotlib backend is set as described in the "Run the Simulation" section above.
* **Switch/Layer parameters:** If you change Switch micro-latency parameters, update both `atheer_sim.py` and `configs/paper.yml` for documentation consistency.

## 🤝 Contributing

Contributions to improve this repository are welcome. Please read `CONTRIBUTING.md` (if available) for guidelines on how to contribute.

## 📚 Citation

If you use this software/repository in your research work, please cite it using the following DOI:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19383901.svg)](https://doi.org/10.5281/zenodo.19383901)

### IEEE Format (Recommended for DTISD/IEEE)
> A. A. M. H. Al-Mutawakel, *Atheer Simulation Evaluation Artifact* (Version v1.1.1). Zenodo, Apr. 2026. doi: 10.5281/zenodo.19383901.

### APA Format (7th Edition)
> Al-Mutawakel, A. A. M. H. (2026). *Atheer Simulation Evaluation Artifact* (Version v1.1.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.19383901

### MLA Format (9th Edition)
> Al-Mutawakel, Ahmed Ali Mohammed Hasan. *Atheer Simulation Evaluation Artifact*. Version v1.1.1, Zenodo, 2 Apr. 2026. *Zenodo*, https://doi.org/10.5281/zenodo.19383901.

### Chicago Format (17th Edition)
> Al-Mutawakel, Ahmed Ali Mohammed Hasan. *Atheer Simulation Evaluation Artifact* (version v1.1.1). Zenodo, 2026. https://doi.org/10.5281/zenodo.19383901.

### Harvard Format
> Al-Mutawakel, A.A.M.H. (2026) *Atheer Simulation Evaluation Artifact*. Version v1.1.1. Available at: https://doi.org/10.5281/zenodo.19383901.

### BibTeX Format
```bibtex
@software{al_mutawakel_2026_19383901,
  author       = {Al-Mutawakel, Ahmed Ali Mohammed Hasan},
  title        = {Atheer Simulation Evaluation Artifact},
  month        = apr,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.1.1},
  doi          = {10.5281/zenodo.19383901},
  url          = {[https://doi.org/10.5281/zenodo.19383901](https://doi.org/10.5281/zenodo.19383901)}
}
## ⚖️ License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## 📧 Contact

For inquiries or support, please contact the authors via the email address mentioned in the research paper or by opening an Issue in this repository.
