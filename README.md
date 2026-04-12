
# Atheer System: Simulation Evaluation Artifact

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19383901.svg)](https://doi.org/10.5281/zenodo.19383901)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is the official **reproducibility artifact** for the simulation-based evaluation (Section VI) of the "Atheer" system, as presented in our research paper. It enables researchers and developers to reproduce the published results, ensuring scientific transparency and reliability.

## 📄 About the Paper

**Title:** A Flexible Offline Mobile Payment Architecture Using a Private APN and NFC Technology: A Case Study from Yemen  
**Authors:** Nabil Almekhlafi, Ahmed Almotwakel (Ahmed Ali Mohammed Hasan Al-Mutawakel), Belal Al-Fuhaidi  
**Venue:** DTISD 2026 IEEE Conference

**Abstract:** Mobile payment systems in Yemen only service a small portion of the population due to poor infrastructure; only 17.7% of the population has access to the internet, and of the accesses, the service is slow, disrupted, and inconsistent. Major mobile wallets operating in the country require a constant connection to the cloud. When the connection is lost, transactions are unable to process. We built “Atheer” to remove this dependency. Atheer is a proprietary Android SDK and backend gateway, with a 4-layer backend architecture. The client-side SDK preloads several batches of cryptographic tokens (Limited Use Keys), and uses NFC, in combination with Host Card Emulation, to complete the transaction without an internet connection. To complete the transaction, the merchant device sends a signed payload through a Private APN (a dedicated cellular tunnel, with no public internet connection) to the Atheer Gateway Switch to validate and update the ledger.

## 🎯 Artifact Scope: 4-Layer End-to-End (E2E) Model

This discrete-event simulation (DES) models the full end-to-end architecture proposed in the paper:
1. **Edge Layer (SDK):** Local NFC tap and HCE cryptogram generation.
2. **Network Layer (Transport):** Uplink/Downlink via S1 (Public Internet) or S2 (Private APN), incorporating packet loss and latency.
3. **Processing Layer (Atheer Switch):** Python-simulated switch with modeled micro-latencies for Redis (idempotency checks) and PostgreSQL (transaction storage).
4. **Integration Layer (Core Bank):** Bank processing via API adapters.

## 📊 Reproducible Results

Running this simulation exactly reproduces the following figures and tables from the paper:
* **Fig. 6** — Transaction Success Rate vs Load (Mean ± 95% CI)
* **Fig. 7** — P95 End-to-End Latency vs Load (Mean ± 95% CI)
* **Table III** — E2E Performance Summary (Mean ± 95% CI)
* **Table IV** — Failure Breakdown at Max Load (500 TPS) (%)

> **Note:** The parameters defined in `configs/paper.yml` are the exact values used to produce the published results. Table II in the paper presents simplified/rounded values for readability.

## 📂 Repository Layout

* `atheer_sim.py` — Main Discrete-Event Simulation (SimPy) engine, plotting, and data export.
* `requirements.txt` — Python dependencies required for the simulation.
* `configs/paper.yml` — Configuration file defining network, switch, and bank parameters.
* `tools/` — Helper scripts (e.g., `build_paper_tables.py` and `extract_tables.py`).
* `docs/` — Additional documentation (`REPRODUCE.md`, `MODEL_ASSUMPTIONS.md`, `PARAMETERS.md`).
* `reproduce/` — Quick-run shell and batch scripts for Linux and Windows.

## ⚙️ Getting Started

### Requirements
* Python 3.10+ (recommended)
* Dependencies: `simpy`, `numpy`, `pandas`, `matplotlib`, `pyyaml`, `jinja2`

### Quick Start
1. **Clone and Install:**
   ```bash
   git clone [https://github.com/mutawakel-hub/atheer-research-artifacts.git](https://github.com/mutawakel-hub/atheer-research-artifacts.git)
   cd atheer-research-artifacts
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run the Simulation:**
   ```bash
   python atheer_sim.py
   ```
   *(If running on a headless server without a GUI, set `export MPLBACKEND=Agg` before running).*

3. **Check Outputs:**
   The simulation will generate a timestamped folder inside `outputs/` containing:
   - Raw transaction CSV data.
   - High-resolution plots (`.png`, `.pdf`, `.svg`) for Figures 6 and 7.
   - Generated `.tex` and `.csv` summary tables (Tables III and IV).

## 🤝 Contributing & Troubleshooting
If you encounter any issues or wish to modify the micro-latency parameters for your own research, please refer to `docs/MODEL_ASSUMPTIONS.md` and `docs/PARAMETERS.md`. Contributions and PRs are welcome.

## 📚 Citation

If you use this software/artifact in your research, please cite the Zenodo artifact alongside the paper.

**IEEE Format (Recommended):**
> A. A. M. H. Al-Mutawakel, N. Almekhlafi, and B. Al-Fuhaidi, *Atheer Simulation Evaluation Artifact* (Version v1.1.1). Zenodo, Apr. 2026. doi: 10.5281/zenodo.19383901.

**BibTeX:**
```bibtex
@software{al_mutawakel_2026_19383901,
  author       = {Al-Mutawakel, Ahmed Ali Mohammed Hasan and Almekhlafi, Nabil and Al-Fuhaidi, Belal},
  title        = {Atheer Simulation Evaluation Artifact},
  month        = apr,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.1.1},
  doi          = {10.5281/zenodo.19383901},
  url          = {[https://doi.org/10.5281/zenodo.19383901](https://doi.org/10.5281/zenodo.19383901)}
}
```ا
## ⚖️ License
This project is licensed under the MIT License - see the `LICENSE` file for details.
```
