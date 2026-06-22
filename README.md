# Atheer — Offline Mobile Payment Architecture

> **A Flexible Offline Mobile Payment Architecture Using NFC and Host Card Emulation: A Cost-Optimized Approach for Low-Infrastructure Environments**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LaTeX](https://img.shields.io/badge/LaTeX-XeLaTeX-red.svg)](https://www.latex-project.org/)
[![Simulation](https://img.shields.io/badge/SimPy-4.1+-green.svg)](https://simpy.readthedocs.io/)
[![Paper Status](https://img.shields.io/badge/Paper-v2.0%20Revised-orange.svg)](./paper_en)

**Research artifacts, simulation code, figures, and LaTeX sources for the Atheer offline mobile payment architecture.**

---

## 📖 Overview

**Atheer** is an offline-first mobile payment architecture designed for low-infrastructure environments. It combines:

- **Host Card Emulation (HCE)** + **Limited Use Keys (LUKs)** for offline tokenization
- **SoftPOS** for zero-CapEx merchant acceptance (no POS hardware required)
- **Mobile Data Routing with Partner-Subsidized Billing** (replaces the original Private APN model in v2)
- **Payload optimization** (180 bytes per settlement request)
- **Cost model** proving data expenditure < 0.5% of MDR revenue

The architecture was evaluated via Discrete Event Simulation (DES) across six load levels (5–500 TPS, N=10), showing **97.6% success rate** at peak load on the Mobile Data pathway versus **76.2%** on the public-internet baseline.

---

## 📦 Repository Contents

```
atheer-research-artifacts/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── CHANGELOG.md                 # Version history
├── CONTRIBUTING.md              # How to contribute
├── CITATION.bib                 # BibTeX entry for citing this work
├── .gitignore
│
├── paper_en/                    # English paper (IEEE Conference)
│   ├── main.tex                 # LaTeX source
│   ├── Atheer_Paper_v2_EN.pdf   # Compiled PDF (10 pages)
│   └── fig1-8.png               # All figures
│
├── paper_ar/                    # Arabic paper (XeLaTeX + polyglossia)
│   ├── main.tex                 # LaTeX source
│   ├── Atheer_Paper_v2_AR.pdf   # Compiled PDF (11 pages)
│   ├── Amiri-Regular.ttf        # Bundled Arabic font
│   ├── Amiri-Bold.ttf
│   └── fig1-8.png
│
├── scripts/                     # Reproducible research scripts
│   ├── simulation/
│   │   └── atheer_simulation_v2.py    # SimPy DES engine
│   └── figures/
│       └── generate_figures.py        # Matplotlib figure generator
│
├── figures/                     # Master copies of all figures (PNG, 200 DPI)
│   ├── fig1_architecture.png
│   ├── fig2_layered.png
│   ├── fig3_interaction.png
│   ├── fig4_state_diagram.png
│   ├── fig5_packet.png
│   ├── fig6_success_rate.png
│   ├── fig7_p95_latency.png
│   └── fig8_cost_model.png
│
├── results/                     # Simulation results
│   └── sim_results/
│       ├── aggregated.json      # Aggregated metrics (mean ± 95% CI)
│       └── raw.json             # Raw per-replication data
│
├── docs/                        # Supplementary documentation
│   ├── REVISION_NOTES.md        # v1 → v2 changes (addressing reviewer feedback)
│   ├── SIMULATION_PARAMETERS.md # Detailed parameter justification
│   ├── COST_MODEL.md            # Cost model derivation and sensitivity
│   └── ETHICAL_STATEMENT.md     # Ethics and responsible research disclosure
│
└── examples/                    # Usage examples
    ├── run_simulation.md        # How to reproduce results
    └── extend_model.md          # How to add new scenarios
```

---

## 🚀 Quick Start

### 1. Reproduce the Simulation

```bash
# Clone this repository
git clone https://github.com/mutawakel-hub/atheer-research-artifacts.git
cd atheer-research-artifacts

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install simpy matplotlib numpy

# Run the simulation (takes ~2-3 minutes)
python scripts/simulation/atheer_simulation_v2.py

# Results will be saved to results/sim_results/
```

### 2. Regenerate Figures

```bash
python scripts/figures/generate_figures.py
# Figures will be saved to figures/
```

### 3. Compile the Paper

#### English version (IEEE Conference, Tectonic):
```bash
cd paper_en
tectonic main.tex
# Output: main.pdf
```

#### Arabic version (XeLaTeX + polyglossia):
```bash
cd paper_ar
tectonic main.tex
# Or: xelatex main.tex && xelatex main.tex
# Output: main.pdf
```

---

## 📊 Key Results

### E2E Performance Summary (mean ± 95% CI, N=10)

| TPS | S1 Success | S2 Success | S1 P95 Latency | S2 P95 Latency |
|-----|------------|------------|----------------|----------------|
| 5   | 99.50 ± 0.14% | 98.32 ± 0.15% | 1.272 s | 0.459 s |
| 25  | 99.50 ± 0.06% | 98.47 ± 0.05% | 1.268 s | 0.458 s |
| 50  | 99.14 ± 0.04% | 98.48 ± 0.05% | 2.144 s | 0.458 s |
| 100 | 98.38 ± 0.03% | 98.48 ± 0.04% | 3.902 s | 0.458 s |
| 250 | 96.02 ± 0.05% | 98.17 ± 0.03% | 9.164 s | 0.538 s |
| **500** | **76.15 ± 0.05%** | **97.61 ± 0.02%** | **14.451 s** | **0.672 s** |

- **S1**: Public Internet baseline (merchant pays data, no prioritization)
- **S2**: Mobile Data with Partner-Subsidized Billing (180-byte payload, wallet absorbs cost)

### Cost Model

| Parameter | Value |
|-----------|-------|
| Daily transactions | 100,000 |
| Payload size | 180 bytes |
| Mobile data price | $1.00 / MB |
| Avg. transaction | $5 |
| MDR | 1% |
| **Daily data cost** | **$17.17** |
| **Daily MDR revenue** | **$5,000** |
| **Cost ratio** | **0.34%** (well below 1% threshold) |

---

## 🔬 Methodology

The architecture was developed using **Design Science Research (DSR)** methodology ([Hevner 2004](https://doi.org/10.2307/25148625); [Peffers 2007](https://doi.org/10.2753/MIS0742-1222240302)). Evaluation was conducted via **Discrete Event Simulation (DES)** built in Python 3.10 with [SimPy](https://simpy.readthedocs.io/) 4.1+.

### Why simulation rather than live deployment?

Live testing on real banking infrastructure in an active conflict zone (Yemen) is infeasible due to:
- Regulatory constraints on financial transactions
- Ethical concerns about testing payment systems on real users
- Practical risks of routing banking settlements through damaged network infrastructure

The DES approach allows controlled experimentation across realistic operating conditions (latency, congestion, packet loss, full loss of connectivity) with reproducible deterministic seeds.

See [`docs/SIMULATION_PARAMETERS.md`](./docs/SIMULATION_PARAMETERS.md) for full parameter justification.

---

## 📝 Version History

### v2.0 (June 2026) — Current Release

**Major revision** addressing peer review feedback from DTISD 2026:

- ✅ Replaced **Private APN** model with **Mobile Data Routing + Partner-Subsidized Billing**
- ✅ Added formal **180-byte payload** proof with field-by-field breakdown
- ✅ Added explicit **Cost Model** with sensitivity analysis
- ✅ Corrected wallet names: Jaib, Alkuraimi, MFloos, Jawali
- ✅ Added **Ethical Statement** addressing Reviewer 2 concerns
- ✅ Added **dual-path provisioning** addressing Reviewer 3's contradiction critique
- ✅ Removed unverifiable claims (zero-rated APN, IaaS revenue stream)
- ✅ Rebuilt simulation with realistic Mobile Data parameters (130ms latency, 1.5% loss)

See [`CHANGELOG.md`](./CHANGELOG.md) and [`docs/REVISION_NOTES.md`](./docs/REVISION_NOTES.md) for details.

### v1.0 (2024) — Original Submission

Original submission to DTISD 2026 using Private APN model. **Rejected** with reviewer feedback that drove the v2 revision. The v1 source is preserved in the `v1-archive` branch for reference.

---

## 📜 Ethical Considerations

This research follows strict ethical guidelines:

1. **No testing on real banking systems** — all evaluation is via simulation
2. **No security claims about named institutions** — wallet names appear only in market description
3. **Fully synthetic data** — no real banking or user data used
4. **Public pricing references only** — ITU, GSMA, Cable.co.uk for cost assumptions
5. **Open source** — full simulation code available for independent verification

See [`docs/ETHICAL_STATEMENT.md`](./docs/ETHICAL_STATEMENT.md) for the complete ethics disclosure.

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@misc{atheer2026,
  author       = {Al-Mekhlafi, Nabil and Al-Mutawakel, Ahmed},
  title        = {Atheer: A Flexible Offline Mobile Payment Architecture Using NFC and Host Card Emulation},
  year         = {2026},
  howpublished = {GitHub Repository},
  url          = {https://github.com/mutawakel-hub/atheer-research-artifacts},
  version      = {2.0}
}
```

---

## 👥 Authors

- **Nabil Al-Mekhlafi** — Faculty of Computer Science, Sana'a University, Yemen
  - Email: nabil.almekhlafi@su.edu.ye
- **Ahmed Al-Mutawakel** — Faculty of Computer Science, Sana'a University, Yemen
  - Email: a.almutawakel@su.edu.ye

---

## 📄 License

This project is licensed under the **MIT License** — see [`LICENSE`](./LICENSE) for details.

- **Code** (simulation scripts, figure generators): MIT License
- **Paper** (LaTeX sources and PDF): MIT License (allows reproduction with attribution)
- **Figures**: MIT License (allows reuse with attribution)

---

## 🤝 Contributing

Contributions are welcome! Please read [`CONTRIBUTING.md`](./CONTRIBUTING.md) for guidelines on:
- Reporting issues
- Suggesting improvements
- Submitting pull requests
- Extending the simulation model

---

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/mutawakel-hub/atheer-research-artifacts/issues)
- **Email**: nabil.almekhlafi@su.edu.ye, a.almutawakel@su.edu.ye
- **Affiliation**: Faculty of Computer Science, Sana'a University, Yemen

---

## 🙏 Acknowledgments

We thank the reviewers of DTISD 2026 for their constructive feedback, which significantly improved the quality of this work. We also acknowledge the open-source community behind SimPy, Matplotlib, and LaTeX/TeX Live.

---

**If this work helps your research, please consider giving it a ⭐ on GitHub.**
