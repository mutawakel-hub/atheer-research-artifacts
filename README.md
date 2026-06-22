# Atheer Simulation Artifact

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![SimPy 4.1+](https://img.shields.io/badge/SimPy-4.1+-green.svg)](https://simpy.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.1-orange.svg)](CHANGELOG.md)

> **Discrete Event Simulation (DES) for evaluating offline mobile payment architectures.**

This repository contains a reproducible simulation framework for evaluating an offline mobile payment architecture that uses:
- Host Card Emulation (HCE) + Limited Use Keys (LUKs) for offline tokenization
- SoftPOS for zero-CapEx merchant acceptance
- Mobile Data Routing with Partner-Subsidized Billing (180-byte optimized payload)
- A cost model proving data expenditure is a fraction of MDR revenue

---

## 📦 Repository Contents

```
atheer-research-artifacts/
├── README.md                       # This file
├── LICENSE                         # MIT License
├── CHANGELOG.md                    # Version history
├── CITATION.bib                    # BibTeX entry
├── CONTRIBUTING.md                 # How to contribute
├── requirements.txt                # Python dependencies
│
├── scripts/                        # Source code
│   ├── simulation/
│   │   └── atheer_simulation_v2.py # Main simulation engine (SimPy DES)
│   └── figures/
│       └── generate_figures.py     # Plot generator (matplotlib)
│
├── figures/                        # Pre-generated figures (PNG, 200 DPI)
│   ├── fig1_architecture.png
│   ├── fig2_layered.png
│   ├── fig3_interaction.png
│   ├── fig4_state_diagram.png
│   ├── fig5_packet.png
│   ├── fig6_success_rate.png
│   ├── fig7_p95_latency.png
│   └── fig8_cost_model.png
│
├── results/                        # Pre-computed results
│   └── sim_results/
│       ├── aggregated.json         # Aggregated metrics (mean ± 95% CI)
│       └── raw.json                # Raw per-replication data
│
├── docs/                           # Documentation
│   ├── SIMULATION_PARAMETERS.md    # Parameter justification
│   ├── COST_MODEL.md               # Cost model derivation and sensitivity
│   └── ETHICAL_STATEMENT.md        # Ethics disclosure
│
└── examples/                       # Usage examples
    ├── run_simulation.md           # How to reproduce results
    └── extend_model.md             # How to add new scenarios
```

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/mutawakel-hub/atheer-research-artifacts.git
cd atheer-research-artifacts

python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 2. Run the simulation

```bash
python scripts/simulation/atheer_simulation_v2.py
```

**Runtime**: ~2-3 minutes (6 load levels × 2 scenarios × 10 replications)

### 3. Generate figures

```bash
python scripts/figures/generate_figures.py
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

### Discrete Event Simulation (DES) with SimPy

The simulation models a four-layer payment pipeline:

```
Transaction Arrival (Poisson)
    ↓
Edge Processing (50 ms fixed — NFC + crypto)
    ↓
Network Uplink (LogNormal latency + Bernoulli loss)
    ↓
Switch Processing (20 ms + micro-random cache lookup)
    ↓
Core Banking (M/M/c queue, c=50 TPS capacity)
    ↓
Network Downlink (LogNormal + Bernoulli loss)
    ↓
E2E Timeout Check
```

### Statistical Analysis

- **N = 10** independent replications per (scenario, load level)
- **Deterministic seeds** for reproducibility
- **95% Confidence Intervals** using normal approximation
- **Warmup period**: 60 seconds (discarded)
- **Measurement window**: 300 seconds

See [`docs/SIMULATION_PARAMETERS.md`](./docs/SIMULATION_PARAMETERS.md) for parameter justification.

---

## ⚙️ Configuration

The simulation is driven by parameters in `scripts/simulation/atheer_simulation_v2.py` (constants `S1_PARAMS` and `S2_PARAMS`). To customize scenarios, edit these dataclasses and re-run.

---

## 📝 Citation

If you use this simulation artifact, please cite:

```bibtex
@misc{almutawakel2024atheer,
  author       = {Al-Mutawakel, Ahmed},
  title        = {Atheer Simulation Evaluation Artifact},
  year         = {2024},
  howpublished = {GitHub Repository},
  url          = {https://github.com/mutawakel-hub/atheer-research-artifacts},
  version      = {2.1},
  note         = {Discrete Event Simulation for offline mobile payment architecture}
}
```

---

## 👤 Author

**Ahmed Al-Mutawakel**
- Affiliation: Faculty of Computer Science, Sana'a University, Yemen
- Email: a.almutawakel@su.edu.ye
- GitHub: [@mutawakel-hub](https://github.com/mutawakel-hub)

---

## 📄 License

This project is licensed under the **MIT License** — see [`LICENSE`](./LICENSE) for details.

---

## 🤝 Contributing

Found a bug? Want to add a scenario? Please see [`CONTRIBUTING.md`](./CONTRIBUTING.md) and open an issue or pull request.

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/mutawakel-hub/atheer-research-artifacts/issues)
- **Email**: a.almutawakel@su.edu.ye

---

## 🙏 Acknowledgments

- The [SimPy](https://simpy.readthedocs.io/) team for the excellent DES framework
- The [Matplotlib](https://matplotlib.org/) team for visualization tools

---

*If this artifact helps your research, please consider giving it a ⭐ on GitHub.*
