# How to Reproduce the Simulation Results

This guide walks you through reproducing the simulation results reported in the Atheer paper.

---

## Prerequisites

- **Python 3.10 or later**
- **pip** (Python package installer)
- **~50 MB free disk space** (for results)
- **~3 minutes** runtime

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/mutawakel-hub/atheer-research-artifacts.git
cd atheer-research-artifacts
```

Or download and extract the ZIP archive.

---

## Step 2: Set Up Python Environment

### Option A: Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install simpy matplotlib numpy
```

### Option B: System-wide Installation

```bash
pip install simpy matplotlib numpy
```

---

## Step 3: Run the Simulation

```bash
python scripts/simulation/atheer_simulation_v2.py
```

### Expected Output

The script will run for approximately 2-3 minutes and produce output like:

```
=== Running S1 (Public Internet) @ 5 TPS ===
  Success: 99.50 +/- 0.14
  P95: 1272.0 +/- 5.0 ms

=== Running S2 (Mobile Data) @ 5 TPS ===
  Success: 98.32 +/- 0.15
  P95: 459.0 +/- 3.0 ms

... (continues for all 6 load levels)

================================================================================
TABLE IV: E2E PERFORMANCE SUMMARY (mean +/- 95% CI, N=10)
================================================================================
   TPS |           S1 Succ% |           S2 Succ% |         S1 P95 (s) |         S2 P95 (s)
------------------------------------------------------------------------------------------
     5 |     99.50 +/- 0.14 |     98.32 +/- 0.15 |    1.272 +/- 0.005 |    0.459 +/- 0.003
    25 |     99.50 +/- 0.06 |     98.47 +/- 0.05 |    1.268 +/- 0.003 |    0.458 +/- 0.001
    50 |     99.14 +/- 0.04 |     98.48 +/- 0.05 |    2.144 +/- 0.005 |    0.458 +/- 0.001
   100 |     98.38 +/- 0.03 |     98.48 +/- 0.04 |    3.902 +/- 0.011 |    0.458 +/- 0.001
   250 |     96.02 +/- 0.05 |     98.17 +/- 0.03 |    9.164 +/- 0.011 |    0.538 +/- 0.001
   500 |     76.15 +/- 0.05 |     97.61 +/- 0.02 |   14.451 +/- 0.004 |    0.672 +/- 0.000
```

### Output Files

Results are saved to `results/sim_results/`:
- `aggregated.json` — Aggregated metrics (mean ± 95% CI) for each scenario and load level
- `raw.json` — Raw per-replication data (for further analysis)

---

## Step 4: Regenerate Figures (Optional)

If you want to regenerate the figures from the simulation results:

```bash
python scripts/figures/generate_figures.py
```

This will produce 8 PNG figures in `figures/`:
- `fig1_architecture.png` — 4-tier architecture diagram
- `fig2_layered.png` — Edge and Switch layer modules
- `fig3_interaction.png` — System interaction diagram
- `fig4_state_diagram.png` — Armed session state diagram
- `fig5_packet.png` — 180-byte payload breakdown
- `fig6_success_rate.png` — Transaction success rate vs. load
- `fig7_p95_latency.png` — P95 latency vs. load
- `fig8_cost_model.png` — Cost sensitivity analysis

---

## Step 5: Verify Results

Compare your results with those reported in the paper (Table IV in Section VII-G):

| TPS | S1 Success | S2 Success | S1 P95 (s) | S2 P95 (s) |
|-----|------------|------------|------------|------------|
| 5 | 99.50 ± 0.14 | 98.32 ± 0.15 | 1.272 | 0.459 |
| 25 | 99.50 ± 0.06 | 98.47 ± 0.05 | 1.268 | 0.458 |
| 50 | 99.14 ± 0.04 | 98.48 ± 0.05 | 2.144 | 0.458 |
| 100 | 98.38 ± 0.03 | 98.48 ± 0.04 | 3.902 | 0.458 |
| 250 | 96.02 ± 0.05 | 98.17 ± 0.03 | 9.164 | 0.538 |
| 500 | 76.15 ± 0.05 | 97.61 ± 0.02 | 14.451 | 0.672 |

**Note**: Due to platform-specific floating-point differences, your results may vary slightly (typically within ±0.01% for success rates and ±0.001s for latencies).

---

## Step 6: Compile the Paper (Optional)

### English Version (IEEE Conference)

```bash
cd paper_en
tectonic main.tex
# Output: main.pdf
```

Or with traditional LaTeX:
```bash
cd paper_en
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Arabic Version (XeLaTeX)

```bash
cd paper_ar
tectonic main.tex
# Output: main.pdf
```

Or:
```bash
cd paper_ar
xelatex main.tex
xelatex main.tex  # run twice to resolve cross-references
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'simpy'`

**Solution**: Install simpy:
```bash
pip install simpy
```

### Issue: Simulation runs but results differ from paper

**Possible causes**:
1. Different Python version (we used 3.10)
2. Different simpy version (we used 4.1.2)
3. Platform-specific floating-point differences

**Solution**: The differences should be within ±0.01% for success rates. If larger, please open an issue with your environment details.

### Issue: LaTeX compilation fails

**For Arabic version**: Ensure all packages are loaded BEFORE polyglossia (see `paper_ar/main.tex` for correct ordering).

**For English version**: Ensure you have `IEEEtran` class installed (Tectonic handles this automatically).

---

## Need Help?

- **Open an issue**: https://github.com/mutawakel-hub/atheer-research-artifacts/issues
- **Email authors**: nabil.almekhlafi@su.edu.ye, a.almutawakel@su.edu.ye

---

**Estimated time to complete all steps**: 5-10 minutes (including compilation)
