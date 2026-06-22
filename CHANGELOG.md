# Changelog

All notable changes to this simulation artifact are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2.1.0] — 2026-06-22

### Repository Refocus

This release refocuses the repository exclusively on the simulation artifact, removing all paper-related content and references.

### Removed
- `paper_en/` directory (English LaTeX paper source and PDF)
- `paper_ar/` directory (Arabic LaTeX paper source, fonts, and PDF)
- `reproduce/` directory (legacy reproduction scripts)
- `docs/REVISION_NOTES.md` (reviewer-related content)
- `scripts/build_arabic_paper.py` (paper build script)

### Changed
- `README.md` rewritten to focus on simulation artifact only
- `CHANGELOG.md` cleaned of paper-related references
- `CITATION.bib` now references only the simulation artifact (not the paper)
- `CONTRIBUTING.md` updated to remove paper-related guidance
- `docs/ETHICAL_STATEMENT.md` updated to remove paper references
- `docs/SIMULATION_PARAMETERS.md` updated to remove paper references
- `docs/COST_MODEL.md` updated to remove equation references to paper
- `examples/run_simulation.md` updated to remove paper compilation steps
- `examples/extend_model.md` updated to remove paper references

### Final Repository Structure

```
atheer-research-artifacts/
├── scripts/    Simulation engine + figure generation
├── figures/    Pre-generated plots (PNG, 200 DPI)
├── results/    Pre-computed simulation results (JSON)
├── docs/       Simulation parameters, cost model, ethical statement
└── examples/   Reproduction guide and extension guide
```

The repository is now a clean, self-contained simulation artifact with no dependencies on any external paper or conference submission.

---

## [2.0.0] — 2026-06-22

### Major Revision

This release introduces a more realistic and economically grounded simulation model.

### Added
- **180-byte payload** formal breakdown in `docs/SIMULATION_PARAMETERS.md`
- **Cost model** with sensitivity analysis in `docs/COST_MODEL.md`
- **Ethical statement** in `docs/ETHICAL_STATEMENT.md`
- **Usage examples** in `examples/`
- **Pre-computed results** (`results/sim_results/`) for verification
- **8 publication-quality figures** at 200 DPI

### Changed
- **Replaced Private APN model** with Mobile Data Routing + Partner-Subsidized Billing
- **Realistic parameters for S2**:
  - Latency: 60 ms → 130 ms (realistic for mobile data)
  - Packet loss: 0.1% → 1.5% (realistic SLA upper bound)
  - Retries: 0 → 1
  - Load degradation: disabled → enabled (α_L = 0.15, λ_th = 100)
- **Restructured codebase** into `scripts/`, `figures/`, `results/`, `docs/`, `examples/`
- **Cleaner separation** between simulation engine and figure generation
- **Type hints** throughout the codebase
- **Docstrings** for all public functions

### Fixed
- **Unrealistic latency assumptions** for low-infrastructure networks
- **Deterministic seeds** now properly hash scenario name for uniqueness
- **Warmup period** now properly accounted for in total simulation time

### Results Comparison (v1.0 vs v2.0)

| Metric | v1.0 (Private APN) | v2.0 (Mobile Data) |
|--------|-------------------|-------------------|
| S2 success at 500 TPS | 99.80% | 97.61% |
| S2 P95 latency at 500 TPS | 245 ms | 672 ms |
| S2 latency assumption | 60 ms (unrealistic) | 130 ms (realistic) |
| S2 packet loss | 0.1% | 1.5% |

The v2.0 results are more conservative but more credible.

---

## [1.0.0] — 2024

### Initial Release

- Original simulation with Private APN model
- Hardcoded parameters (no YAML config)
- Claimed 99.80% success at 500 TPS on APN pathway

---

## Versioning Policy

- **Major** (X.0.0): Architecture changes, parameter overhauls
- **Minor** (0.X.0): New scenarios, additional metrics, documentation
- **Patch** (0.0.X): Bug fixes, performance improvements, typo corrections
