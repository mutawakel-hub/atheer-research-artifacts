# Contributing to Atheer Research Artifacts

Thank you for your interest in contributing! This project welcomes contributions from researchers, engineers, and practitioners working on offline payment systems, mobile security, and financial inclusion.

---

## 🐛 Reporting Issues

If you find a bug, an error in the paper, or a problem with the simulation:

1. **Search existing issues** first to avoid duplicates
2. **Open a new issue** with:
   - Clear title and description
   - Steps to reproduce (for code issues)
   - Expected vs. actual behavior
   - Relevant environment info (Python version, OS, LaTeX engine)
   - Screenshots if applicable

---

## 💡 Suggesting Enhancements

For new features, scenarios, or analyses:

1. Open an issue with the `enhancement` label
2. Describe the use case and motivation
3. Propose the implementation approach
4. Wait for discussion before submitting a PR

---

## 🔀 Pull Requests

### Before you start

- For major changes, please open an issue first to discuss what you would like to change
- For minor fixes (typos, parameter tuning), feel free to submit a PR directly

### PR Process

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/my-new-scenario`
3. **Make your changes** following the code style below
4. **Test your changes**:
   ```bash
   # Run simulation
   python scripts/simulation/atheer_simulation_v2.py
   
   # Regenerate figures
   python scripts/figures/generate_figures.py
   
   # Compile papers
   cd paper_en && tectonic main.tex
   cd ../paper_ar && tectonic main.tex
   ```
5. **Update documentation** if needed (README, CHANGELOG, docs/)
6. **Commit with clear messages**:
   ```
   feat(simulation): add S3 scenario for hybrid routing
   fix(figures): correct axis label in Fig. 6
   docs: add sensitivity analysis for data pricing
   ```
7. **Open a Pull Request** with:
   - Description of changes
   - Reference to related issues (`Closes #123`)
   - Test results / screenshots

---

## 🧪 Code Style

### Python (Simulation Scripts)

- Follow [PEP 8](https://pep8.org/) with line length 100
- Use type hints for function signatures
- Add docstrings for all public functions
- Use meaningful variable names (no single letters except loop counters)
- Keep functions focused — one function, one responsibility

```python
def sample_network_latency_ms(
    params: ScenarioParams,
    current_load_tps: float
) -> float:
    """Sample one-way network latency from a LogNormal distribution,
    with optional load-dependent degradation.
    
    Args:
        params: Scenario parameters
        current_load_tps: Current offered load in TPS
        
    Returns:
        Latency in milliseconds
    """
    ...
```

### LaTeX (Paper Sources)

- Use `\begin{table*}` for wide tables (not `tabularx` with X columns)
- Cite references with `\cite{key}` (no spaces)
- Use `\ref{}` for cross-references with non-breaking space (`Section~\ref{sec:intro}`)
- Wrap long equations in `align` or `multline` (not `equation` with `\quad`)
- All figures need `\caption{}` and `\label{}`

---

## 📊 Adding New Simulation Scenarios

To add a new scenario (e.g., S3: Hybrid Routing):

1. **Define parameters** in `scripts/simulation/atheer_simulation_v2.py`:
   ```python
   S3_PARAMS = ScenarioParams(
       name="S3: Hybrid Routing",
       mean_latency_ms=100.0,
       # ...
   )
   ```

2. **Update the main loop** to run S3 alongside S1 and S2

3. **Add the scenario** to figures in `scripts/figures/generate_figures.py`

4. **Update documentation**:
   - `docs/SIMULATION_PARAMETERS.md` — justify new parameters
   - `CHANGELOG.md` — describe the new scenario
   - Paper LaTeX — add results table and discussion

5. **Test thoroughly**:
   - Run simulation with new scenario
   - Verify figures render correctly
   - Check statistical significance

---

## 📝 Updating the Paper

### English Version (`paper_en/`)

- IEEE Conference format (2-column)
- Compile with: `tectonic main.tex`
- Verify all tables fit within column or page width

### Arabic Version (`paper_ar/`)

- XeLaTeX + polyglossia + bidi (RTL)
- **Critical**: Load all packages BEFORE polyglossia (which loads bidi)
- Bundled Amiri fonts — no system font installation required
- Compile with: `tectonic main.tex` or `xelatex main.tex && xelatex main.tex`

---

## 🧭 Research Ethics

When contributing to this project, please adhere to:

1. **No security testing on real banking systems** without documented permission
2. **No claims about named institutions** without verifiable evidence
3. **Use only synthetic data** in simulations
4. **Cite all sources** properly
5. **Disclose conflicts of interest**

See [`docs/ETHICAL_STATEMENT.md`](./docs/ETHICAL_STATEMENT.md) for full guidelines.

---

## 📋 Code of Conduct

Be respectful, constructive, and inclusive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/) in all interactions.

---

## ❓ Questions?

- **Issues**: [GitHub Issues](https://github.com/mutawakel-hub/atheer-research-artifacts/issues)
- **Email**: nabil.almekhlafi@su.edu.ye, a.almutawakel@su.edu.ye

Thank you for contributing to Atheer! 🙏
