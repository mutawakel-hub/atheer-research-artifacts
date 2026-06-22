# Extending the Simulation Model

This guide shows how to add new scenarios, parameters, and analyses to the Atheer simulation.

---

## Use Cases

- Add a new network scenario (e.g., S3: Hybrid Wi-Fi + Mobile Data)
- Test sensitivity to specific parameters (e.g., higher bank capacity)
- Model new attack scenarios (e.g., partial DDoS)
- Compare with additional baselines (e.g., SMS-based fallback)

---

## 1. Adding a New Scenario

### Step 1: Define Parameters

Open `scripts/simulation/atheer_simulation_v2.py` and add a new `ScenarioParams` instance after the existing S1 and S2:

```python
S3_PARAMS = ScenarioParams(
    name="S3: Hybrid Wi-Fi + Mobile Data",
    mean_latency_ms=80.0,           # Wi-Fi when available, mobile data otherwise
    sigma_latency=0.25,
    base_packet_loss=0.008,         # 0.8% - Wi-Fi is more reliable
    max_packet_loss=0.025,          # capped at 2.5%
    max_retries=1,
    e2e_timeout_s=8.0,              # tighter timeout due to better network
    load_degradation_enabled=True,
    alpha_L=0.10,                   # very mild degradation
    lambda_th=150.0,                # higher threshold
    switch_overhead_ms=20.0,
    bank_capacity_tps=50.0,
)
```

### Step 2: Add to Main Loop

In the `main()` function, add a new loop for S3:

```python
all_results = {"S1": {}, "S2": {}, "S3": {}}  # Add S3
raw_results = {"S1": {}, "S2": {}, "S3": {}}  # Add S3

for tps in LOAD_LEVELS:
    # ... existing S1 and S2 code ...
    
    print(f"\n=== Running S3 (Hybrid) @ {tps} TPS ===")
    s3_res = run_scenario(S3_PARAMS, tps)
    s3_agg = aggregate(s3_res)
    all_results["S3"][tps] = s3_agg
    raw_results["S3"][tps] = s3_res
    print(f"  Success: {s3_agg['success_rate_mean']} +/- {s3_agg['success_rate_ci']}")
    print(f"  P95: {s3_agg['p95_latency_ms_mean']} +/- {s3_agg['p95_latency_ms_ci']} ms")
```

### Step 3: Update the Output Table

Modify the table printing section to include S3:

```python
print(f"{'TPS':>6} | {'S1 Succ%':>18} | {'S2 Succ%':>18} | {'S3 Succ%':>18} | ...")
for tps in LOAD_LEVELS:
    s1 = all_results["S1"][tps]
    s2 = all_results["S2"][tps]
    s3 = all_results["S3"][tps]
    # ... print all three ...
```

### Step 4: Update Figures

In `scripts/figures/generate_figures.py`, add S3 to the plotting functions:

```python
def plot_success_rate(results):
    # ... existing code ...
    s3_means = [results["S3"][str(t)]["success_rate_mean"] for t in tps_levels]
    s3_cis = [results["S3"][str(t)]["success_rate_ci"] for t in tps_levels]
    
    ax.errorbar(tps_levels, s3_means, yerr=s3_cis, marker='^',
                color='#5A8F5A', label='S3: Hybrid Wi-Fi + Mobile Data',
                capsize=3, linewidth=1.5, markersize=6)
    
    # ... rest of plotting code ...
```

---

## 2. Adjusting Existing Parameters

### Sensitivity Analysis Example

To test how S2 performance varies with different latency assumptions:

```python
# Create a parameter sweep
latencies = [60, 100, 130, 200, 300]  # ms
results_sweep = {}

for lat in latencies:
    params = ScenarioParams(
        name=f"S2 (latency={lat}ms)",
        mean_latency_ms=lat,
        # ... other params same as S2_PARAMS ...
    )
    
    results_sweep[lat] = run_scenario(params, 500)  # test at 500 TPS only

# Save sweep results
with open("results/latency_sweep.json", "w") as f:
    json.dump({str(k): aggregate(v) for k, v in results_sweep.items()}, f, indent=2)
```

---

## 3. Adding New Metrics

### Example: Track P99 Latency

In the `Simulation` class:

```python
class Simulation:
    def __init__(self):
        self.success_latencies = []
        self.failures = []
        self.current_offered_load = 0.0
    
    @property
    def p99_latency_ms(self):
        if not self.success_latencies:
            return float('inf')
        sorted_l = sorted(self.success_latencies)
        idx = int(0.99 * len(sorted_l))
        idx = min(idx, len(sorted_l) - 1)
        return sorted_l[idx]
```

In the `run_replication` function:

```python
def run_replication(params: ScenarioParams, tps: int, seed: int) -> dict:
    # ... existing code ...
    return {
        # ... existing fields ...
        "p99_latency_ms": sim.p99_latency_ms,
    }
```

In the `aggregate` function:

```python
def aggregate(results):
    # ... existing code ...
    p99_lats = [r["p99_latency_ms"] for r in results if r["p99_latency_ms"] != float('inf')]
    mean_p99 = statistics.mean(p99_lats) if p99_lats else float('nan')
    
    return {
        # ... existing fields ...
        "p99_latency_ms_mean": round(mean_p99, 3),
    }
```

---

## 4. Modeling New Attack Scenarios

### Example: Partial DDoS Attack

To model a scenario where the gateway experiences 30% capacity reduction due to DDoS:

```python
DDOS_PARAMS = ScenarioParams(
    name="S2 + 30% DDoS",
    mean_latency_ms=130.0,
    sigma_latency=0.35,            # higher variance due to attack
    base_packet_loss=0.05,         # 5% loss due to filtered traffic
    max_packet_loss=0.10,
    max_retries=2,
    e2e_timeout_s=10.0,
    load_degradation_enabled=True,
    alpha_L=0.30,                  # stronger degradation under attack
    lambda_th=50.0,                # lower threshold
    switch_overhead_ms=35.0,       # switch is overloaded
    bank_capacity_tps=35.0,        # reduced capacity
)
```

---

## 5. Validating Against Real Data

If you have access to real network measurements (e.g., from a partner operator):

### Step 1: Export Real Data to JSON

```json
{
  "measurements": [
    {"timestamp": "2024-01-01T10:00:00", "rtt_ms": 145, "loss": 0.012},
    {"timestamp": "2024-01-01T10:01:00", "rtt_ms": 132, "loss": 0.008}
  ]
}
```

### Step 2: Calibrate Simulation Parameters

```python
import json
import statistics

with open("real_measurements.json") as f:
    data = json.load(f)

rtts = [m["rtt_ms"] for m in data["measurements"]]
losses = [m["loss"] for m in data["measurements"]]

# Fit LogNormal parameters
import numpy as np
log_rtts = np.log(rtts)
mu = statistics.mean(log_rtts)
sigma = statistics.stdev(log_rtts)

# Update S2_PARAMS
S2_PARAMS_CALIBRATED = ScenarioParams(
    name="S2: Mobile Data (Calibrated)",
    mean_latency_ms=statistics.mean(rtts),
    sigma_latency=sigma,
    base_packet_loss=statistics.mean(losses),
    # ... other params ...
)
```

### Step 3: Run Validation

Compare simulation output against real measurements to validate the model.

---

## 6. Contributing Your Extensions

If you develop a useful extension, please consider contributing it back:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-scenario`)
3. Test your changes thoroughly
4. Update documentation (`docs/`, `CHANGELOG.md`)
5. Open a Pull Request

See `CONTRIBUTING.md` for detailed guidelines.

---

## 7. Common Pitfalls

### Pitfall 1: Wrong Parameter Order
The order of parameters in `ScenarioParams` matters. Always use keyword arguments:
```python
# ✅ Correct
S3_PARAMS = ScenarioParams(name="S3", mean_latency_ms=80.0, ...)

# ❌ Wrong (positional args are error-prone)
S3_PARAMS = ScenarioParams("S3", 80.0, ...)
```

### Pitfall 2: Forgetting to Update Results Dictionary
When adding S3, you must update both `all_results` and `raw_results` dictionaries to include the new scenario.

### Pitfall 3: Not Updating the Figures Script
If you add a new scenario but don't update `generate_figures.py`, the figures won't show the new data.

### Pitfall 4: Changing Seeds
The deterministic seeds ensure reproducibility. If you change the seed formula, your results will differ from the paper.

---

## Need Help?

Open an issue at: https://github.com/mutawakel-hub/atheer-research-artifacts/issues
