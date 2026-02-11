import simpy
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
from datetime import datetime

# =========================
# CONFIGURATION (seconds)
# =========================

@dataclass
class ScenarioConfig:
    key: str
    name: str

    # Network (seconds)
    latency_mean_s: float
    latency_std_s: float
    packet_loss: float  # 0..1
    retries: int

    # Timeouts (seconds)
    queue_timeout_s: float
    e2e_timeout_s: float

    # Bank
    bank_capacity: int
    service_time_s: float = 0.02  # 20ms

    # Offline local processing (e.g., NFC tap)
    local_time_s: float = 0.05    # 50ms

    # Degradation (applies only if enabled)
    enable_degradation: bool = False
    degrade_threshold_tps: float = 10.0
    degrade_latency_alpha: float = 0.60
    degrade_loss_alpha: float = 1.50
    degrade_max_loss: float = 0.30


# Two main scenarios only (Priority removed)
SCENARIOS = [
    ScenarioConfig(
        key="S1_PUBLIC",
        name="S1: Public Internet",
        latency_mean_s=0.400,  # 400ms
        latency_std_s=0.200,   # 200ms jitter
        packet_loss=0.10,      # 10%
        retries=2,
        queue_timeout_s=12.0,
        e2e_timeout_s=15.0,
        bank_capacity=12,
        enable_degradation=True,
        degrade_threshold_tps=10.0,
        degrade_latency_alpha=0.75,
        degrade_loss_alpha=1.25,
        degrade_max_loss=0.35,
    ),
    ScenarioConfig(
        key="S2_PRIVATE",
        name="S2: Private APN (Atheer) [Network-APN]",
        latency_mean_s=0.060,  # 60ms
        latency_std_s=0.015,   # 15ms
        packet_loss=0.001,     # 0.1%
        retries=1,
        queue_timeout_s=4.0,
        e2e_timeout_s=5.0,
        bank_capacity=12,      # SAME as S1 to isolate network effect
        enable_degradation=False,
    ),
]

LOAD_POINTS_TPS = [1, 5, 10, 20, 30, 40, 50]
WARMUP_S = 5.0
MEASURE_S = 60.0
NUM_RUNS = 5

BASE_SEED = 20260211  # keep fixed for reproducibility


# =========================
# Helpers
# =========================

def lognormal_mu_sigma(mean: float, std: float) -> Tuple[float, float]:
    """
    Convert desired mean/std of a LogNormal random variable into (mu, sigma)
    where X ~ LogNormal(mu, sigma) (numpy definition).
    """
    mean = max(mean, 1e-6)
    std = max(std, 1e-9)
    variance = std * std
    sigma2 = math.log(1.0 + variance / (mean * mean))
    sigma = math.sqrt(sigma2)
    mu = math.log(mean) - sigma2 / 2.0
    return mu, sigma


def ci95(mean: float, std: float, n: int) -> Tuple[float, float]:
    if n <= 1 or std == 0 or math.isnan(std) or pd.isna(std):
        return mean, mean
    half = 1.96 * (std / math.sqrt(n))
    return mean - half, mean + half


# =========================
# Simulation Core
# =========================

class PaymentSystem:
    def __init__(self, env: simpy.Environment, cfg: ScenarioConfig, offered_tps: float, rng: np.random.Generator):
        self.env = env
        self.cfg = cfg
        self.offered_tps = offered_tps
        self.rng = rng
        self.bank = simpy.Resource(env, capacity=cfg.bank_capacity)
        self.stats = []

    def effective_network_params(self) -> Tuple[float, float, float]:
        mean_s = self.cfg.latency_mean_s
        std_s = self.cfg.latency_std_s
        loss = self.cfg.packet_loss

        if self.cfg.enable_degradation:
            th = max(1e-9, self.cfg.degrade_threshold_tps)
            overload = max(0.0, (self.offered_tps - th) / th)
            mean_s = mean_s * (1.0 + self.cfg.degrade_latency_alpha * overload)
            std_s = std_s * (1.0 + self.cfg.degrade_latency_alpha * overload)
            loss = min(self.cfg.degrade_max_loss, loss * (1.0 + self.cfg.degrade_loss_alpha * overload))

        return mean_s, std_s, loss

    def sample_network_latency(self) -> float:
        mean_s, std_s, _ = self.effective_network_params()
        mu, sigma = lognormal_mu_sigma(mean_s, std_s)
        val = float(self.rng.lognormal(mean=mu, sigma=sigma))
        return max(0.001, val)

    def maybe_lost(self) -> bool:
        _, _, loss = self.effective_network_params()
        return self.rng.random() < loss

    def transmit_with_retries(self, direction: str, start_time: float):
        """
        Yields until hop completes. Returns tuple:
        (success, attempts_used, time_spent, fail_reason)
        """
        attempts = 0
        time_spent = 0.0

        while attempts <= self.cfg.retries:
            attempts += 1
            latency = self.sample_network_latency()

            if self.maybe_lost():
                rto = max(0.01, 2.0 * latency)
                yield self.env.timeout(rto)
                time_spent += rto

                if (self.env.now - start_time) > self.cfg.e2e_timeout_s:
                    return False, attempts, time_spent, "FAILED_E2E_TIMEOUT"
                continue

            yield self.env.timeout(latency)
            time_spent += latency
            return True, attempts, time_spent, None

        return False, attempts, time_spent, f"FAILED_{direction.upper()}_NETWORK"

    def process_transaction(self, tx_id: int):
        start_time = self.env.now
        in_measure_window = (start_time >= WARMUP_S) and (start_time <= WARMUP_S + MEASURE_S)

        # 1) Local
        yield self.env.timeout(self.cfg.local_time_s)

        # Early E2E
        if (self.env.now - start_time) > self.cfg.e2e_timeout_s:
            if in_measure_window:
                self.record(tx_id, start_time, "FAILED", "FAILED_E2E_TIMEOUT", 0, 0, 0.0, 0.0)
            return

        # 2) Uplink
        uplink = self.env.process(self.transmit_with_retries("UPLINK", start_time))
        uplink_ok, uplink_attempts, uplink_time, uplink_reason = yield uplink

        if not uplink_ok:
            if in_measure_window:
                self.record(tx_id, start_time, "FAILED", uplink_reason or "FAILED_UPLINK_NETWORK",
                            uplink_attempts, 0, 0.0, uplink_time)
            return

        # 3) Bank queue + service
        if (self.env.now - start_time) > self.cfg.e2e_timeout_s:
            if in_measure_window:
                self.record(tx_id, start_time, "FAILED", "FAILED_E2E_TIMEOUT", uplink_attempts, 0, 0.0, uplink_time)
            return

        req = self.bank.request()
        q_start = self.env.now
        results = yield req | self.env.timeout(self.cfg.queue_timeout_s)

        if req not in results:
            try:
                req.cancel()
            except Exception:
                pass

            if in_measure_window:
                self.record(tx_id, start_time, "FAILED", "FAILED_QUEUE_TIMEOUT",
                            uplink_attempts, 0, self.env.now - q_start, uplink_time)
            return

        queue_wait = self.env.now - q_start
        yield self.env.timeout(self.cfg.service_time_s)
        self.bank.release(req)

        # 4) Downlink
        downlink = self.env.process(self.transmit_with_retries("DOWNLINK", start_time))
        downlink_ok, downlink_attempts, downlink_time, downlink_reason = yield downlink

        net_time = uplink_time + downlink_time

        if not downlink_ok:
            if in_measure_window:
                self.record(tx_id, start_time, "FAILED", downlink_reason or "FAILED_DOWNLINK_NETWORK",
                            uplink_attempts, downlink_attempts, queue_wait, net_time)
            return

        total_duration = self.env.now - start_time
        if total_duration > self.cfg.e2e_timeout_s:
            if in_measure_window:
                self.record(tx_id, start_time, "FAILED", "FAILED_E2E_TIMEOUT",
                            uplink_attempts, downlink_attempts, queue_wait, net_time)
            return

        if in_measure_window:
            self.record(tx_id, start_time, "SUCCESS", "NONE",
                        uplink_attempts, downlink_attempts, queue_wait, net_time)

    def record(self, tx_id: int, start_time: float, status: str, reason: str,
               uplink_attempts: int, downlink_attempts: int, queue_wait_s: float, net_time_s: float):
        end_time = self.env.now
        self.stats.append({
            "tx_id": tx_id,
            "start_time_s": start_time,
            "end_time_s": end_time,
            "duration_s": end_time - start_time,
            "status": status,
            "reason": reason,
            "uplink_attempts": uplink_attempts,
            "downlink_attempts": downlink_attempts,
            "queue_wait_s": queue_wait_s,
            "net_time_s": net_time_s,
            "bank_service_s": self.cfg.service_time_s,
            "local_time_s": self.cfg.local_time_s,
        })


def transaction_generator(env: simpy.Environment, system: PaymentSystem, tps: float,
                          end_arrivals_s: float, rng: np.random.Generator):
    tx_id = 0
    while env.now < end_arrivals_s:
        ia = float(rng.exponential(scale=1.0 / max(tps, 1e-9)))
        yield env.timeout(ia)
        tx_id += 1
        env.process(system.process_transaction(tx_id))


# =========================
# Runner + Analysis
# =========================

def run_simulation() -> pd.DataFrame:
    all_rows = []

    end_arrivals_s = WARMUP_S + MEASURE_S
    extra_tail = max(cfg.e2e_timeout_s for cfg in SCENARIOS) + 1.0
    run_until_s = end_arrivals_s + extra_tail

    print(f"{'Scenario':<38} | {'Load':>4} | {'Run':>3} | {'Success%':>8} | {'P95(s)':>7} | {'P99(s)':>7}")
    print("-" * 80)

    for cfg in SCENARIOS:
        for tps in LOAD_POINTS_TPS:
            for run_idx in range(NUM_RUNS):
                seed = BASE_SEED + (hash(cfg.key) % 100000) * 1000 + int(tps) * 10 + run_idx
                rng = np.random.default_rng(seed)

                env = simpy.Environment()
                system = PaymentSystem(env, cfg, offered_tps=tps, rng=rng)

                env.process(transaction_generator(env, system, tps, end_arrivals_s, rng))
                env.run(until=run_until_s)

                df_run = pd.DataFrame(system.stats)
                if df_run.empty:
                    success_rate = 0.0
                    p95 = float("nan")
                    p99 = float("nan")
                else:
                    success_mask = df_run["status"] == "SUCCESS"
                    success_rate = float(success_mask.mean() * 100.0)
                    succ = df_run.loc[success_mask, "duration_s"]
                    p95 = float(succ.quantile(0.95)) if len(succ) else float("nan")
                    p99 = float(succ.quantile(0.99)) if len(succ) else float("nan")

                    for _, r in df_run.iterrows():
                        all_rows.append({
                            "ScenarioKey": cfg.key,
                            "Scenario": cfg.name,
                            "Load_TPS": tps,
                            "Run": run_idx,
                            **r.to_dict()
                        })

                print(f"{cfg.name:<38} | {tps:>4} | {run_idx:>3} | {success_rate:>7.2f} | {p95:>7.3f} | {p99:>7.3f}")

    return pd.DataFrame(all_rows)


def summarize_and_plot(df: pd.DataFrame, out_dir: Path, ts: str):
    # Per run metrics
    def run_metrics(g: pd.DataFrame):
        success = (g["status"] == "SUCCESS")
        sr = success.mean() * 100.0
        succ = g.loc[success, "duration_s"]
        p95 = succ.quantile(0.95) if len(succ) else np.nan
        p99 = succ.quantile(0.99) if len(succ) else np.nan
        return pd.Series({"SuccessRate": sr, "P95": p95, "P99": p99})

    per_run = df.groupby(["Scenario", "Load_TPS", "Run"]).apply(run_metrics).reset_index()

    agg = per_run.groupby(["Scenario", "Load_TPS"]).agg(
        SR_mean=("SuccessRate", "mean"),
        SR_std=("SuccessRate", "std"),
        P95_mean=("P95", "mean"),
        P95_std=("P95", "std"),
        P99_mean=("P99", "mean"),
        P99_std=("P99", "std"),
        N=("Run", "nunique"),
    ).reset_index()

    # CI bounds
    agg["SR_CI_L"] = agg.apply(lambda r: ci95(r["SR_mean"], r["SR_std"], int(r["N"]))[0], axis=1)
    agg["SR_CI_U"] = agg.apply(lambda r: ci95(r["SR_mean"], r["SR_std"], int(r["N"]))[1], axis=1)
    agg["P95_CI_L"] = agg.apply(lambda r: ci95(r["P95_mean"], r["P95_std"], int(r["N"]))[0], axis=1)
    agg["P95_CI_U"] = agg.apply(lambda r: ci95(r["P95_mean"], r["P95_std"], int(r["N"]))[1], axis=1)

    # ----- Figure: Success Rate vs Load -----
    plt.figure(figsize=(10, 6))
    for scen in agg["Scenario"].unique():
        s = agg[agg["Scenario"] == scen].sort_values("Load_TPS")
        x = s["Load_TPS"].values
        y = s["SR_mean"].values
        yerr = np.vstack([y - s["SR_CI_L"].values, s["SR_CI_U"].values - y])
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=scen)

    plt.title("Transaction Success Rate vs Load (Mean ± 95% CI)")
    plt.xlabel("Load (TPS)")
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig1 = out_dir / f"figure_success_rate_ci_{ts}.png"
    plt.savefig(fig1, dpi=200)
    plt.show()

    # ----- Figure: P95 Latency vs Load -----
    plt.figure(figsize=(10, 6))
    for scen in agg["Scenario"].unique():
        s = agg[agg["Scenario"] == scen].sort_values("Load_TPS")
        x = s["Load_TPS"].values
        y = s["P95_mean"].values
        yerr = np.vstack([y - s["P95_CI_L"].values, s["P95_CI_U"].values - y])
        plt.errorbar(x, y, yerr=yerr, marker="s", capsize=3, label=scen)

    plt.title("P95 End-to-End Latency vs Load (Mean ± 95% CI)")
    plt.xlabel("Load (TPS)")
    plt.ylabel("Latency (s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig2 = out_dir / f"figure_p95_latency_ci_{ts}.png"
    plt.savefig(fig2, dpi=200)
    plt.show()

    # ----- Failure Breakdown at Max Load -----
    max_load = max(LOAD_POINTS_TPS)
    high = df[df["Load_TPS"] == max_load].copy()
    breakdown = high.groupby(["Scenario", "reason"]).size().unstack(fill_value=0)
    breakdown_pct = breakdown.div(breakdown.sum(axis=1), axis=0) * 100.0

    print(f"\nTable: Failure Breakdown (at {max_load} TPS) [%]")
    print(breakdown_pct.round(2).fillna(0))

    breakdown_pct.to_csv(out_dir / f"failure_breakdown_{ts}.csv")

    return agg, breakdown_pct, fig1, fig2


def build_summary_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - long summary (Scenario, Load_TPS, metrics + CI)
      - wide table (Load_TPS x {Scenario -> formatted strings})
    """
    def per_run_metrics(g):
        success = (g["status"] == "SUCCESS")
        sr = success.mean() * 100.0
        succ = g.loc[success, "duration_s"]
        p95 = succ.quantile(0.95) if len(succ) else np.nan
        p99 = succ.quantile(0.99) if len(succ) else np.nan
        return pd.Series({"SuccessRate": sr, "P95": p95, "P99": p99})

    per_run = df.groupby(["Scenario", "Load_TPS", "Run"]).apply(per_run_metrics).reset_index()

    agg = per_run.groupby(["Scenario", "Load_TPS"]).agg(
        SR_mean=("SuccessRate", "mean"),
        SR_std=("SuccessRate", "std"),
        P95_mean=("P95", "mean"),
        P95_std=("P95", "std"),
        P99_mean=("P99", "mean"),
        P99_std=("P99", "std"),
        N=("Run", "nunique"),
    ).reset_index()

    for col in ["SR", "P95", "P99"]:
        agg[f"{col}_CI_L"] = agg.apply(lambda r: ci95(r[f"{col}_mean"], r[f"{col}_std"], int(r["N"]))[0], axis=1)
        agg[f"{col}_CI_U"] = agg.apply(lambda r: ci95(r[f"{col}_mean"], r[f"{col}_std"], int(r["N"]))[1], axis=1)

    # Pretty strings for paper (Mean ± halfCI)
    agg["SuccessRate (Mean±CI)"] = agg.apply(lambda r: f'{r["SR_mean"]:.2f} ± {(r["SR_CI_U"]-r["SR_mean"]):.2f}', axis=1)
    agg["P95 (Mean±CI)"] = agg.apply(lambda r: f'{r["P95_mean"]:.3f} ± {(r["P95_CI_U"]-r["P95_mean"]):.3f}', axis=1)
    agg["P99 (Mean±CI)"] = agg.apply(lambda r: f'{r["P99_mean"]:.3f} ± {(r["P99_CI_U"]-r["P99_mean"]):.3f}', axis=1)

    wide = agg.pivot(
        index="Load_TPS",
        columns="Scenario",
        values=["SuccessRate (Mean±CI)", "P95 (Mean±CI)", "P99 (Mean±CI)"]
    ).sort_index()

    return agg, wide


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    print("Starting Atheer Simulation Evaluation (Improved, No-Priority)...")
    results = run_simulation()

    if results.empty:
        print("No results generated.")
        raise SystemExit(0)

    # Output folder + timestamp to avoid PermissionError / file locks
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw per-transaction data
    raw_path = out_dir / f"atheer_simulation_results_{ts}.csv"
    results.to_csv(raw_path, index=False)
    print(f"\nSaved raw results: {raw_path}")

    # Plots + breakdown
    print("Generating plots & breakdown...")
    _, breakdown_pct, fig1, fig2 = summarize_and_plot(results, out_dir, ts)
    print(f"Saved figures:\n - {fig1}\n - {fig2}")

    # Paper summary tables
    agg_long, wide = build_summary_tables(results)
    agg_long_path = out_dir / f"agg_long_{ts}.csv"
    wide_path = out_dir / f"agg_wide_{ts}.csv"
    tex_path = out_dir / f"table_wide_{ts}.tex"

    agg_long.to_csv(agg_long_path, index=False)
    wide.to_csv(wide_path)

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(wide.to_latex())

    print("\n=== Paper Summary (Wide Table) ===")
    print(wide)

    print(f"\nSaved summary tables:\n - {agg_long_path}\n - {wide_path}\n - {tex_path}")
    print("\nDone.")
