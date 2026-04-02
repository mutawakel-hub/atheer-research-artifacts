"""tools/extract_tables.py
Memory-efficient extraction of IEEE paper tables from the large simulation CSV.
Reads in chunks (~100k rows) to avoid loading the entire 1.9 GB file at once.

Usage:
  python tools/extract_tables.py --csv outputs/atheer_simulation_results_20260401_052214.csv
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def ci95_half(mean: float, std: float, n: int) -> float:
    """Returns the half-width of a 95% confidence interval."""
    if n <= 1 or std == 0 or (isinstance(std, float) and math.isnan(std)):
        return 0.0
    return 1.96 * (std / math.sqrt(n))


def main():
    parser = argparse.ArgumentParser(
        description="Extract paper tables (III & IV) from raw simulation CSV"
    )
    parser.add_argument("--csv", required=True, help="Path to the raw simulation CSV")
    parser.add_argument(
        "--chunksize", type=int, default=200_000, help="Rows per chunk (default 200k)"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # We only need these columns — cuts memory significantly
    use_cols = ["Scenario", "Load_TPS", "Run", "status", "duration_s", "reason"]

    # --- Accumulators (per-run) ---
    run_total = defaultdict(int)        # (Scenario, TPS, Run) -> count
    run_success = defaultdict(int)      # (Scenario, TPS, Run) -> success count
    run_durations = defaultdict(list)   # (Scenario, TPS, Run) -> [durations...]

    # --- Failure reasons at max load (500 TPS) ---
    fail_reason_counts = defaultdict(lambda: defaultdict(int))  # scenario -> reason -> count
    fail_totals = defaultdict(int)  # scenario -> total count

    print(f"Reading {csv_path} in chunks of {args.chunksize:,}...")
    chunk_num = 0

    for chunk in pd.read_csv(csv_path, usecols=use_cols, chunksize=args.chunksize):
        chunk_num += 1
        if chunk_num % 20 == 0:
            print(f"  ... processed {chunk_num * args.chunksize:,} rows")

        for (scen, tps, run_idx), group in chunk.groupby(
            ["Scenario", "Load_TPS", "Run"]
        ):
            key = (scen, int(tps), int(run_idx))

            run_total[key] += len(group)

            success_mask = group["status"] == "SUCCESS"
            n_success = int(success_mask.sum())
            run_success[key] += n_success

            if n_success > 0:
                run_durations[key].extend(
                    group.loc[success_mask, "duration_s"].tolist()
                )

            # Failure breakdown at max load only
            if int(tps) == 500:
                fail_totals[scen] += len(group)
                for reason, cnt in group["reason"].value_counts().items():
                    fail_reason_counts[scen][reason] += int(cnt)

    print(f"Done reading. Chunks processed: {chunk_num}")

    # =====================================================================
    # Compute per-run metrics
    # =====================================================================
    rows = []
    for (scen, tps, run_idx), total in run_total.items():
        succ = run_success.get((scen, tps, run_idx), 0)
        sr = (succ / total) * 100.0 if total > 0 else 0.0

        durations = run_durations.get((scen, tps, run_idx), [])
        if durations:
            p95 = float(np.percentile(durations, 95))
        else:
            p95 = float("nan")

        rows.append(
            {
                "Scenario": scen,
                "Load_TPS": tps,
                "Run": run_idx,
                "SuccessRate": sr,
                "P95": p95,
            }
        )

    df_runs = pd.DataFrame(rows)

    # =====================================================================
    # Aggregate across runs (mean ± 95% CI)
    # =====================================================================
    agg = (
        df_runs.groupby(["Scenario", "Load_TPS"])
        .agg(
            SR_mean=("SuccessRate", "mean"),
            SR_std=("SuccessRate", "std"),
            P95_mean=("P95", "mean"),
            P95_std=("P95", "std"),
            N=("Run", "nunique"),
        )
        .reset_index()
        .sort_values(["Scenario", "Load_TPS"])
    )

    agg["SR_ci"] = agg.apply(
        lambda r: ci95_half(r["SR_mean"], r["SR_std"], int(r["N"])), axis=1
    )
    agg["P95_ci"] = agg.apply(
        lambda r: ci95_half(r["P95_mean"], r["P95_std"], int(r["N"])), axis=1
    )

    # Split by scenario
    s1 = agg[agg["Scenario"].str.contains("S1")].sort_values("Load_TPS")
    s2 = agg[agg["Scenario"].str.contains("S2")].sort_values("Load_TPS")
    load_points = sorted(agg["Load_TPS"].unique())

    # =====================================================================
    # PRINT: Human-readable Table III
    # =====================================================================
    print("\n" + "=" * 90)
    print("  TABLE III — E2E Performance Summary  (Mean ± 95% CI,  N=30)")
    print("=" * 90)
    hdr = f"{'TPS':>6} | {'S1 Success(%)':>20} | {'S2 Success(%)':>20} | {'S1 P95 (s)':>18} | {'S2 P95 (s)':>18}"
    print(hdr)
    print("-" * 90)

    for tps in load_points:
        r1 = s1[s1["Load_TPS"] == tps].iloc[0] if len(s1[s1["Load_TPS"] == tps]) else None
        r2 = s2[s2["Load_TPS"] == tps].iloc[0] if len(s2[s2["Load_TPS"] == tps]) else None
        col1 = f"{r1['SR_mean']:.2f} ± {r1['SR_ci']:.2f}" if r1 is not None else "N/A"
        col2 = f"{r2['SR_mean']:.2f} ± {r2['SR_ci']:.2f}" if r2 is not None else "N/A"
        col3 = f"{r1['P95_mean']:.3f} ± {r1['P95_ci']:.3f}" if r1 is not None else "N/A"
        col4 = f"{r2['P95_mean']:.3f} ± {r2['P95_ci']:.3f}" if r2 is not None else "N/A"
        print(f"{tps:>6} | {col1:>20} | {col2:>20} | {col3:>18} | {col4:>18}")

    # =====================================================================
    # PRINT: LaTeX code for Table III
    # =====================================================================
    print("\n" + "=" * 90)
    print("  COPY-PASTE LaTeX ROWS for Table III")
    print("=" * 90)

    for tps in load_points:
        r1 = s1[s1["Load_TPS"] == tps].iloc[0] if len(s1[s1["Load_TPS"] == tps]) else None
        r2 = s2[s2["Load_TPS"] == tps].iloc[0] if len(s2[s2["Load_TPS"] == tps]) else None

        s1_sr  = f"{r1['SR_mean']:.2f} $\\pm$ {r1['SR_ci']:.2f}" if r1 is not None else "N/A"
        s2_sr  = f"{r2['SR_mean']:.2f} $\\pm$ {r2['SR_ci']:.2f}" if r2 is not None else "N/A"
        s1_p95 = f"{r1['P95_mean']:.3f} $\\pm$ {r1['P95_ci']:.3f}" if r1 is not None else "N/A"
        s2_p95 = f"{r2['P95_mean']:.3f} $\\pm$ {r2['P95_ci']:.3f}" if r2 is not None else "N/A"

        print(f"{tps}   & {s1_sr} & {s2_sr} & {s1_p95} & {s2_p95} \\\\")
        print("\\hline")

    # =====================================================================
    # PRINT: Human-readable Table IV (Failure Breakdown at 500 TPS)
    # =====================================================================
    print("\n" + "=" * 90)
    print("  TABLE IV — Failure Breakdown at 500 TPS  (N=30)")
    print("=" * 90)

    for scen in sorted(fail_totals.keys()):
        total = fail_totals[scen]
        reasons = fail_reason_counts[scen]

        success_pct  = (reasons.get("NONE", 0) / total) * 100.0
        uplink_pct   = (reasons.get("FAILED_UPLINK_NETWORK", 0) / total) * 100.0
        downlink_pct = (reasons.get("FAILED_DOWNLINK_NETWORK", 0) / total) * 100.0
        timeout_pct  = (
            reasons.get("FAILED_E2E_TIMEOUT", 0)
            + reasons.get("FAILED_QUEUE_TIMEOUT", 0)
        ) / total * 100.0

        print(f"\n  {scen}:")
        print(f"    Total transactions : {total:,}")
        print(f"    Success            : {success_pct:.2f}%")
        print(f"    Uplink Loss        : {uplink_pct:.2f}%")
        print(f"    Downlink Loss      : {downlink_pct:.2f}%")
        print(f"    E2E Timeout        : {timeout_pct:.2f}%")

    # =====================================================================
    # PRINT: LaTeX code for Table IV
    # =====================================================================
    print("\n" + "=" * 90)
    print("  COPY-PASTE LaTeX ROWS for Table IV")
    print("=" * 90)

    for scen in sorted(fail_totals.keys()):
        total = fail_totals[scen]
        reasons = fail_reason_counts[scen]

        success_pct  = (reasons.get("NONE", 0) / total) * 100.0
        uplink_pct   = (reasons.get("FAILED_UPLINK_NETWORK", 0) / total) * 100.0
        downlink_pct = (reasons.get("FAILED_DOWNLINK_NETWORK", 0) / total) * 100.0
        timeout_pct  = (
            reasons.get("FAILED_E2E_TIMEOUT", 0)
            + reasons.get("FAILED_QUEUE_TIMEOUT", 0)
        ) / total * 100.0

        label = "S1: Public Internet" if "S1" in scen else "S2: Private APN"

        if "S2" in scen:
            print(
                f"{label} & \\textbf{{{success_pct:.2f}\\%}} "
                f"& \\textbf{{{uplink_pct:.2f}\\%}} "
                f"& \\textbf{{{downlink_pct:.2f}\\%}} "
                f"& \\textbf{{{timeout_pct:.2f}\\%}} \\\\"
            )
        else:
            print(
                f"{label} & {success_pct:.2f}\\% "
                f"& {uplink_pct:.2f}\\% "
                f"& {downlink_pct:.2f}\\% "
                f"& {timeout_pct:.2f}\\% \\\\"
            )

    print("\n" + "=" * 90)
    print("Done. Copy the LaTeX rows above into your .tex files.")
    print("=" * 90)


if __name__ == "__main__":
    main()
