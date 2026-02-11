"""Optional helper: build paper-ready tables from raw per-transaction CSV.

Usage:
  python tools/build_paper_tables.py --raw outputs/atheer_simulation_results.csv --out paper_artifacts/
"""

import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

def ci95(mean, std, n):
    if n <= 1 or pd.isna(std) or std == 0:
        return (mean, mean)
    half = 1.96 * (std / np.sqrt(n))
    return (mean - half, mean + half)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Raw per-transaction CSV")
    ap.add_argument("--out", default="paper_artifacts", help="Output directory")
    ap.add_argument("--max-load", type=float, default=None, help="Max load for failure breakdown (default: max in CSV)")
    args = ap.parse_args()

    df = pd.read_csv(args.raw)

    # Normalize expected columns
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "status": rename[c] = "status"
        if lc == "reason": rename[c] = "reason"
        if lc == "scenario": rename[c] = "Scenario"
        if lc in ("load_tps","load"): rename[c] = "Load_TPS"
        if lc == "run": rename[c] = "Run"
        if lc in ("duration_s","duration"): rename[c] = "duration_s"
    df = df.rename(columns=rename)

    required = {"Scenario","Load_TPS","Run","status","reason","duration_s"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing expected columns: {missing}")

    def per_run_metrics(g):
        ok = (g["status"] == "SUCCESS")
        sr = float(ok.mean() * 100.0)
        succ = g.loc[ok, "duration_s"]
        p95 = float(succ.quantile(0.95)) if len(succ) else np.nan
        p99 = float(succ.quantile(0.99)) if len(succ) else np.nan
        return pd.Series({"SuccessRate": sr, "P95": p95, "P99": p99})

    per_run = df.groupby(["Scenario","Load_TPS","Run"]).apply(per_run_metrics).reset_index()
    agg = per_run.groupby(["Scenario","Load_TPS"]).agg(
        SR_mean=("SuccessRate","mean"),
        SR_std=("SuccessRate","std"),
        P95_mean=("P95","mean"),
        P95_std=("P95","std"),
        P99_mean=("P99","mean"),
        P99_std=("P99","std"),
        N=("Run","nunique"),
    ).reset_index()

    for m in ["SR","P95","P99"]:
        agg[f"{m}_CI_L"] = agg.apply(lambda r: ci95(r[f"{m}_mean"], r[f"{m}_std"], int(r["N"]))[0], axis=1)
        agg[f"{m}_CI_U"] = agg.apply(lambda r: ci95(r[f"{m}_mean"], r[f"{m}_std"], int(r["N"]))[1], axis=1)

    agg["SuccessRate (Mean±CI)"] = agg.apply(lambda r: f'{r["SR_mean"]:.2f} ± {(r["SR_CI_U"]-r["SR_mean"]):.2f}', axis=1)
    agg["P95 (Mean±CI)"] = agg.apply(lambda r: f'{r["P95_mean"]:.3f} ± {(r["P95_CI_U"]-r["P95_mean"]):.3f}', axis=1)
    agg["P99 (Mean±CI)"] = agg.apply(lambda r: f'{r["P99_mean"]:.3f} ± {(r["P99_CI_U"]-r["P99_mean"]):.3f}', axis=1)

    table = agg[["Scenario","Load_TPS","SuccessRate (Mean±CI)","P95 (Mean±CI)","P99 (Mean±CI)"]].sort_values(["Scenario","Load_TPS"])

    max_load = float(args.max_load) if args.max_load is not None else float(df["Load_TPS"].max())
    high = df[df["Load_TPS"] == max_load].copy()
    breakdown = high.groupby(["Scenario","reason"]).size().unstack(fill_value=0)
    breakdown_pct = breakdown.div(breakdown.sum(axis=1), axis=0) * 100.0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    table.to_csv(out_dir / f"table_summary_long_{ts}.csv", index=False)
    breakdown_pct.round(2).to_csv(out_dir / f"table_failure_breakdown_{ts}.csv")

    print("Saved tables to:", out_dir)

if __name__ == "__main__":
    main()
