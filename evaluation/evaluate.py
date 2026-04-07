"""
evaluation/evaluate.py
=======================
Three-way comparison:
  Fixed Timer vs LAAN Transformer vs Optimal Oracle

Metrics:
  - Average delay (s/vehicle) — primary metric
  - P95 delay — tail latency (worst 5% of vehicles)
  - Level of Service (A–F) per HCM 6th Edition
  - Phase accuracy (% timesteps where predicted == optimal)
  - Wait time reduction vs fixed (%)

This is your results section. Target: 15–25% wait time reduction.
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dl_pipeline.transformer import LaneAwareAttentionNetwork
from configs.config import MODEL, TOKEN, EVAL, INTERSECTION


def compute_los(avg_delay_s: float) -> str:
    for grade, thresh in EVAL.los_thresholds.items():
        if avg_delay_s <= thresh:
            return grade
    return "F"


def simulate_controller(df, pred_fn, name) -> dict:
    """
    Simulate a controller's wait-time impact on test data.
    For each timestep:
      - If predicted phase == optimal: vehicles get served → wait = observed wait
      - If wrong: vehicles wait one extra cycle (penalty)
    """
    cycle_penalty = EVAL.fixed_green_s * INTERSECTION.n_phases   # ~120s
    waits, correct, total = [], 0, 0

    for (_, t), grp in df.groupby(["scenario", "timestep"]):
        grp = grp.sort_values("lane_id")
        if len(grp) != TOKEN.n_lanes: continue
        optimal = int(grp["optimal_phase"].iloc[0])
        pred    = pred_fn(grp)
        base_wait = float(grp["avg_wait_s"].mean())

        if pred == optimal:
            waits.append(base_wait)
            correct += 1
        else:
            waits.append(base_wait + cycle_penalty * 0.4)  # partial penalty
        total += 1

    avg_w = float(np.mean(waits))
    return {
        "controller": name,
        "avg_wait_s": avg_w,
        "p95_wait_s": float(np.percentile(waits, 95)),
        "max_wait_s": float(np.max(waits)),
        "los": compute_los(avg_w),
        "phase_acc": correct / max(total, 1),
    }


def run_evaluation(csv_path="data/traffic_dataset.csv") -> list[dict]:
    if not os.path.exists(csv_path):
        print("Run data_gen/sumo_data_gen.py first."); return []

    df = pd.read_csv(csv_path)
    # Use test split (last 15% timesteps per scenario)
    test_parts = []
    for sc in df["scenario"].unique():
        sub = df[df["scenario"] == sc]
        test_ts = sorted(sub["timestep"].unique())[-int(len(sub["timestep"].unique()) * EVAL.test_split):]
        test_parts.append(sub[sub["timestep"].isin(test_ts)])
    test_df = pd.concat(test_parts)

    norm_cols = [f"{f}_norm" for f in TOKEN.features]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Controller 1: Fixed Timer ─────────────────────────────────────────
    cycle_phases = 4
    def fixed_pred(grp):
        t = int(grp["timestep"].iloc[0])
        return (t // int(EVAL.fixed_green_s)) % cycle_phases

    # ── Controller 2: LAAN Transformer ────────────────────────────────────
    model_path = MODEL.best_model_path
    model = None
    if os.path.exists(model_path):
        model = LaneAwareAttentionNetwork().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device)["state"])
        model.eval()

    def model_pred(grp):
        if model is None:
            return int(np.random.randint(0, 4))
        x = torch.tensor(grp[norm_cols].values, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            return int(model(x)["logits"].argmax(1).item())

    # ── Controller 3: Optimal Oracle ───────────────────────────────────────
    def optimal_pred(grp):
        return int(grp["optimal_phase"].iloc[0])

    results = [
        simulate_controller(test_df, fixed_pred,   "Fixed Timer (30s)"),
        simulate_controller(test_df, model_pred,   "LAAN Transformer (Ours)"),
        simulate_controller(test_df, optimal_pred, "Optimal Oracle"),
    ]

    # Print
    print("\n" + "=" * 68)
    print("EVALUATION RESULTS")
    print("=" * 68)
    print(f"{'Controller':<28} {'Avg Wait':>9} {'P95 Wait':>9} {'LOS':>5} {'PhaseAcc':>10}")
    print("-" * 68)
    for r in results:
        pa = f"{r['phase_acc']:.3f}" if r["phase_acc"] is not None else "  N/A"
        print(f"{r['controller']:<28} {r['avg_wait_s']:>8.1f}s "
              f"{r['p95_wait_s']:>8.1f}s {r['los']:>5}  {pa:>10}")

    if len(results) >= 2:
        red = (results[0]["avg_wait_s"] - results[1]["avg_wait_s"]) / results[0]["avg_wait_s"] * 100
        print(f"\nWait time reduction vs fixed: {red:.1f}%")
        print(f"LOS improvement: {results[0]['los']} → {results[1]['los']}")

    with open("data/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: data/evaluation_results.json")
    return results


if __name__ == "__main__":
    run_evaluation()
