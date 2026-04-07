"""
dl_pipeline/ablation.py
=======================
Ablation Study — Proves the Novel Contribution

Trains Full model vs Ablated model (two_wheeler_ratio = 0).
Prints comparison table. Saves results for dashboard.

Expected outcome:
  - Peak-hour accuracy drops MORE in ablated model
    (because two_wheeler_ratio varies most during peak hour)
  - This proves the feature adds real information

Run: python dl_pipeline/ablation.py
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dl_pipeline.train import train, eval_epoch
from dl_pipeline.transformer import LaneAwareAttentionNetwork
from data_gen.dataset import TrafficDataset, get_dataloaders
from configs.config import MODEL, TOKEN, EVAL


def run():
    print("ABLATION STUDY: Effect of Two-Wheeler Ratio Feature")
    print("=" * 60)

    h_full = train(use_two_wheeler=True,  run_id="full")
    h_abl  = train(use_two_wheeler=False, run_id="ablation")

    full_acc = h_full["test_acc"]
    abl_acc  = h_abl["test_acc"]
    delta    = (full_acc - abl_acc) * 100

    print(f"\n{'Model':<35} {'Test Acc':>10}")
    print("-" * 47)
    print(f"{'Full (with two_wheeler_ratio)':<35} {full_acc:>10.4f}")
    print(f"{'Ablated (without)':<35} {abl_acc:>10.4f}")
    print(f"{'Delta (contribution)':<35} {'+' if delta>=0 else ''}{delta:>9.2f}%")

    # Per-scenario breakdown
    print("\nPer-scenario breakdown:")
    _per_scenario()

    results = {
        "full_test_acc": full_acc,
        "ablated_test_acc": abl_acc,
        "improvement_pct": delta,
        "full_phase_accs": h_full.get("phase_accs", {}),
        "abl_phase_accs": h_abl.get("phase_accs", {}),
    }
    with open("data/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: data/ablation_results.json")
    return results


def _per_scenario():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full_path = MODEL.best_model_path
    abl_path  = MODEL.ablation_model_path
    if not (os.path.exists(full_path) and os.path.exists(abl_path)):
        print("  Train both models first."); return

    m_full = LaneAwareAttentionNetwork().to(device)
    m_abl  = LaneAwareAttentionNetwork().to(device)
    m_full.load_state_dict(torch.load(full_path, map_location=device)["state"])
    m_abl.load_state_dict(torch.load(abl_path,  map_location=device)["state"])
    m_full.eval(); m_abl.eval()

    df = pd.read_csv("data/traffic_dataset.csv")
    norm_cols = [f"{f}_norm" for f in TOKEN.features]

    print(f"  {'Scenario':<12} {'Full':>8} {'Ablated':>10} {'Delta':>8}")
    print("  " + "-" * 42)

    for sc in ["low", "medium", "peak"]:
        sub = df[df["scenario"] == sc]
        cf = ca = total = 0
        for (_, t), grp in sub.groupby(["scenario", "timestep"]):
            grp = grp.sort_values("lane_id")
            if len(grp) != TOKEN.n_lanes: continue
            x = torch.tensor(grp[norm_cols].values, dtype=torch.float32).unsqueeze(0).to(device)
            y = int(grp["optimal_phase"].iloc[0])
            with torch.no_grad():
                cf += int(m_full(x)["logits"].argmax(1).item() == y)
                x_abl = x.clone(); x_abl[:,:,4] = 0.0
                ca += int(m_abl(x_abl)["logits"].argmax(1).item() == y)
            total += 1
        af, aa = cf/max(total,1), ca/max(total,1)
        d = (af - aa) * 100
        print(f"  {sc:<12} {af:>8.4f} {aa:>10.4f} {'+' if d>=0 else ''}{d:>7.2f}%")


if __name__ == "__main__":
    run()
