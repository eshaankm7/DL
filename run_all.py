"""
run_all.py
==========
Single command to run the ENTIRE project end-to-end.

  python run_all.py

Steps:
  1. Generate dataset (SUMO if available, synthetic fallback otherwise)
  2. Train full model
  3. Train ablated model (ablation study)
  4. Evaluate all three controllers
  5. Print final summary

Then:
  streamlit run dashboard/app.py
"""

import os, sys, time

def banner(msg):
    print(f"\n{'═'*60}\n  {msg}\n{'═'*60}")

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║  ITSC — Lane-Aware Attention Network                     ║
║  Mixed Traffic Signal Control · Indian Urban Roads       ║
║  CV ↔ DL Integrated Pipeline                             ║
╚══════════════════════════════════════════════════════════╝
    """)
    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("simulation/output", exist_ok=True)

    # Step 1
    banner("STEP 1 / 4 — Generating Dataset (SUMO or Synthetic)")
    from data_gen.sumo_data_gen import generate_full_dataset
    generate_full_dataset()

    # Step 2
    banner("STEP 2 / 4 — Training LAAN (Full Model with two_wheeler_ratio)")
    from dl_pipeline.train import train
    t0 = time.time()
    h_full = train(use_two_wheeler=True, run_id="full")
    print(f"  Done in {time.time()-t0:.0f}s | Test acc: {h_full['test_acc']:.4f}")

    # Step 3
    banner("STEP 3 / 4 — Ablation Study (Without two_wheeler_ratio)")
    h_abl = train(use_two_wheeler=False, run_id="ablation")
    delta = (h_full["test_acc"] - h_abl["test_acc"]) * 100
    print(f"  Done | Novel feature contribution: +{delta:.2f}%")

    # Step 4
    banner("STEP 4 / 4 — Evaluating vs Fixed Timer Baseline")
    from evaluation.evaluate import run_evaluation
    results = run_evaluation()

    # Summary
    banner("FINAL SUMMARY")
    if results and len(results) >= 2:
        fixed = results[0]; ours = results[1]
        red = (fixed["avg_wait_s"] - ours["avg_wait_s"]) / fixed["avg_wait_s"] * 100
        print(f"  Fixed timer avg wait:    {fixed['avg_wait_s']:.1f}s  (LOS {fixed['los']})")
        print(f"  Our model avg wait:      {ours['avg_wait_s']:.1f}s   (LOS {ours['los']})")
        print(f"  Wait time reduction:     {red:.1f}%")
        print(f"  Phase accuracy:          {ours['phase_acc']*100:.1f}%")
        print(f"  Novel feature (2W ratio) contribution: +{delta:.2f}% accuracy")

    print("""
╔══════════════════════════════════════════════════════════╗
║  All outputs saved in data/ and checkpoints/             ║
║                                                          ║
║  NEXT: Launch dashboard                                  ║
║    streamlit run dashboard/app.py                        ║
╚══════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    main()
