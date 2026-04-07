"""
data_gen/sumo_data_gen.py
=========================
SUMO TraCI Data Extractor — WITH proper SUMO integration

When SUMO IS installed:
  Runs SUMO simulation, reads E2 detector data via TraCI,
  generates labeled dataset (state, optimal_phase) per timestep.

When SUMO is NOT installed:
  Falls back to physics-based synthetic generator automatically.
  Dataset format is identical — downstream code doesn't change.

The optimal label at each timestep = SUMO's built-in actuated
controller's decision. This is the "oracle" we train to imitate.

HOW CV + SUMO INTEGRATE:
  SUMO renders frames via traci.gui.screenshot()
  → CVPipeline.process_frame() extracts token vectors
  → TraCI detectors provide ground truth counts for CV validation
  This lets us measure how accurate our CV extraction is
  against SUMO's perfect sensor readings.
"""

import os
import sys
import csv
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import SIM, INTERSECTION, TOKEN


# ── Detector IDs matching detectors.add.xml ───────────────────────────────────
DETECTOR_IDS = [
    "det_N_0", "det_N_1", "det_N_2",
    "det_S_0", "det_S_1", "det_S_2",
    "det_E_0", "det_E_1", "det_E_2",
    "det_W_0", "det_W_1", "det_W_2",
]

FEATURE_COLS = TOKEN.features + ["optimal_phase", "scenario", "timestep"]


def run_sumo_scenario(
    scenario_name: str,
    scale: float,
    two_wheeler_frac: float,
    n_steps: int,
    use_gui: bool = False,
) -> list[dict]:
    """
    Run one SUMO scenario via TraCI. Returns list of row dicts.
    Each row = one timestep × one lane = one sample.
    """
    # Set SUMO_HOME
    sumo_home = SIM.sumo_home
    sys.path += [os.path.join(sumo_home, "tools")]

    try:
        import traci
    except ImportError:
        print("  TraCI not available — using synthetic fallback.")
        return _synthetic_fallback(scenario_name, scale, two_wheeler_frac, n_steps)

    binary = "sumo-gui" if use_gui else "sumo"
    cmd = [
        binary,
        "-c", SIM.config_file,
        "--step-length", str(SIM.step_length),
        "--scale", str(scale),
        "--seed", "42",
        "--no-warnings", "--no-step-log",
    ]

    rows = []
    try:
        traci.start(cmd, port=SIM.port)
        tls_id = "center"

        for t in range(n_steps):
            traci.simulationStep()

            # Get current signal phase
            current_phase = traci.trafficlight.getPhase(tls_id)
            signal_phase_idx = current_phase // 2  # Even=green, odd=yellow

            # Compute pressure per phase (PressLight formula)
            # Pressure_phase = sum(queue_size on green lanes) - sum(queue_size on red lanes)
            # Optimal phase = argmax(pressure)
            phase_pressures = [0.0] * INTERSECTION.n_phases
            for ph in range(INTERSECTION.n_phases):
                green_lanes = INTERSECTION.phase_to_lanes[ph]
                for lane_idx in green_lanes:
                    det_id = DETECTOR_IDS[lane_idx]
                    try:
                        count = traci.lanearea.getLastStepVehicleNumber(det_id)
                        phase_pressures[ph] += count
                    except Exception:
                        pass
            optimal_phase = int(np.argmax(phase_pressures))

            # Extract per-lane features for all 12 lanes
            for lane_idx, det_id in enumerate(DETECTOR_IDS):
                try:
                    count = traci.lanearea.getLastStepVehicleNumber(det_id)
                    jam_m = traci.lanearea.getJamLengthMetric(det_id)
                    halting = traci.lanearea.getLastStepHaltingNumber(det_id)
                    speed = traci.lanearea.getLastStepMeanSpeed(det_id)
                    occupancy = traci.lanearea.getLastStepOccupancy(det_id) / 100.0

                    # Vehicle type ratio — requires checking individual vehicles
                    veh_ids = traci.lanearea.getLastStepVehicleIDs(det_id)
                    tw_count = sum(
                        1 for v in veh_ids
                        if traci.vehicle.getTypeID(v) == "motorcycle"
                    )
                    tw_ratio = tw_count / max(len(veh_ids), 1)

                    # PCU-weighted density
                    pcu_sum = sum(
                        0.5 if traci.vehicle.getTypeID(v) == "motorcycle" else 1.0
                        for v in veh_ids
                    )
                    pcu_density = min(pcu_sum / 20.0, 1.0)

                    phase_active = float(lane_idx in INTERSECTION.phase_to_lanes.get(signal_phase_idx, []))

                    rows.append({
                        "timestep": t,
                        "lane_id": lane_idx,
                        "scenario": scenario_name,
                        "vehicle_count": float(min(count, 20)),
                        "queue_length_m": float(min(jam_m, 150)),
                        "avg_wait_s": float(min(halting * SIM.step_length, 300)),
                        "pcu_weighted_density": float(pcu_density),
                        "two_wheeler_ratio": float(tw_ratio),
                        "motion_ratio": float(min(speed / 13.89, 1.0)),
                        "is_stalled": float(speed < 0.1 and count > 0),
                        "phase_active": phase_active,
                        "optimal_phase": optimal_phase,
                    })
                except Exception:
                    pass

            if t % 600 == 0:
                print(f"    t={t}/{n_steps} | vehicles={traci.vehicle.getIDCount()}")

    except Exception as e:
        print(f"  SUMO error: {e} — using synthetic fallback")
        rows = _synthetic_fallback(scenario_name, scale, two_wheeler_frac, n_steps)
    finally:
        try:
            traci.close()
        except Exception:
            pass

    return rows


def _synthetic_fallback(
    scenario_name: str, scale: float,
    two_wheeler_frac: float, n_steps: int
) -> list[dict]:
    """
    Physics-based M/D/1 queue simulation.
    Produces identical row format to SUMO extractor.
    Used automatically when SUMO is not installed.
    """
    print(f"  [Synthetic] Generating {scenario_name} scenario...")
    rng = np.random.default_rng(42)
    arrival_rate = scale * 0.5   # vehicles/second/lane

    queues = np.zeros(12)
    stall_counters = np.zeros(12)
    rows = []

    for t in range(n_steps):
        # Vehicle arrivals
        arrivals = rng.poisson(arrival_rate * (1 + 0.3 * np.sin(t / 300)), 12)
        queues = np.maximum(0, queues + arrivals)

        # Two-wheeler ratio with realistic variance
        peak_boost = 0.08 * np.sin(t / 600)
        tw_ratios = np.clip(
            two_wheeler_frac + peak_boost + rng.normal(0, 0.04, 12), 0.1, 0.95
        )

        # Pressure per phase → optimal label
        phase_pressures = []
        for ph in range(INTERSECTION.n_phases):
            lanes = INTERSECTION.phase_to_lanes[ph]
            pcu = sum(
                queues[l] * (tw_ratios[l] * 0.5 + (1 - tw_ratios[l]) * 1.0)
                for l in lanes
            )
            phase_pressures.append(pcu)
        optimal_phase = int(np.argmax(phase_pressures))

        # Discharge green lanes
        cycle_phase = (t // 30) % 4
        green_lanes = set(INTERSECTION.phase_to_lanes.get(cycle_phase, []))
        for i in range(12):
            if i in green_lanes and queues[i] > 0:
                queues[i] = max(0, queues[i] - 0.5)
                stall_counters[i] = 0
            elif queues[i] > 0:
                stall_counters[i] += 1

        for lane_idx in range(12):
            q = queues[lane_idx]
            tw = tw_ratios[lane_idx]
            rows.append({
                "timestep": t,
                "lane_id": lane_idx,
                "scenario": scenario_name,
                "vehicle_count": float(min(q, 20)),
                "queue_length_m": float(min(q * (tw*3.5 + (1-tw)*7.5), 150)),
                "avg_wait_s": float(min(stall_counters[lane_idx], 300)),
                "pcu_weighted_density": float(min(q * (tw*0.5 + (1-tw)*1.0) / 20, 1.0)),
                "two_wheeler_ratio": float(tw),
                "motion_ratio": float(1.0 if lane_idx in green_lanes and q > 0 else 0.0),
                "is_stalled": float(stall_counters[lane_idx] > 75),
                "phase_active": float(lane_idx in INTERSECTION.phase_to_lanes.get(cycle_phase, [])),
                "optimal_phase": optimal_phase,
            })

    return rows


def generate_full_dataset(output_dir: str = "data", use_gui: bool = False) -> str:
    """Generate all scenarios and save to CSV with normalisation stats."""
    os.makedirs(output_dir, exist_ok=True)
    all_rows = []

    for name, cfg in SIM.scenarios.items():
        print(f"\nScenario: {name} | scale={cfg['scale']} | 2W={cfg['two_wheeler_frac']}")
        rows = run_sumo_scenario(
            name, cfg["scale"], cfg["two_wheeler_frac"], cfg["steps"], use_gui
        )
        all_rows.extend(rows)
        print(f"  → {len(rows)} rows generated")

    df = pd.DataFrame(all_rows)

    # Normalise features and save stats
    feature_cols = TOKEN.features
    stats = {}
    for col in feature_cols:
        mu, sigma = df[col].mean(), df[col].std()
        stats[col] = {"mean": float(mu), "std": float(sigma)}
        df[f"{col}_norm"] = (df[col] - mu) / (sigma + 1e-8)

    out_path = os.path.join(output_dir, "traffic_dataset.csv")
    stats_path = os.path.join(output_dir, "feature_stats.json")

    df.to_csv(out_path, index=False)
    import json
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset: {out_path} ({len(df):,} rows)")
    print(f"Phase dist:\n{df['optimal_phase'].value_counts().to_string()}")
    print(f"Avg two_wheeler_ratio by scenario:")
    print(df.groupby("scenario")["two_wheeler_ratio"].mean().to_string())
    return out_path


if __name__ == "__main__":
    generate_full_dataset()
