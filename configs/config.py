"""
configs/config.py
=================
Single source of truth for every constant in the project.
Change values here. Never hardcode numbers elsewhere.
"""

from dataclasses import dataclass, field
import os


# ══════════════════════════════════════════════════════════════════════════════
# INTERSECTION GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IntersectionConfig:
    n_approaches: int = 4                   # N, S, E, W
    n_lanes_per_approach: int = 3           # straight-left, straight-right, left-turn
    n_total_lanes: int = 12                 # 4 × 3
    n_phases: int = 4                       # NS-Straight, NS-Left, EW-Straight, EW-Left
    phase_names: list = field(default_factory=lambda: [
        "NS_Straight", "NS_Left", "EW_Straight", "EW_Left"
    ])

    # Phase → lane index mapping
    phase_to_lanes: dict = field(default_factory=lambda: {
        0: [0, 1, 6, 7],   # NS Straight: N-straight + S-straight
        1: [2, 8],          # NS Left:     N-left + S-left
        2: [3, 4, 9, 10],  # EW Straight: E-straight + W-straight
        3: [5, 11],         # EW Left:     E-left + W-left
    })


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTER VISION (CV PIPELINE)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CVConfig:
    # Camera
    frame_width: int = 1280
    frame_height: int = 720
    fps: float = 25.0

    # Bird's Eye View (Homography)
    # Default source points for 1280×720, ~45° overhead camera
    # CALIBRATE THESE for your actual camera!
    bev_src_points: list = field(default_factory=lambda: [
        [350, 280], [930, 280], [1200, 650], [80, 650]
    ])
    bev_width: int = 600
    bev_height: int = 800
    pixels_per_metre: float = 8.0       # BEV pixels per real-world metre

    # Lane boundaries in BEV frame (x-coordinates, pixels)
    # Divides BEV width into 12 lane columns
    lane_boundaries_x: list = field(default_factory=lambda: [
        50, 100, 150,    # North approach lanes (left side)
        200, 250, 300,   # East approach lanes
        350, 400, 450,   # South approach lanes
        500, 550         # West approach lanes
    ])

    # YOLOv8
    yolo_model: str = "yolov8n.pt"     # nano=fastest; yolov8x.pt=most accurate
    yolo_conf: float = 0.40
    yolo_classes: list = field(default_factory=lambda: [2, 3, 5, 7])
    # COCO class IDs: 2=car, 3=motorcycle, 5=bus, 7=truck

    # MOG2 Background Subtraction
    mog2_history: int = 300
    mog2_threshold: float = 50.0
    mog2_detect_shadows: bool = True
    stall_frames_threshold: int = 75    # 3 seconds at 25fps = stalled vehicle

    # Vehicle type classification (for novel feature)
    # PCU = Passenger Car Unit (space equivalence)
    pcu_values: dict = field(default_factory=lambda: {
        "car": 1.0, "truck": 2.5, "bus": 3.0,
        "motorcycle": 0.5, "bicycle": 0.3
    })
    two_wheeler_classes: set = field(default_factory=lambda: {"motorcycle", "bicycle"})


# ══════════════════════════════════════════════════════════════════════════════
# DEEP LEARNING (TRANSFORMER)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TokenConfig:
    """
    Defines EXACTLY what one token is.
    This answers "Loophole 1" from the analysis.
    One token = one lane's state vector at one timestep.
    """
    features: list = field(default_factory=lambda: [
        # Index  Feature                  Range        Source
        "vehicle_count",           # 0-20         YOLO count
        "queue_length_m",          # 0-150m       Homography distance
        "avg_wait_s",              # 0-300s       MOG2 stall timer
        "pcu_weighted_density",    # 0-1          YOLO + PCU weights
        "two_wheeler_ratio",       # 0-1          ← NOVEL FEATURE
        "motion_ratio",            # 0-1          MOG2 foreground fraction
        "is_stalled",              # 0 or 1       MOG2 stall detection
        "phase_active",            # 0 or 1       current signal state
    ])
    d_feature: int = 8             # Must equal len(features)
    n_lanes: int = 12


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4               # d_model must be divisible by n_heads
    n_encoder_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1
    n_phases: int = 4

    # Training
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    n_epochs: int = 60
    batch_size: int = 128
    patience: int = 12
    grad_clip: float = 1.0

    # Paths
    checkpoint_dir: str = "checkpoints"
    best_model_path: str = "checkpoints/best_model.pt"
    ablation_model_path: str = "checkpoints/ablation_model.pt"


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION (SUMO)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationConfig:
    sumo_home: str = os.environ.get("SUMO_HOME", "/usr/share/sumo")
    net_file: str = "simulation/intersection.net.xml"
    route_file: str = "simulation/routes.rou.xml"
    config_file: str = "simulation/intersection.sumocfg"
    step_length: float = 1.0       # seconds per simulation step
    port: int = 8813

    # Traffic scenarios
    scenarios: dict = field(default_factory=lambda: {
        "low":    {"scale": 0.3, "two_wheeler_frac": 0.50, "steps": 3600},
        "medium": {"scale": 0.6, "two_wheeler_frac": 0.65, "steps": 3600},
        "peak":   {"scale": 0.9, "two_wheeler_frac": 0.72, "steps": 3600},
    })


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalConfig:
    fixed_green_s: float = 30.0
    fixed_yellow_s: float = 3.0
    fixed_all_red_s: float = 1.0

    # HCM 6th Edition Level of Service (seconds/vehicle delay)
    los_thresholds: dict = field(default_factory=lambda: {
        "A": 10, "B": 20, "C": 35, "D": 55, "E": 80, "F": float("inf")
    })

    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42


# ── Global singletons ─────────────────────────────────────────────────────────
INTERSECTION = IntersectionConfig()
CV = CVConfig()
TOKEN = TokenConfig()
MODEL = ModelConfig()
SIM = SimulationConfig()
EVAL = EvalConfig()
