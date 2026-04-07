"""
data_gen/dataset.py
===================
PyTorch Dataset wrapping the traffic CSV.

Input tensor:  [12 lanes, 8 features]  (normalised)
Label tensor:  int in {0,1,2,3}        (optimal phase)
"""

import os, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import TOKEN, MODEL, EVAL


class TrafficDataset(Dataset):
    def __init__(self, csv_path: str = "data/traffic_dataset.csv"):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Dataset not found. Run: python data_gen/sumo_data_gen.py"
            )
        df = pd.read_csv(csv_path)
        norm_cols = [f"{f}_norm" for f in TOKEN.features]

        self.samples = []
        for (_, t), grp in df.groupby(["scenario", "timestep"]):
            grp = grp.sort_values("lane_id")
            if len(grp) != TOKEN.n_lanes:
                continue
            features = grp[norm_cols].values.astype(np.float32)  # [12, 8]
            label = int(grp["optimal_phase"].iloc[0])
            self.samples.append((features, label))

        print(f"Dataset: {len(self.samples):,} samples loaded from {csv_path}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        f, l = self.samples[idx]
        return torch.tensor(f), torch.tensor(l, dtype=torch.long)


def get_dataloaders(csv_path="data/traffic_dataset.csv"):
    ds = TrafficDataset(csv_path)
    n = len(ds)
    n_tr = int(n * EVAL.train_split)
    n_vl = int(n * EVAL.val_split)
    n_te = n - n_tr - n_vl
    tr, vl, te = random_split(
        ds, [n_tr, n_vl, n_te],
        generator=torch.Generator().manual_seed(EVAL.random_seed)
    )
    kw = dict(batch_size=MODEL.batch_size, num_workers=0)
    return (DataLoader(tr, shuffle=True, **kw),
            DataLoader(vl, shuffle=False, **kw),
            DataLoader(te, shuffle=False, **kw))
