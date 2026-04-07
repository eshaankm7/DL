"""
dl_pipeline/transformer.py
===========================
LANE-AWARE ATTENTION NETWORK (LAAN)
======================================

Input:  [batch, 12 lanes, 8 features]  ← directly from CVPipeline output
Output: [batch, 4 phases]              ← optimal phase probabilities

Architecture decisions, all justified:

  Q: Why Transformer and not LSTM?
  A: LSTM processes lanes sequentially (N→S→E→W). Our Transformer
     attends to ALL 12 lanes simultaneously. It can learn in one
     forward pass that a blocked North approach means more green
     is needed for South. LSTM can't learn this cross-lane dependency
     without many layers of hidden state.

  Q: Why not raw ViT (image patches)?
  A: ViT requires image patches in a grid [H/P, W/P]. Our input is
     lane vectors — there's no spatial grid. Using ViT would be
     architecturally wrong. LAAN treats each lane as one token,
     which is the correct problem formulation.

  Q: What is one token?
  A: One token = one lane's 8-feature state vector.
     12 lanes = 12 tokens. The Transformer learns which lanes
     should influence each other's signal timing.

  Q: What's novel in the architecture?
  A: The two_wheeler_ratio feature (index 4 of each token).
     No published ITSC Transformer (CoLight, AttendLight, PressLight)
     includes vehicle type composition in the state representation.
     Our ablation study proves this feature contributes X% accuracy.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import TOKEN, MODEL, INTERSECTION


class LaneEmbedding(nn.Module):
    """
    Learnable lane identity embeddings.

    Each of the 12 lanes gets a unique trainable embedding vector.
    This gives the model knowledge of WHICH lane it's processing:
    North-0 vs East-1 have fundamentally different roles in signal timing.

    This is different from positional encoding (which is positional order).
    Lanes have IDENTITY, not position.
    """
    def __init__(self, n_lanes: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(n_lanes, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_lanes, d_model]
        lane_ids = torch.arange(x.size(1), device=x.device)
        return x + self.embed(lane_ids).unsqueeze(0)


class LaneAwareAttentionNetwork(nn.Module):
    """
    The complete Transformer-based traffic signal classifier.

    Forward pass:
      [B, 12, 8] → Input Projection → [B, 12, 128]
                → Lane Embedding    → [B, 12, 128]
                → 3× TransformerEncoderLayer (4-head attention)
                → Mean Pool         → [B, 128]
                → Classifier MLP    → [B, 4] logits
    """

    def __init__(self):
        super().__init__()
        d_model = MODEL.d_model
        n_lanes = TOKEN.n_lanes
        d_feat  = TOKEN.d_feature

        # ── Input projection: 8-dim feature → 128-dim token ──────────────
        self.input_proj = nn.Sequential(
            nn.Linear(d_feat, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # ── Lane identity embeddings ───────────────────────────────────────
        self.lane_embed = LaneEmbedding(n_lanes, d_model)

        # ── Transformer encoder: 3 layers, 4 heads ────────────────────────
        # Each layer applies self-attention across all 12 lane tokens.
        # After 3 layers, each lane's representation encodes
        # the influence of all other lanes.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=MODEL.n_heads,
            dim_feedforward=MODEL.d_ff,
            dropout=MODEL.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,     # Pre-LayerNorm: more stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=MODEL.n_encoder_layers,
            enable_nested_tensor=False,
        )

        # ── Classifier head ────────────────────────────────────────────────
        # Mean-pools all 12 lane representations → intersection state vector
        # Then classifies into 4 signal phases.
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(MODEL.dropout),
            nn.Linear(d_model // 2, INTERSECTION.n_phases),
        )

        self._init_weights()
        print(f"LAAN initialized | params={self.count_parameters():,} | "
              f"d_model={d_model} | layers={MODEL.n_encoder_layers} | heads={MODEL.n_heads}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,         # [B, 12, 8]
        return_attention: bool = False,
    ) -> dict:
        """
        Args:
            x: Lane state tensor [batch, n_lanes=12, d_feature=8]
        Returns:
            dict['logits']:  [B, 4]  raw phase scores
            dict['encoded']: [B, 12, 128]  per-lane representations
        """
        # Project to model dimension
        x = self.input_proj(x)          # [B, 12, 128]

        # Add lane identity
        x = self.lane_embed(x)          # [B, 12, 128]

        # Self-attention: every lane attends to every other lane
        encoded = self.encoder(x)        # [B, 12, 128]

        # Global intersection state = mean of all lane representations
        pooled = encoded.mean(dim=1)     # [B, 128]

        # Phase classification
        logits = self.classifier(pooled) # [B, 4]

        return {"logits": logits, "encoded": encoded}

    def predict(self, token_array: torch.Tensor) -> int:
        """Single-sample inference. Returns optimal phase index (0-3)."""
        self.eval()
        with torch.no_grad():
            if token_array.dim() == 2:
                token_array = token_array.unsqueeze(0)
            out = self.forward(token_array)
            return int(out["logits"].argmax(-1).item())

    def predict_with_confidence(
        self, token_array: torch.Tensor
    ) -> tuple[int, float, list[float]]:
        """
        Returns (phase_idx, confidence, all_probs).
        Used by dashboard to show confidence bars.
        """
        self.eval()
        with torch.no_grad():
            if token_array.dim() == 2:
                token_array = token_array.unsqueeze(0)
            out = self.forward(token_array)
            probs = F.softmax(out["logits"], dim=-1).squeeze(0)
            phase = int(probs.argmax().item())
            conf = float(probs[phase].item())
            return phase, conf, probs.tolist()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = LaneAwareAttentionNetwork()
    dummy = torch.randn(4, TOKEN.n_lanes, TOKEN.d_feature)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Logits: {out['logits'].shape}")
    print("Model OK.")
