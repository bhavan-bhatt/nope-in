"""
IV Surface CNN Encoder.

Encodes 4×5 implied volatility surface grids into 64-dimensional embeddings
via a small CNN, pre-trained as an autoencoder on historical surfaces.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

TENORS = ["W1", "W2", "M1", "M2"]
MONEYNESS_BUCKETS = [0.90, 0.95, 1.00, 1.05, 1.10]


class IVSurfaceCNNEncoder(nn.Module):
    """Encode a 4×5 IV surface grid into a fixed-size embedding."""

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(2, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(2, 3), padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, embedding_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, 4, 5) surface values (NaN cells should be 0)
            mask: (batch, 1, 4, 5) boolean — True where data exists
        Returns:
            (batch, embedding_dim) embedding
        """
        x = x * mask.float()
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.pool(h).flatten(1)
        return self.fc(h)


class IVSurfaceAutoencoder(nn.Module):
    """Self-supervised autoencoder for IV surface pre-training."""

    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.encoder = IVSurfaceCNNEncoder(embedding_dim=embedding_dim)
        self.decoder_fc = nn.Linear(embedding_dim, 32 * 2 * 3)
        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=(2, 3), padding=0)
        self.deconv2 = nn.ConvTranspose2d(16, 1, kernel_size=(2, 3), padding=(0, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x, mask)
        h = self.decoder_fc(z).view(-1, 32, 2, 3)
        h = F.relu(self.deconv1(h))
        recon = self.deconv2(h)
        # Upsample/pad to target 4×5 if needed
        recon = F.interpolate(recon, size=(4, 5), mode="bilinear", align_corners=False)
        return recon, z


def _surfaces_to_tensors(surfaces: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert (N, 4, 5) surfaces to model input tensors with NaN masking."""
    arr = np.asarray(surfaces, dtype=np.float32)
    mask = np.isfinite(arr)
    filled = np.where(mask, arr, 0.0)
    x = torch.from_numpy(filled[:, None, :, :])
    m = torch.from_numpy(mask[:, None, :, :])
    return x, m


def pretrain_surface_encoder(
    surfaces: np.ndarray,
    output_path: Path,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    embedding_dim: int = 64,
    device: str | None = None,
) -> IVSurfaceCNNEncoder:
    """
    Pre-train the IV surface autoencoder on historical surface grids.
    Saves encoder state dict to output_path.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    x, mask = _surfaces_to_tensors(surfaces)
    model = IVSurfaceAutoencoder(embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    n = x.shape[0]
    indices = np.arange(n)

    for epoch in range(epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            xb = x[batch_idx].to(device)
            mb = mask[batch_idx].to(device)

            recon, _ = model(xb, mb)
            diff = (recon - xb) ** 2
            loss = diff[mb].mean() if mb.any() else diff.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        scheduler.step(avg_loss)
        if (epoch + 1) % 50 == 0:
            log.info("Surface AE epoch %d/%d — loss %.6f", epoch + 1, epochs, avg_loss)

    encoder = model.encoder
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), output_path)
    log.info("Saved surface encoder to %s", output_path)
    return encoder


def _extract_surface_matrix(row: pd.Series) -> np.ndarray:
    grid = np.full((4, 5), np.nan, dtype=float)
    for ti, tenor in enumerate(TENORS):
        for bi, bucket in enumerate(MONEYNESS_BUCKETS):
            col = f"surface_{tenor}_{bucket:.2f}"
            if col in row.index:
                grid[ti, bi] = row[col]
    return grid


def encode_surfaces_to_parquet(
    vol_surface_df: pd.DataFrame,
    encoder: IVSurfaceCNNEncoder,
    output_path: Path,
    device: str | None = None,
    batch_size: int = 256,
) -> pd.DataFrame:
    """
    Run frozen encoder on all surfaces; append surface_emb_0..63 columns.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = encoder.to(device)
    encoder.eval()

    matrices = np.stack([_extract_surface_matrix(row) for _, row in vol_surface_df.iterrows()])
    x, mask = _surfaces_to_tensors(matrices)

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size].to(device)
            mb = mask[start : start + batch_size].to(device)
            emb = encoder(xb, mb).cpu().numpy()
            embeddings.append(emb)

    emb_matrix = np.vstack(embeddings)
    out = vol_surface_df.copy()
    for i in range(emb_matrix.shape[1]):
        out[f"surface_emb_{i}"] = emb_matrix[:, i]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    log.info("Wrote surface embeddings: %d rows → %s", len(out), output_path)
    return out
