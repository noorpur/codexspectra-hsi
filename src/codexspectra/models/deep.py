from __future__ import annotations

import torch
from torch import nn


class SpectralMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, width: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, width),
            nn.GELU(),
            nn.BatchNorm1d(width),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpectralCNN1D(nn.Module):
    def __init__(self, n_bands: int, n_classes: int, channels: int = 48, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels, 7, padding=3),
            nn.GELU(),
            nn.BatchNorm1d(channels),
            nn.Conv1d(channels, channels * 2, 5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(channels * 2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels * 2, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x[:, None, :]
        return self.head(self.encoder(x))


class SpectralAutoencoder(nn.Module):
    def __init__(self, n_features: int, latent_dim: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_features, 128), nn.GELU(), nn.Linear(128, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.GELU(), nn.Linear(128, n_features))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z
