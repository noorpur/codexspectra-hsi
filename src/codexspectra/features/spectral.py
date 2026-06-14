from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class SpectralPreprocessor:
    def __init__(
        self,
        smoothing_window: int = 11,
        smoothing_polyorder: int = 2,
        use_snv: bool = True,
        use_derivatives: bool = True,
        pca_components: int | None = 32,
    ) -> None:
        self.smoothing_window = smoothing_window
        self.smoothing_polyorder = smoothing_polyorder
        self.use_snv = use_snv
        self.use_derivatives = use_derivatives
        self.pca_components = pca_components
        self.scaler = StandardScaler()
        self.pca: PCA | None = None

    def _clean(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        lo, hi = np.percentile(x, [0.5, 99.5])
        x = np.clip(x, lo, hi)
        return x

    def _smooth(self, x: np.ndarray) -> np.ndarray:
        bands = x.shape[1]
        window = min(self.smoothing_window, bands - 1 if bands % 2 == 0 else bands)
        if window < 5:
            return x
        if window % 2 == 0:
            window -= 1
        return savgol_filter(x, window_length=window, polyorder=self.smoothing_polyorder, axis=1)

    def _snv(self, x: np.ndarray) -> np.ndarray:
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True) + 1e-8
        return (x - mu) / sd

    def _derivatives(self, x: np.ndarray) -> np.ndarray:
        d1 = np.gradient(x, axis=1)
        d2 = np.gradient(d1, axis=1)
        return np.concatenate([x, d1, d2], axis=1)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        x = self._clean(x)
        x = self._smooth(x)
        if self.use_snv:
            x = self._snv(x)
        if self.use_derivatives:
            x = self._derivatives(x)
        x = self.scaler.fit_transform(x)
        if self.pca_components:
            n = min(self.pca_components, x.shape[1], x.shape[0] - 1)
            self.pca = PCA(n_components=max(1, n), random_state=1337)
            x = self.pca.fit_transform(x)
        return x.astype(np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = self._clean(x)
        x = self._smooth(x)
        if self.use_snv:
            x = self._snv(x)
        if self.use_derivatives:
            x = self._derivatives(x)
        x = self.scaler.transform(x)
        if self.pca is not None:
            x = self.pca.transform(x)
        return x.astype(np.float32)
