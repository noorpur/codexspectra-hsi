import numpy as np

from codexspectra.features.spectral import SpectralPreprocessor


def test_preprocessor_shapes():
    x = np.random.default_rng(0).normal(size=(20, 32)).astype("float32")
    prep = SpectralPreprocessor(pca_components=5)
    z = prep.fit_transform(x)
    assert z.shape == (20, 5)
    z2 = prep.transform(x[:3])
    assert z2.shape == (3, 5)
