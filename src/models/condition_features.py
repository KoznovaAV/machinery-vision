"""Colour measurements used alongside image embeddings for visible snow."""

import numpy as np
from PIL import Image, ImageOps


def weather_measurements(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB").resize((128, 128))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    values = pixels.max(axis=2)
    saturation = (values - pixels.min(axis=2)) / np.maximum(values, 1e-6)
    result = []
    for section in (slice(None), slice(64, None), slice(85, None), slice(42, 85)):
        v = values[section]
        s = saturation[section]
        result.extend((
            float(np.mean((v > 0.65) & (s < 0.15))),
            float(np.mean((v > 0.50) & (s < 0.20))),
            float(np.mean(v)),
            float(np.mean(s)),
        ))
    return np.asarray(result, dtype=np.float32)
