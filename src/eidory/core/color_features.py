from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from eidory.core.image_loader import open_local_image


COLOR_VECTOR_VERSION = "hsv-hist-v1"
HUE_BINS = 12
SATURATION_BINS = 4
VALUE_BINS = 4
COLOR_VECTOR_DIM = HUE_BINS * SATURATION_BINS * VALUE_BINS


def encode_image_color(path: str, *, max_edge: int = 160) -> np.ndarray:
    with open_local_image(path) as image:
        image = ImageOps.exif_transpose(image)
        image = _to_rgb(image)
        image.thumbnail((max_edge, max_edge), Image.Resampling.BILINEAR)
        hsv = np.asarray(image.convert("HSV"), dtype=np.uint16)

    if hsv.size == 0:
        raise ValueError(f"empty image: {Path(path).name}")

    h = np.minimum((hsv[..., 0] * HUE_BINS) // 256, HUE_BINS - 1)
    s = np.minimum((hsv[..., 1] * SATURATION_BINS) // 256, SATURATION_BINS - 1)
    v = np.minimum((hsv[..., 2] * VALUE_BINS) // 256, VALUE_BINS - 1)
    bins = h * (SATURATION_BINS * VALUE_BINS) + s * VALUE_BINS + v
    hist = np.bincount(bins.reshape(-1), minlength=COLOR_VECTOR_DIM).astype(np.float32)
    total = float(hist.sum())
    if total <= 0:
        raise ValueError(f"empty color histogram: {Path(path).name}")
    return hist / total


def encode_query_color(rgb: tuple[int, int, int]) -> np.ndarray:
    red, green, blue = (max(0, min(255, int(channel))) for channel in rgb)
    hue, saturation, value = colorsys.rgb_to_hsv(red / 255.0, green / 255.0, blue / 255.0)
    query = np.zeros((COLOR_VECTOR_DIM,), dtype=np.float32)

    for h_bin in range(HUE_BINS):
        h_center = (h_bin + 0.5) / HUE_BINS
        hue_distance = abs(h_center - hue)
        hue_distance = min(hue_distance, 1.0 - hue_distance)
        for s_bin in range(SATURATION_BINS):
            s_center = (s_bin + 0.5) / SATURATION_BINS
            saturation_distance = abs(s_center - saturation)
            for v_bin in range(VALUE_BINS):
                v_center = (v_bin + 0.5) / VALUE_BINS
                value_distance = abs(v_center - value)
                index = h_bin * (SATURATION_BINS * VALUE_BINS) + s_bin * VALUE_BINS + v_bin
                query[index] = _color_similarity(
                    hue_distance=hue_distance,
                    saturation_distance=saturation_distance,
                    value_distance=value_distance,
                    saturation=saturation,
                    value=value,
                )
    return query


def _color_similarity(
    *,
    hue_distance: float,
    saturation_distance: float,
    value_distance: float,
    saturation: float,
    value: float,
) -> float:
    if value < 0.12:
        hue_term = 0.0
        saturation_term = saturation_distance / 0.75
        value_term = value_distance / 0.18
    elif saturation < 0.12:
        hue_term = 0.0
        saturation_term = saturation_distance / 0.22
        value_term = value_distance / 0.22
    else:
        hue_term = hue_distance / 0.10
        saturation_term = saturation_distance / 0.30
        value_term = value_distance / 0.35
    distance = hue_term * hue_term + saturation_term * saturation_term + value_term * value_term
    return float(np.exp(-0.5 * distance))


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    if image.mode != "RGB":
        return image.convert("RGB")
    return image
