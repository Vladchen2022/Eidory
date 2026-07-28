from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from eidory.core.image_loader import open_local_image


ART_ANALYSIS_VERSION = "art-analysis-v3"
DEFAULT_PALETTE_SIZE = 8
DEFAULT_MAX_EDGE = 448
_FIT_SAMPLE_LIMIT = 60_000
_VISIBLE_ALPHA_THRESHOLD = 16
_PALETTE_MERGE_DELTA_E = 7.0
_MIN_PALETTE_PROPORTION = 0.005


@dataclass(frozen=True)
class PaletteEntry:
    rgb: tuple[int, int, int]
    proportion: float
    spatial_distribution: tuple[float, ...]

    @property
    def hex_code(self) -> str:
        return "#{:02X}{:02X}{:02X}".format(*self.rgb)


@dataclass(frozen=True)
class DistributionEntry:
    label: str
    proportion: float
    spatial_distribution: tuple[float, ...] = (0.0,) * 9


@dataclass(frozen=True)
class ArtAnalysisResult:
    palette: tuple[PaletteEntry, ...]
    lightness_distribution: tuple[DistributionEntry, ...]
    saturation_distribution: tuple[DistributionEntry, ...]
    temperature_distribution: tuple[DistributionEntry, ...]
    average_lightness: float
    tonal_range: float
    lightness_stddev: float
    contrast_label: str
    shadow_clipping: float
    highlight_clipping: float
    harmony_label: str
    harmony_description: str
    detail_density: float
    detail_label: str
    detail_distribution: tuple[float, ...]
    visual_center_x: float
    visual_center_y: float
    horizontal_similarity: float
    sampled_pixels: int
    sampled_width: int
    sampled_height: int


def analyze_image_art(
    image_path: str,
    *,
    palette_size: int = DEFAULT_PALETTE_SIZE,
    max_edge: int = DEFAULT_MAX_EDGE,
) -> ArtAnalysisResult:
    path = Path(image_path)
    with open_local_image(str(path)) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
        image.thumbnail((max(32, int(max_edge)), max(32, int(max_edge))), Image.Resampling.LANCZOS)
        rgba = np.asarray(image, dtype=np.uint8)

    if rgba.size == 0:
        raise ValueError(f"empty image: {path.name}")

    visible = rgba[..., 3] >= _VISIBLE_ALPHA_THRESHOLD
    rgb = rgba[..., :3][visible]
    if rgb.size == 0:
        raise ValueError(f"image has no visible pixels: {path.name}")

    rgb_float = rgb.astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb_float.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    palette_rgb, palette_proportions, palette_labels = _extract_palette(
        lab,
        palette_size=max(1, int(palette_size)),
    )
    visible_y, visible_x = np.nonzero(visible)
    cell_ids = np.minimum(visible_y * 3 // rgba.shape[0], 2) * 3 + np.minimum(
        visible_x * 3 // rgba.shape[1],
        2,
    )
    palette = tuple(
        PaletteEntry(
            rgb=rgb_value,
            proportion=proportion,
            spatial_distribution=_normalized_cells(cell_ids[palette_labels == index]),
        )
        for index, (rgb_value, proportion) in enumerate(
            zip(palette_rgb, palette_proportions, strict=False)
        )
    )

    lightness = np.clip(lab[:, 0], 0.0, 100.0)
    lightness_distribution = _make_distribution(
        lightness,
        bins=(0.0, 15.0, 32.0, 50.0, 68.0, 85.0, 100.01),
        labels=("极暗", "暗部", "中暗", "中间调", "亮部", "高光"),
        cell_ids=cell_ids,
    )

    channel_max = rgb_float.max(axis=1)
    channel_min = rgb_float.min(axis=1)
    saturation = np.divide(
        channel_max - channel_min,
        channel_max,
        out=np.zeros_like(channel_max),
        where=channel_max > 1e-6,
    )
    saturation_distribution = _make_distribution(
        saturation,
        bins=(0.0, 0.18, 0.55, 1.001),
        labels=("低饱和", "中饱和", "高饱和"),
        cell_ids=cell_ids,
    )
    temperature_distribution = _temperature_distribution(lab, saturation, cell_ids)

    average_lightness = float(np.mean(lightness))
    low, high = np.percentile(lightness, (5.0, 95.0))
    tonal_range = float(max(0.0, high - low))
    lightness_stddev = float(np.std(lightness))
    contrast_label = (
        "低对比"
        if lightness_stddev < 12.0
        else "中对比"
        if lightness_stddev < 24.0
        else "高对比"
    )
    harmony_label, harmony_description = _describe_harmony(palette)
    detail_metrics = _detail_metrics(rgba[..., :3], visible)

    return ArtAnalysisResult(
        palette=palette,
        lightness_distribution=lightness_distribution,
        saturation_distribution=saturation_distribution,
        temperature_distribution=temperature_distribution,
        average_lightness=average_lightness,
        tonal_range=tonal_range,
        lightness_stddev=lightness_stddev,
        contrast_label=contrast_label,
        shadow_clipping=float(np.mean(lightness <= 2.0)),
        highlight_clipping=float(np.mean(lightness >= 98.0)),
        harmony_label=harmony_label,
        harmony_description=harmony_description,
        detail_density=detail_metrics[0],
        detail_label=detail_metrics[1],
        detail_distribution=detail_metrics[2],
        visual_center_x=detail_metrics[3],
        visual_center_y=detail_metrics[4],
        horizontal_similarity=detail_metrics[5],
        sampled_pixels=int(rgb.shape[0]),
        sampled_width=int(rgba.shape[1]),
        sampled_height=int(rgba.shape[0]),
    )


def _extract_palette(
    lab: np.ndarray,
    *,
    palette_size: int,
) -> tuple[tuple[tuple[int, int, int], ...], tuple[float, ...], np.ndarray]:
    centers = _deterministic_lab_centers(lab, palette_size)
    distances = _squared_distances(lab, centers)
    labels = np.argmin(distances, axis=1)
    counts = np.bincount(labels, minlength=len(centers)).astype(np.int64)
    clusters = [
        [centers[index].astype(np.float64), int(count)]
        for index, count in enumerate(counts)
        if count > 0
    ]
    clusters.sort(key=lambda entry: entry[1], reverse=True)

    merged: list[list[object]] = []
    for center, count in clusters:
        nearest_index = None
        nearest_distance = float("inf")
        for index, (existing_center, _existing_count) in enumerate(merged):
            distance = float(np.linalg.norm(center - existing_center))
            if distance < nearest_distance:
                nearest_index = index
                nearest_distance = distance
        if nearest_index is not None and nearest_distance < _PALETTE_MERGE_DELTA_E:
            existing_center, existing_count = merged[nearest_index]
            total = int(existing_count) + int(count)
            merged[nearest_index] = [
                (existing_center * int(existing_count) + center * int(count)) / total,
                total,
            ]
        else:
            merged.append([center, int(count)])

    merged.sort(key=lambda entry: int(entry[1]), reverse=True)
    lab_centers = np.asarray([entry[0] for entry in merged], dtype=np.float32).reshape(1, -1, 3)
    rgb_centers = cv2.cvtColor(lab_centers, cv2.COLOR_LAB2RGB).reshape(-1, 3)
    rgb_centers = np.clip(np.rint(rgb_centers * 255.0), 0, 255).astype(np.uint8)
    total_pixels = float(sum(int(entry[1]) for entry in merged))
    final_centers = np.asarray([entry[0] for entry in merged], dtype=np.float32)
    final_labels = np.argmin(_squared_distances(lab, final_centers), axis=1)
    final_counts = np.bincount(final_labels, minlength=len(final_centers)).astype(np.int64)
    order = np.argsort(-final_counts)
    remap = np.empty(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order))
    final_labels = remap[final_labels]
    rgb_centers = rgb_centers[order]
    final_counts = final_counts[order]
    final_centers = final_centers[order]
    significant = final_counts >= max(1, int(np.ceil(total_pixels * _MIN_PALETTE_PROPORTION)))
    if not np.all(significant):
        final_centers = final_centers[significant]
        rgb_centers = rgb_centers[significant]
        final_labels = np.argmin(_squared_distances(lab, final_centers), axis=1)
        final_counts = np.bincount(final_labels, minlength=len(final_centers)).astype(np.int64)
        significant_order = np.argsort(-final_counts)
        significant_remap = np.empty(len(significant_order), dtype=np.int64)
        significant_remap[significant_order] = np.arange(len(significant_order))
        final_labels = significant_remap[final_labels]
        rgb_centers = rgb_centers[significant_order]
        final_counts = final_counts[significant_order]
    return (
        tuple((int(rgb[0]), int(rgb[1]), int(rgb[2])) for rgb in rgb_centers),
        tuple(float(count) / total_pixels for count in final_counts),
        final_labels,
    )


def render_art_analysis_view(
    image_path: str,
    result: ArtAnalysisResult,
    *,
    mode: str,
    selected_palette_index: int = 0,
    max_width: int,
    max_height: int,
) -> Image.Image:
    with open_local_image(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
        image.thumbnail((max(1, max_width), max(1, max_height)), Image.Resampling.LANCZOS)
        rgba = np.asarray(image, dtype=np.uint8).copy()
    rgb = rgba[..., :3].copy()
    if mode == "detail":
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        scale = max(float(np.percentile(magnitude, 96.0)), 1.0)
        heat = np.clip(magnitude / scale * 255.0, 0, 255).astype(np.uint8)
        heat_rgb = cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        rgba[..., :3] = cv2.addWeighted(rgb, 0.38, heat_rgb, 0.62, 0)
        return Image.fromarray(rgba, "RGBA")

    if mode in {
        "lightness_highlight",
        "saturation_highlight",
        "temperature_highlight",
    }:
        pixels = rgb.reshape(-1, 3).astype(np.float32) / 255.0
        pixel_lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
        selected = max(0, int(selected_palette_index))
        if mode == "lightness_highlight":
            values = np.clip(pixel_lab[:, 0], 0.0, 100.0)
            bins = np.asarray((0.0, 15.0, 32.0, 50.0, 68.0, 85.0, 100.01))
            selected = min(selected, len(bins) - 2)
            mask = (values >= bins[selected]) & (values < bins[selected + 1])
        else:
            channel_max = pixels.max(axis=1)
            channel_min = pixels.min(axis=1)
            saturation = np.divide(
                channel_max - channel_min,
                channel_max,
                out=np.zeros_like(channel_max),
                where=channel_max > 1e-6,
            )
            if mode == "saturation_highlight":
                bins = np.asarray((0.0, 0.18, 0.55, 1.001))
                selected = min(selected, len(bins) - 2)
                mask = (saturation >= bins[selected]) & (saturation < bins[selected + 1])
            else:
                chroma = np.hypot(pixel_lab[:, 1], pixel_lab[:, 2])
                neutral = (chroma < 12.0) | (saturation < 0.12)
                hue = (
                    np.degrees(np.arctan2(pixel_lab[:, 2], pixel_lab[:, 1])) + 360.0
                ) % 360.0
                warm = ~neutral & ((hue <= 135.0) | (hue >= 315.0))
                temperature_masks = (warm, ~neutral & ~warm, neutral)
                selected = min(selected, len(temperature_masks) - 1)
                mask = temperature_masks[selected]
        return _render_highlight_mask(rgba, rgb, mask.reshape(rgb.shape[:2]))

    palette = np.asarray([entry.rgb for entry in result.palette], dtype=np.uint8)
    pixels = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    pixel_lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    palette_lab = cv2.cvtColor(
        (palette.astype(np.float32) / 255.0).reshape(-1, 1, 3),
        cv2.COLOR_RGB2LAB,
    ).reshape(-1, 3)
    labels = np.argmin(_squared_distances(pixel_lab, palette_lab), axis=1)
    if mode == "palette":
        rgba[..., :3] = palette[labels].reshape(rgb.shape)
        return Image.fromarray(rgba, "RGBA")
    if mode == "highlight":
        selected = int(np.clip(selected_palette_index, 0, len(palette) - 1))
        mask = (labels == selected).reshape(rgb.shape[:2])
        return _render_highlight_mask(rgba, rgb, mask)
    return Image.fromarray(rgba, "RGBA")


def _render_highlight_mask(
    rgba: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
) -> Image.Image:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    subdued = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    subdued = np.clip(subdued.astype(np.float32) * 0.34 + 18.0, 0, 255).astype(np.uint8)
    rgba[..., :3] = subdued
    rgba[mask, :3] = rgb[mask]
    return Image.fromarray(rgba, "RGBA")


def _deterministic_lab_centers(lab: np.ndarray, palette_size: int) -> np.ndarray:
    if lab.shape[0] > _FIT_SAMPLE_LIMIT:
        indices = np.linspace(0, lab.shape[0] - 1, _FIT_SAMPLE_LIMIT, dtype=np.int64)
        sample = lab[indices]
    else:
        sample = lab

    mean = np.mean(sample, axis=0)
    first_index = int(np.argmin(np.sum((sample - mean) ** 2, axis=1)))
    centers = [sample[first_index].astype(np.float64)]
    closest = np.sum((sample - centers[0]) ** 2, axis=1)

    for _index in range(1, min(palette_size, sample.shape[0])):
        farthest_index = int(np.argmax(closest))
        if float(closest[farthest_index]) < 0.25:
            break
        center = sample[farthest_index].astype(np.float64)
        centers.append(center)
        closest = np.minimum(closest, np.sum((sample - center) ** 2, axis=1))

    center_array = np.asarray(centers, dtype=np.float64)
    for _iteration in range(18):
        distances = _squared_distances(sample, center_array)
        labels = np.argmin(distances, axis=1)
        updated = center_array.copy()
        for index in range(len(center_array)):
            members = sample[labels == index]
            if members.size:
                updated[index] = np.mean(members, axis=0)
        movement = float(np.max(np.linalg.norm(updated - center_array, axis=1)))
        center_array = updated
        if movement < 0.08:
            break
    return center_array.astype(np.float32)


def _squared_distances(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)


def _make_distribution(
    values: np.ndarray,
    *,
    bins: tuple[float, ...],
    labels: tuple[str, ...],
    cell_ids: np.ndarray,
) -> tuple[DistributionEntry, ...]:
    bin_edges = np.asarray(bins, dtype=np.float32)
    counts, _edges = np.histogram(values, bins=bin_edges)
    total = float(counts.sum())
    if total <= 0:
        return tuple(
            DistributionEntry(
                label=label,
                proportion=0.0,
                spatial_distribution=(0.0,) * 9,
            )
            for label in labels
        )
    return tuple(
        DistributionEntry(
            label=label,
            proportion=float(count) / total,
            spatial_distribution=_normalized_cells(
                cell_ids[
                    (values >= bin_edges[index])
                    & (values < bin_edges[index + 1])
                ]
            ),
        )
        for index, (label, count) in enumerate(zip(labels, counts, strict=False))
    )


def _temperature_distribution(
    lab: np.ndarray,
    saturation: np.ndarray,
    cell_ids: np.ndarray,
) -> tuple[DistributionEntry, ...]:
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    neutral = (chroma < 12.0) | (saturation < 0.12)
    hue = (np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) + 360.0) % 360.0
    warm = ~neutral & ((hue <= 135.0) | (hue >= 315.0))
    cool = ~neutral & ~warm
    total = float(len(lab))
    return (
        DistributionEntry(
            "暖色",
            float(np.count_nonzero(warm)) / total,
            _normalized_cells(cell_ids[warm]),
        ),
        DistributionEntry(
            "冷色",
            float(np.count_nonzero(cool)) / total,
            _normalized_cells(cell_ids[cool]),
        ),
        DistributionEntry(
            "中性色",
            float(np.count_nonzero(neutral)) / total,
            _normalized_cells(cell_ids[neutral]),
        ),
    )


def _normalized_cells(cell_ids: np.ndarray) -> tuple[float, ...]:
    counts = np.bincount(cell_ids.astype(np.int64), minlength=9).astype(np.float64)
    total = float(counts.sum())
    if total <= 0.0:
        return (0.0,) * 9
    return tuple(float(value / total) for value in counts[:9])


def _describe_harmony(
    palette: tuple[PaletteEntry, ...],
) -> tuple[str, str]:
    significant = [
        entry
        for entry in palette
        if entry.proportion >= 0.025
    ]
    if not significant:
        return "未识别", "有效色彩不足，无法判断配色关系。"

    rgb = np.asarray([entry.rgb for entry in significant], dtype=np.float32) / 255.0
    lab = cv2.cvtColor(rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    colorful = lab[chroma >= 12.0]
    if len(colorful) < 2:
        return "单色 / 中性倾向", "主要颜色的色相差异较小，画面更多依靠明度层次组织。"

    hues = (np.degrees(np.arctan2(colorful[:, 2], colorful[:, 1])) + 360.0) % 360.0
    differences: list[float] = []
    for first in range(len(hues)):
        for second in range(first + 1, len(hues)):
            raw = abs(float(hues[first]) - float(hues[second]))
            differences.append(min(raw, 360.0 - raw))

    if differences and max(differences) <= 55.0:
        return "邻近色倾向", "主要色相彼此接近，整体统一，变化主要来自明度和饱和度。"
    if any(145.0 <= difference <= 180.0 for difference in differences):
        return "互补色倾向", "主要色相存在明显对置关系，容易形成色彩张力和视觉焦点。"
    if len(hues) >= 3:
        ordered = np.sort(hues)
        gaps = np.diff(np.concatenate((ordered, ordered[:1] + 360.0)))
        if bool(np.all((gaps >= 75.0) & (gaps <= 165.0))):
            return "三角色倾向", "主要色相分布较均衡，画面具有多方向的色彩对比。"
    return "多色相组合", "主要色相分布较分散，不属于单一的经典配色关系。"


def _detail_metrics(
    rgb: np.ndarray,
    visible: np.ndarray,
) -> tuple[float, str, tuple[float, ...], float, float, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy) / 1020.0
    valid_magnitude = magnitude[visible]
    detail_density = (
        float(np.mean(valid_magnitude >= 0.075))
        if valid_magnitude.size
        else 0.0
    )
    detail_label = (
        "低细节"
        if detail_density < 0.08
        else "中等细节"
        if detail_density < 0.22
        else "高细节"
    )

    height, width = gray.shape
    y_indices, x_indices = np.indices((height, width))
    cell_ids = np.minimum(y_indices * 3 // max(height, 1), 2) * 3 + np.minimum(
        x_indices * 3 // max(width, 1),
        2,
    )
    detail_weight = np.where(visible, magnitude, 0.0)
    detail_cells = np.bincount(
        cell_ids.reshape(-1),
        weights=detail_weight.reshape(-1),
        minlength=9,
    ).astype(np.float64)
    cell_total = float(detail_cells.sum())
    detail_distribution = (
        tuple(float(value / cell_total) for value in detail_cells[:9])
        if cell_total > 1e-9
        else (0.0,) * 9
    )

    rgb_float = rgb.astype(np.float32) / 255.0
    channel_max = rgb_float.max(axis=2)
    channel_min = rgb_float.min(axis=2)
    saturation = np.divide(
        channel_max - channel_min,
        channel_max,
        out=np.zeros_like(channel_max),
        where=channel_max > 1e-6,
    )
    luminance_contrast = np.abs(gray - float(np.median(gray[visible]))) / 255.0
    saliency = np.where(
        visible,
        magnitude * 0.64 + saturation * 0.21 + luminance_contrast * 0.15,
        0.0,
    )
    saliency_total = float(saliency.sum())
    if saliency_total > 1e-9:
        visual_center_x = float((saliency * x_indices).sum() / saliency_total) / max(width - 1, 1)
        visual_center_y = float((saliency * y_indices).sum() / saliency_total) / max(height - 1, 1)
    else:
        visual_center_x = 0.5
        visual_center_y = 0.5

    half_width = width // 2
    if half_width <= 0:
        horizontal_similarity = 1.0
    else:
        left = gray[:, :half_width]
        right = np.fliplr(gray[:, width - half_width :])
        valid_pair = visible[:, :half_width] & np.fliplr(visible[:, width - half_width :])
        difference = np.abs(left - right) / 255.0
        horizontal_similarity = (
            float(1.0 - np.mean(difference[valid_pair]))
            if np.any(valid_pair)
            else 0.0
        )
    return (
        detail_density,
        detail_label,
        detail_distribution,
        float(np.clip(visual_center_x, 0.0, 1.0)),
        float(np.clip(visual_center_y, 0.0, 1.0)),
        float(np.clip(horizontal_similarity, 0.0, 1.0)),
    )
