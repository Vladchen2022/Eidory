from __future__ import annotations

from dataclasses import dataclass
from math import floor, sqrt
from typing import Literal

import cv2
import numpy as np
from PIL import Image


LineTopMode = Literal["line", "color_limit"]
LineTopPreset = Literal["photo", "illustration"]


@dataclass(frozen=True)
class LineTopSettings:
    mode: LineTopMode = "line"
    opacity: float = 1.0
    edge_strength: float = 2.0
    line_thickness: float = 0.0
    overlay_contrast: float = 1.0
    overlay_brightness: float = 0.0
    color_limit_steps: int = 8
    color_limit_grayscale: bool = False
    color_limit_shape_simplification: int = 1
    smart_enhance: bool = True
    smart_preset: LineTopPreset = "photo"
    enhanced_line_engine: bool = True

    def cache_key(self) -> tuple[object, ...]:
        return (
            self.mode,
            round(float(self.opacity), 3),
            round(float(self.edge_strength), 3),
            round(float(self.line_thickness), 3),
            round(float(self.overlay_contrast), 3),
            round(float(self.overlay_brightness), 3),
            int(self.color_limit_steps),
            bool(self.color_limit_grayscale),
            int(self.color_limit_shape_simplification),
            bool(self.smart_enhance),
            self.smart_preset,
            bool(self.enhanced_line_engine),
        )


def render_linetop_image(source: Image.Image, settings: LineTopSettings) -> Image.Image:
    source_rgba = source.convert("RGBA")
    rgba = np.array(source_rgba, dtype=np.uint8)
    photo_like = settings.smart_preset == "photo"

    if settings.mode == "color_limit":
        rendered = _color_limited_rgba_from_image(
            rgba,
            color_count=int(np.clip(round(settings.color_limit_steps), 0, 15)),
            grayscale=settings.color_limit_grayscale,
            shape_simplification=int(np.clip(round(settings.color_limit_shape_simplification), 1, 10)),
            photo_like=photo_like,
            smart_enhance=settings.smart_enhance,
            contrast=settings.overlay_contrast,
            brightness=settings.overlay_brightness,
        )
        rendered = _blend_rgba(rgba, rendered, settings.opacity)
        return Image.fromarray(rendered, "RGBA")

    if settings.enhanced_line_engine:
        gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
        alpha_mask = _thin_line_mask_from_gray(
            gray,
            photo_like=photo_like,
            edge_strength=settings.edge_strength,
            contrast=settings.overlay_contrast,
            brightness=settings.overlay_brightness,
            line_thickness=settings.line_thickness,
        )
    else:
        alpha_mask = _legacy_line_mask_from_rgba(
            rgba,
            photo_like=photo_like,
            edge_strength=settings.edge_strength,
            contrast=settings.overlay_contrast,
            brightness=settings.overlay_brightness,
            line_thickness=settings.line_thickness,
        )
    rendered = _line_mask_to_white_rgba(alpha_mask, settings.opacity)
    return Image.fromarray(rendered, "RGBA")


def render_linetop_overlay_image(source: Image.Image, settings: LineTopSettings) -> Image.Image:
    source_rgba = source.convert("RGBA")
    rgba = np.array(source_rgba, dtype=np.uint8)
    photo_like = settings.smart_preset == "photo"

    if settings.mode == "color_limit":
        rendered = _color_limited_rgba_from_image(
            rgba,
            color_count=int(np.clip(round(settings.color_limit_steps), 0, 15)),
            grayscale=settings.color_limit_grayscale,
            shape_simplification=int(np.clip(round(settings.color_limit_shape_simplification), 1, 10)),
            photo_like=photo_like,
            smart_enhance=settings.smart_enhance,
            contrast=settings.overlay_contrast,
            brightness=settings.overlay_brightness,
        )
        return Image.fromarray(rendered, "RGBA")

    if settings.enhanced_line_engine:
        gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
        alpha_mask = _thin_line_mask_from_gray(
            gray,
            photo_like=photo_like,
            edge_strength=settings.edge_strength,
            contrast=settings.overlay_contrast,
            brightness=settings.overlay_brightness,
            line_thickness=settings.line_thickness,
        )
    else:
        alpha_mask = _legacy_line_mask_from_rgba(
            rgba,
            photo_like=photo_like,
            edge_strength=settings.edge_strength,
            contrast=settings.overlay_contrast,
            brightness=settings.overlay_brightness,
            line_thickness=settings.line_thickness,
        )
    rendered = _line_mask_to_transparent_rgba(alpha_mask)
    return Image.fromarray(rendered, "RGBA")


def _line_mask_to_white_rgba(alpha_mask: np.ndarray, opacity: float) -> np.ndarray:
    alpha = np.clip(alpha_mask.astype(np.float32) * np.clip(opacity, 0.05, 1.0) / 255.0, 0.0, 1.0)
    value = np.clip(255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)
    output = np.empty((alpha_mask.shape[0], alpha_mask.shape[1], 4), dtype=np.uint8)
    output[:, :, 0] = value
    output[:, :, 1] = value
    output[:, :, 2] = value
    output[:, :, 3] = 255
    return output


def _line_mask_to_transparent_rgba(alpha_mask: np.ndarray) -> np.ndarray:
    output = np.zeros((alpha_mask.shape[0], alpha_mask.shape[1], 4), dtype=np.uint8)
    output[:, :, 3] = alpha_mask
    return output


def _blend_rgba(source: np.ndarray, rendered: np.ndarray, opacity: float) -> np.ndarray:
    alpha = float(np.clip(opacity, 0.05, 1.0))
    if alpha >= 0.999:
        return rendered
    blended = source.astype(np.float32) * (1.0 - alpha) + rendered.astype(np.float32) * alpha
    blended[:, :, 3] = rendered[:, :, 3]
    return np.clip(blended, 0, 255).astype(np.uint8)


def _remove_tiny_components(binary: np.ndarray, photo_like: bool) -> np.ndarray:
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)
    filtered = np.zeros(binary.shape, dtype=np.uint8)
    min_area = 14 if photo_like else 8
    min_long_side = 18 if photo_like else 12
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < min_area and max(width, height) < min_long_side:
            continue
        filtered[labels == label] = 255
    return filtered


def _thin_line_mask_from_gray(
    gray: np.ndarray,
    *,
    photo_like: bool,
    edge_strength: float,
    contrast: float,
    brightness: float,
    line_thickness: float,
) -> np.ndarray:
    alpha = float(np.clip(1.05 + contrast * 0.22, 0.9, 1.7))
    beta = float(np.clip(brightness * 38.0, -18.0, 18.0))
    prepared = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    if photo_like:
        prepared = cv2.bilateralFilter(prepared, 7, 30, 30)
    else:
        prepared = cv2.GaussianBlur(prepared, (3, 3), 0.6)

    try:
        edge_drawing = cv2.ximgproc.createEdgeDrawing()
        params = edge_drawing.Params()
        params.EdgeDetectionOperator = cv2.ximgproc.EdgeDrawing.SOBEL
        params.GradientThresholdValue = (
            int(round(np.clip(34.0 + edge_strength * 4.0, 28.0, 44.0)))
            if photo_like
            else int(round(np.clip(24.0 + edge_strength * 3.0, 18.0, 34.0)))
        )
        params.AnchorThresholdValue = 0
        params.ScanInterval = 1
        params.MinPathLength = 18 if photo_like else 12
        params.Sigma = 1.2 if photo_like else 0.8
        params.NFAValidation = False
        edge_drawing.setParams(params)
        edge_drawing.detectEdges(prepared)
        edge_image = edge_drawing.getEdgeImage()
    except Exception:
        edge_image = cv2.Canny(
            prepared,
            42.0 if photo_like else 28.0,
            110.0 if photo_like else 78.0,
            apertureSize=3,
            L2gradient=True,
        )
    _threshold_value, edge_image = cv2.threshold(edge_image, 0, 255, cv2.THRESH_BINARY)

    thinned = cv2.ximgproc.thinning(edge_image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    thinned = _remove_tiny_components(thinned, photo_like)
    if photo_like:
        thinned = cv2.ximgproc.thinning(thinned, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    max_alpha = 176.0 if photo_like else 210.0
    alpha_mask = cv2.convertScaleAbs(thinned, alpha=max_alpha / 255.0)
    if line_thickness > 0.0:
        radius = max(0, int(round(line_thickness * 0.35)))
        if radius > 0:
            kernel_size = radius * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            alpha_mask = cv2.dilate(alpha_mask, kernel)
    return alpha_mask


def _legacy_line_mask_from_rgba(
    rgba: np.ndarray,
    *,
    photo_like: bool,
    edge_strength: float,
    contrast: float,
    brightness: float,
    line_thickness: float,
) -> np.ndarray:
    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
    alpha = float(np.clip(1.0 + contrast * 0.35, 0.8, 2.4))
    beta = float(np.clip(brightness * 45.0, -32.0, 32.0))
    tuned = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    tuned = cv2.GaussianBlur(tuned, (3, 3), 0.8 if photo_like else 0.45)
    low = float(np.clip(38.0 - edge_strength * 4.0, 14.0, 70.0))
    high = float(np.clip(115.0 - edge_strength * 7.0, 48.0, 160.0))
    edges = cv2.Canny(tuned, low, high, apertureSize=3, L2gradient=True)
    edges = cv2.ximgproc.thinning(edges, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    edges = _remove_tiny_components(edges, photo_like)
    max_alpha = 150.0 if photo_like else 190.0
    alpha_mask = cv2.convertScaleAbs(edges, alpha=max_alpha / 255.0)
    if line_thickness > 0.0:
        radius = max(0, int(round(line_thickness * 0.45)))
        if radius > 0:
            kernel_size = radius * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            alpha_mask = cv2.dilate(alpha_mask, kernel)
    return alpha_mask


def _collect_palette(rgb: np.ndarray) -> list[tuple[tuple[int, int, int], int]]:
    colors, counts = np.unique(rgb.reshape(-1, 3), axis=0, return_counts=True)
    return [
        ((int(color[0]), int(color[1]), int(color[2])), int(count))
        for color, count in zip(colors, counts, strict=False)
    ]


def _effective_shape_simplification_level(ui_level: float) -> float:
    normalized = float(np.clip((ui_level - 1.0) / 9.0, 0.0, 1.0))
    return 1.0 + normalized * 3.0


def _shape_simplification_progress(effective_level: float) -> float:
    return float(np.clip((effective_level - 1.0) / 3.0, 0.0, 1.0))


def _contour_circularity(contour: np.ndarray) -> float:
    area = abs(float(cv2.contourArea(contour)))
    perimeter = float(cv2.arcLength(contour, True))
    if area <= 1.0 or perimeter <= 1.0:
        return 0.0
    return float((4.0 * np.pi * area) / (perimeter * perimeter))


def _pick_replacement_color(
    rgb: np.ndarray,
    component_mask: np.ndarray,
    current_color: tuple[int, int, int],
    ring_radius: int,
) -> tuple[int, int, int] | None:
    kernel_size = max(3, ring_radius * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(component_mask, kernel)
    ring_mask = cv2.subtract(dilated, component_mask)
    neighbor_pixels = rgb[ring_mask > 0]
    if neighbor_pixels.size == 0:
        return None
    colors, counts = np.unique(neighbor_pixels.reshape(-1, 3), axis=0, return_counts=True)
    if colors.size == 0:
        return None
    current = np.array(current_color, dtype=np.uint8)
    keep = np.any(colors != current, axis=1)
    if not np.any(keep):
        return None
    colors = colors[keep]
    counts = counts[keep]
    color = colors[int(np.argmax(counts))]
    return int(color[0]), int(color[1]), int(color[2])


def _merge_small_color_regions(quantized_rgb: np.ndarray, effective_level: float, photo_like: bool) -> np.ndarray:
    if effective_level <= 1.0 or quantized_rgb.size == 0:
        return quantized_rgb
    output = quantized_rgb.copy()
    progress = _shape_simplification_progress(effective_level)
    simplification_amount = progress**1.03
    image_area = float(quantized_rgb.shape[0] * quantized_rgb.shape[1])
    min_area_threshold = max(
        20.0 if photo_like else 14.0,
        image_area * (0.00002 + simplification_amount * (0.0050 if photo_like else 0.0038)),
    )
    long_side_threshold = (16.0 if photo_like else 12.0) + simplification_amount * (88.0 if photo_like else 68.0)
    ring_radius = max(1, int(round(1.0 + simplification_amount * (5.0 if photo_like else 4.0))))
    passes = 3 if simplification_amount >= 0.72 else 2 if simplification_amount >= 0.24 else 1

    for pass_index in range(passes):
        changed = False
        palette = sorted(_collect_palette(output), key=lambda entry: entry[1])
        pass_area_threshold = min_area_threshold * (1.0 + pass_index * 0.45)
        pass_long_side_threshold = long_side_threshold * (1.0 + pass_index * 0.18)
        for color, _count in palette:
            lower = np.array(color, dtype=np.uint8)
            mask = cv2.inRange(output, lower, lower)
            component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
            for label in range(1, component_count):
                area = int(stats[label, cv2.CC_STAT_AREA])
                width = int(stats[label, cv2.CC_STAT_WIDTH])
                height = int(stats[label, cv2.CC_STAT_HEIGHT])
                long_side = float(max(width, height))
                short_side = float(min(width, height))
                tiny_island = area < pass_area_threshold
                compact_patch = (
                    area < pass_area_threshold * 2.2
                    and long_side < pass_long_side_threshold
                    and short_side < pass_long_side_threshold * 0.65
                )
                if not tiny_island and not compact_patch:
                    continue
                component_mask = (labels == label).astype(np.uint8) * 255
                replacement = _pick_replacement_color(output, component_mask, color, ring_radius + pass_index)
                if replacement is None:
                    continue
                output[component_mask > 0] = replacement
                changed = True
        if not changed:
            break
    return output


def _blur_for_shape_simplification(input_rgb: np.ndarray, effective_level: float, photo_like: bool) -> np.ndarray:
    if effective_level <= 1.0 or input_rgb.size == 0:
        return input_rgb
    output = input_rgb.copy()
    progress = _shape_simplification_progress(effective_level)
    normalized = progress**0.92
    blur_level = normalized * 9.0
    primary_sigma = blur_level * (0.70 if photo_like else 0.58)
    if primary_sigma > 0.05:
        primary_radius = max(1, int(round(primary_sigma * 2.4)))
        primary_kernel = primary_radius * 2 + 1
        output = cv2.GaussianBlur(
            output,
            (primary_kernel, primary_kernel),
            primary_sigma,
            primary_sigma,
            borderType=cv2.BORDER_REPLICATE,
        )
    if progress >= 0.10:
        median_steps = int(np.clip(floor(progress * 4.2), 0, 4))
        output = cv2.medianBlur(output, 3 + median_steps * 2)
    if progress >= 0.18:
        downsample_amount = float(np.clip((progress - 0.18) / 0.82, 0.0, 1.0))
        scale = float(np.clip(1.0 - downsample_amount * (0.30 if photo_like else 0.24), 0.68, 1.0))
        reduced = cv2.resize(output, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        output = cv2.resize(reduced, (input_rgb.shape[1], input_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    if progress >= 0.40:
        mean_shift_amount = float(np.clip((progress - 0.40) / 0.60, 0.0, 1.0))
        spatial_radius = 4.0 + mean_shift_amount * (12.0 if photo_like else 9.0)
        color_radius = 7.0 + mean_shift_amount * (20.0 if photo_like else 15.0)
        max_level = 1 if mean_shift_amount >= 0.72 else 0
        output = cv2.pyrMeanShiftFiltering(output, spatial_radius, color_radius, maxLevel=max_level)
    if progress >= 0.64:
        finishing_amount = float(np.clip((progress - 0.64) / 0.36, 0.0, 1.0))
        finishing_sigma = finishing_amount * (2.0 if photo_like else 1.6)
        finishing_radius = max(1, int(round(finishing_sigma * 2.2)))
        finishing_kernel = finishing_radius * 2 + 1
        output = cv2.GaussianBlur(
            output,
            (finishing_kernel, finishing_kernel),
            finishing_sigma,
            finishing_sigma,
            borderType=cv2.BORDER_REPLICATE,
        )
    return output


def _geometrize_quantized_shapes(quantized_rgb: np.ndarray, effective_level: float, photo_like: bool) -> np.ndarray:
    if effective_level <= 1.0 or quantized_rgb.size == 0:
        return quantized_rgb
    progress = _shape_simplification_progress(effective_level)
    if progress < 0.16:
        return quantized_rgb
    palette = sorted(_collect_palette(quantized_rgb), key=lambda entry: entry[1], reverse=True)
    if not palette:
        return quantized_rgb
    output = quantized_rgb.copy()
    simplification_amount = progress**1.04
    image_area = float(quantized_rgb.shape[0] * quantized_rgb.shape[1])
    min_area_threshold = max(
        36.0 if photo_like else 24.0,
        image_area * (0.00003 + simplification_amount * (0.0021 if photo_like else 0.0016)),
    )
    open_radius = max(0, int(round(simplification_amount * (5.0 if photo_like else 4.0))))
    close_radius = max(1, int(round(1.0 + simplification_amount * (7.0 if photo_like else 5.0))))
    blur_radius = max(0, int(round(simplification_amount * (3.0 if photo_like else 2.0))))

    for color, _count in palette:
        lower = np.array(color, dtype=np.uint8)
        mask = cv2.inRange(quantized_rgb, lower, lower)
        if open_radius > 0:
            kernel_size = open_radius * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if close_radius > 0:
            kernel_size = close_radius * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if blur_radius > 0:
            softened = cv2.GaussianBlur(mask, (0, 0), blur_radius * 0.9)
            _threshold_value, mask = cv2.threshold(softened, 127, 255, cv2.THRESH_BINARY)

        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = abs(float(cv2.contourArea(contour)))
            if area < min_area_threshold:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter < 8.0:
                continue
            base_contour = cv2.convexHull(contour) if progress >= 0.72 else contour
            epsilon = perimeter * (0.018 + simplification_amount * (0.17 if photo_like else 0.14))
            approx = cv2.approxPolyDP(base_contour, epsilon, True)
            circularity = _contour_circularity(contour)
            drew_shape = False
            if progress >= 0.52 and len(contour) >= 16:
                ellipse = cv2.fitEllipse(contour)
                (center, size, angle) = ellipse
                width, height = size
                ellipse_area = float(np.pi * width * height * 0.25)
                area_delta = abs(ellipse_area - area) / area if area > 1.0 else 1.0
                aspect = max(width, height) / max(1.0, min(width, height))
                max_aspect = 3.7 + progress * 0.9
                max_area_delta = 0.46 + progress * 0.24
                if (
                    width >= 6.0
                    and height >= 6.0
                    and aspect <= max_aspect
                    and circularity >= (0.36 if photo_like else 0.40)
                    and area_delta <= max_area_delta
                ):
                    cv2.ellipse(output, (center, size, angle), color, cv2.FILLED, cv2.LINE_AA)
                    drew_shape = True
            if drew_shape:
                continue
            if len(approx) < 3:
                bounds = cv2.boundingRect(contour)
                cv2.rectangle(output, bounds, color, cv2.FILLED, cv2.LINE_AA)
                continue
            cv2.drawContours(output, [approx], 0, color, cv2.FILLED, cv2.LINE_AA)
    return output


def _color_limited_rgba_from_image(
    rgba: np.ndarray,
    *,
    color_count: int,
    grayscale: bool,
    shape_simplification: int,
    photo_like: bool,
    smart_enhance: bool,
    contrast: float,
    brightness: float,
) -> np.ndarray:
    rgb = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    limited_color_count = int(np.clip(color_count, 0, 15))
    prepared = rgb.copy()
    alpha = float(np.clip(contrast, 0.5, 3.0))
    beta = float(np.clip(brightness * 92.0, -36.0, 36.0))
    prepared = cv2.convertScaleAbs(prepared, alpha=alpha, beta=beta)
    if grayscale:
        gray = cv2.cvtColor(prepared, cv2.COLOR_RGB2GRAY)
        prepared = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    ui_level = float(np.clip(shape_simplification, 1.0, 10.0))
    effective_level = _effective_shape_simplification_level(ui_level)
    progress = _shape_simplification_progress(effective_level)
    simplification_amount = progress**1.10 * 0.45
    if limited_color_count > 0 and effective_level > 1.0:
        prepared = _blur_for_shape_simplification(prepared, effective_level, photo_like)

    if limited_color_count == 0:
        output = cv2.cvtColor(prepared, cv2.COLOR_RGB2RGBA)
        output[:, :, 3] = rgba[:, :, 3]
        return output

    if smart_enhance:
        flatten = float(np.clip(16.0 - float(limited_color_count) + simplification_amount * 5.0, 1.0, 20.0))
        prepared = cv2.bilateralFilter(
            prepared,
            9 if photo_like else 7,
            (28.0 if photo_like else 20.0) + flatten * (2.8 if photo_like else 1.9),
            (24.0 if photo_like else 16.0) + flatten * (2.1 if photo_like else 1.5),
        )
        gray = cv2.cvtColor(prepared, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(
            gray,
            42.0 if photo_like else 28.0,
            110.0 if photo_like else 78.0,
            apertureSize=3,
            L2gradient=True,
        )
        edge_darken = (
            float(np.clip(0.88 - (15 - limited_color_count) * 0.012, 0.72, 0.88))
            if photo_like
            else float(np.clip(0.92 - (15 - limited_color_count) * 0.010, 0.78, 0.92))
        )
        prepared = prepared.copy()
        prepared[edges > 0] = np.clip(prepared[edges > 0].astype(np.float32) * edge_darken, 0, 255).astype(np.uint8)

    training = prepared
    max_train_pixels = 90_000.0
    if training.shape[0] * training.shape[1] > max_train_pixels:
        scale = sqrt(max_train_pixels / float(training.shape[0] * training.shape[1]))
        training = cv2.resize(training, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    limited_color_count = min(limited_color_count, max(1, int(training.shape[0] * training.shape[1])))

    samples = training.reshape((-1, 3)).astype(np.float32)
    _compactness, _labels, centers = cv2.kmeans(
        samples,
        limited_color_count,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.8),
        3,
        cv2.KMEANS_PP_CENTERS,
    )

    quantized_rgb = _map_to_nearest_centers(prepared, centers)
    if effective_level > 1.0:
        quantized_rgb = _merge_small_color_regions(quantized_rgb, effective_level, photo_like)
        if progress >= 0.18:
            quantized_rgb = _geometrize_quantized_shapes(quantized_rgb, effective_level, photo_like)

    quantized_rgba = cv2.cvtColor(quantized_rgb, cv2.COLOR_RGB2RGBA)
    quantized_rgba[:, :, 3] = rgba[:, :, 3]
    return quantized_rgba


def _map_to_nearest_centers(prepared: np.ndarray, centers: np.ndarray) -> np.ndarray:
    centers = centers.astype(np.float32)
    flat = prepared.reshape((-1, 3)).astype(np.float32)
    labels = np.empty(flat.shape[0], dtype=np.int32)
    chunk_size = 262_144
    for start in range(0, flat.shape[0], chunk_size):
        chunk = flat[start : start + chunk_size]
        distances = np.sum((chunk[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels[start : start + chunk.shape[0]] = np.argmin(distances, axis=1)
    quantized = centers[labels].reshape(prepared.shape)
    return np.clip(quantized, 0, 255).astype(np.uint8)
