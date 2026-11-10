"""Local, non-accumulating lightness/chroma adjustment for masked AI edits."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def tone_options(value=None, *, new=False):
    # Legacy candidates retain their original rendering until explicitly changed.
    result = dict(match=new, strength=100, lightness=0, saturation=100)
    result.update(value or {})
    if not isinstance(result['match'], bool):
        raise ValueError('自动亮度匹配参数无效。')
    for key, low, high in [('strength', 0, 100), ('lightness', -50, 50), ('saturation', 0, 200)]:
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('明度、纯度或匹配强度超出支持范围。')
    return {key: result[key] for key in ('match', 'strength', 'lightness', 'saturation')}


def _lab(image):
    return cv2.cvtColor(np.asarray(image.convert('RGB'), dtype=np.float32)/255, cv2.COLOR_RGB2LAB)


def estimate_lightness(original, edited, hard_mask):
    """Estimate one L* offset from unpainted context, never the edited object.

    Prefer a band outside the selected area; fall back to the remaining scene
    if the band is too small. A full mask cannot supply a trustworthy reference.
    """
    scale = min(1., 1024/max(original.size))
    dimensions = tuple(max(1, round(n*scale)) for n in original.size)
    reference = original.resize(dimensions, Image.Resampling.LANCZOS)
    target = edited.resize(dimensions, Image.Resampling.LANCZOS)
    mask = np.asarray(hard_mask.convert('L').resize(dimensions, Image.Resampling.NEAREST)) > 0
    ref_l, target_l = _lab(reference)[:, :, 0], _lab(target)[:, :, 0]
    # White paper/walls are useful references for removing a gray patch. Their
    # known target appearance is white even if the original highlight clipped.
    eligible = ~mask & (ref_l > 3) & (target_l > 3)
    if 'A' in reference.getbands():
        eligible &= np.asarray(reference.getchannel('A')) >= 250
    radius = max(8, round(min(dimensions)*.03))
    band = cv2.dilate(mask.astype('uint8'), np.ones((radius*2+1, radius*2+1), dtype='uint8')).astype(bool)
    selected = eligible & band
    if np.count_nonzero(selected) < 64:
        selected = eligible
    values = (ref_l-target_l)[selected]
    values = values[np.abs(values) <= 35]
    if values.size < 64:
        return dict(offset=0., samples=int(values.size), limited=False,
                    reason='未找到足够的未涂选背景，自动匹配未生效；可手动调整明度。')
    center = float(np.median(values))
    deviation = float(np.median(np.abs(values-center)))
    inliers = values[np.abs(values-center) <= max(1.5, deviation*3)]
    offset = float(np.median(inliers))
    return dict(offset=float(np.clip(offset, -15, 15)), samples=int(inliers.size),
                limited=abs(offset) > 15, reason='')


def adjust_edit(edited, original, hard_mask, blend_mask, options, background_report=None):
    """Return an adjusted model output; the caller composites only painted pixels.

    L* changes perceptual lightness; scaling Lab a/b changes chroma while keeping
    its hue angle. Bounded strips avoid full-size float buffers for large sources.
    """
    options = tone_options(options)
    report = dict(offset=0., samples=0, limited=False, reason='')
    if options['match'] and options['strength']:
        report = background_report or estimate_lightness(original, edited, hard_mask)
    offset = report['offset']*options['strength']/100 + options['lightness']
    saturation = options['saturation']/100
    output = edited.convert('RGB').copy()
    if abs(offset) < 1e-5 and saturation == 1:
        return output, report
    bbox = blend_mask.getbbox()
    if not bbox:
        return output, report
    x0, y0, x1, y1 = bbox
    strip = max(1, min(256, 2_000_000//max(1, x1-x0)))
    for y in range(y0, y1, strip):
        rect = (x0, y, x1, min(y1, y+strip))
        lab = _lab(edited.crop(rect))
        lab[:, :, 0] = np.clip(lab[:, :, 0]+offset, 0, 100)
        lab[:, :, 1:] *= saturation
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        pixels = np.rint(np.clip(rgb, 0, 1)*255).astype('uint8')
        output.paste(Image.fromarray(pixels), rect[:2])
    return output, report


def analysis_signature(job):
    """Immutable inputs only: rendering the candidate must not bust this cache."""
    paths = [Path(job['input_path']), Path(job['file_path']).with_suffix('.raw.png'), Path(job['mask_path'])]
    return [2, *[[p.stat().st_size, p.stat().st_mtime_ns] for p in paths]]


def cached_analysis(job, offered=None):
    signature = analysis_signature(job)
    for entry in (offered, job.get('tone_analysis')):
        if isinstance(entry, dict) and entry.get('signature') == signature:
            report = entry.get('report', {})
            offset = report.get('offset')
            if isinstance(offset, (int, float)) and math.isfinite(offset) and abs(offset) <= 15 and 'reason' in report:
                return entry
    return None


class TonePreview:
    """One bounded, immutable screen-resolution cache; no I/O during dragging."""
    MAX_PIXELS = 1_000_000
    MAX_EDGE = 1920

    def __init__(self, original, edited, hard, blend, *, analysis=None):
        self.analysis = analysis or {'report': estimate_lightness(original, edited, hard)}
        scale = min(1., self.MAX_EDGE/max(original.size), math.sqrt(self.MAX_PIXELS/(original.width*original.height)))
        size = tuple(max(1, int(n*scale)) for n in original.size)
        self.original = original.resize(size, Image.Resampling.LANCZOS).convert('RGBA')
        edited = edited.resize(size, Image.Resampling.LANCZOS).convert('RGB')
        blend = blend.convert('L').resize(size, Image.Resampling.BILINEAR)
        # Downsampling the feather must not bleed beyond the downsampled selection.
        selected = np.asarray(hard.convert('L').resize(size, Image.Resampling.NEAREST)) > 0
        blend = Image.fromarray(np.where(selected, np.asarray(blend), 0).astype('uint8'))
        self.bbox = blend.getbbox()
        self.lab = None
        if self.bbox:
            self.raw = edited.crop(self.bbox)
            self.reference = self.original.crop(self.bbox).convert('RGB')
            self.alpha = self.original.getchannel('A').crop(self.bbox)
            self.blend = blend.crop(self.bbox)
            self.lab = _lab(self.raw)

    @classmethod
    def from_job(cls, job):
        raw = Path(job['file_path']).with_suffix('.raw.png')
        with Image.open(job['input_path']) as original, Image.open(raw) as edited, \
                Image.open(job['mask_path']) as hard, Image.open(job['blend_path']) as blend:
            analysis = cached_analysis(job) or dict(signature=analysis_signature(job), report=estimate_lightness(original, edited, hard))
            return cls(original, edited, hard, blend, analysis=analysis)

    @property
    def nbytes(self):
        total = self.original.width*self.original.height*4
        if self.lab is not None:
            total += self.lab.nbytes
            total += sum(im.width*im.height*len(im.getbands()) for im in (self.raw, self.reference, self.alpha, self.blend))
        return total

    def render(self, options):
        options = tone_options(options)
        output = self.original.copy()
        if self.lab is None:
            return output
        offset = options['lightness']
        if options['match']:
            offset += self.analysis['report']['offset']*options['strength']/100
        saturation = options['saturation']/100
        if abs(offset) < 1e-5 and saturation == 1:
            adjusted = self.raw
        else:
            lab = self.lab.copy()
            lab[:, :, 0] = np.clip(lab[:, :, 0]+offset, 0, 100)
            lab[:, :, 1:] *= saturation
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            adjusted = Image.fromarray(np.rint(np.clip(rgb, 0, 1)*255).astype('uint8'))
        patch = Image.composite(adjusted, self.reference, self.blend)
        patch.putalpha(self.alpha)
        output.paste(patch, self.bbox[:2])
        return output
