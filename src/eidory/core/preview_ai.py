"""Preview generation/editing, using the existing durable ComfyUI transport.

Candidates share the generated-reference store (project 0, origin preview), so
backup/migration includes them without creating a synthetic creative project.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import secrets
import shutil
import time
import uuid

import cv2
import numpy as np
from PIL import Image, ImageDraw

from eidory.core.comfy_reference import ACTIVE_STATES, ComfyClient, GeneratedReferences, local_url
from eidory.core.comfy_workflows import official_turbo_graph
from eidory.core.edit_tone import adjust_edit, tone_options, cached_analysis, analysis_signature, estimate_lightness
from eidory.core.image_loader import open_local_image, srgb_pixels
from eidory.core.image_upscale import ImageReplacement, UpscaleJobs, digest, source_info
from eidory.core.thumbnailer import Thumbnailer
from eidory.core.time_utils import utc_now_iso


def raster_mask(drawing, size, *, feather=0):
    w, h = size
    if min(size) < 1 or max(size) > 32768 or w*h > 64_000_000:
        raise ValueError('遮罩支持最多 6400 万像素的图片。')
    if not 0 <= feather <= 64 or len(drawing) > 4000:
        raise ValueError('遮罩参数无效。')
    mask = Image.new('L', size)
    painter = ImageDraw.Draw(mask)
    count = 0
    for stroke in drawing:
        points, diameter = stroke['points'], stroke['diameter']
        count += len(points)
        if (count > 100_000 or not points or not .001 <= diameter <= 2
                or any(not 0 <= p <= 1 for point in points for p in point)):
            raise ValueError('遮罩笔画无效或过多，请清空后重画。')
        radius = max(.5, diameter*min(size)/2)
        xy = [(x*(w-1), y*(h-1)) for x, y in points]
        color = 0 if stroke.get('erase') else 255
        if len(xy) > 1:
            painter.line(xy, fill=color, width=max(1, round(2*radius)), joint='curve')
        for x, y in xy:
            painter.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
    if feather:
        # Padding includes the image boundary. Distance is zero outside selected
        # pixels: feathering can never leak a model edit into unpainted areas.
        binary = np.pad(np.asarray(mask), 1)
        distance = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        mask = Image.fromarray(np.rint(np.minimum(distance/feather, 1)*255).astype('uint8'))
    return mask


class PreviewAIJobs(GeneratedReferences):
    def all(self, image_id=None, mode=None):
        return [j for j in super().all(0) if j.get('origin') == 'preview'
                and (image_id is None or j.get('image_id') == image_id)
                and (mode is None or j.get('mode') == mode)]

    def active(self):
        return next((j for j in reversed(self.all()) if j['state'] in ACTIVE_STATES), None)

    def create(self, image_id, mode, prompt, url, *, drawing=None, feather=8, width=1024, height=1024):
        if mode not in {'generate', 'edit'} or not prompt.strip():
            raise ValueError('请填写要生成的画面或修改要求。')
        if GeneratedReferences(self.store, self.paths).active() or UpscaleJobs(self.store, self.paths).active():
            raise ValueError('请等待当前生图、编辑或放大任务完成。')
        if min(width, height) < 256 or max(width, height) > 2048 or width % 32 or height % 32:
            raise ValueError('生图尺寸需为 256–2048 之间的 32 的倍数。')
        key, seed = str(uuid.uuid4()), secrets.randbelow(2**53)
        job = dict(id=key, prompt_id=key, client_id='eidory-'+key, project_id=0,
                   origin='preview', mode=mode, image_id=image_id,
                   prompt=prompt.strip(), seed=seed, url=local_url(url),
                   workflow_id='qwen21-official-turbo-'+mode, workflow_name='千问官方 Turbo · 8 步',
                   created_at=utc_now_iso(), submitted_at=time.time(), state='submitting', error='',
                   project_title='预览', node_title='AI 编辑' if mode == 'edit' else 'AI 生图',
                   output_node='8', requested_width=width, requested_height=height)
        self.save(job)
        try:
            graph = official_turbo_graph()
            graph['13']['inputs']['noise_seed'] = seed
            graph['4']['inputs']['prompt'] = prompt.strip()
            graph['5']['inputs'].update(width=width, height=height)
            if mode == 'edit':
                job['tone'] = tone_options(new=True)
                image = self.store.get_image(image_id)
                if not image or image.is_missing:
                    raise ValueError('原图不存在。')
                source = Path(image.file_path)
                info, sha = source_info(source), digest(source)
                folder = self.root / 'inputs' / key
                folder.mkdir(parents=True, exist_ok=True)
                backup = folder / ('original'+source.suffix)
                shutil.copy2(source, backup)
                if digest(backup) != sha or digest(source) != sha:
                    raise ValueError('准备期间原图发生变化，请重试。')
                with open_local_image(backup) as original:
                    pixels = srgb_pixels(original)
                    hard = raster_mask(drawing or [], pixels.size)
                    if not hard.getbbox():
                        raise ValueError('请先涂选需要修改的区域。')
                    pixels.save(folder / 'input.png')
                    hard.convert('RGB').save(folder / 'mask.png')
                    raster_mask(drawing, pixels.size, feather=feather).save(folder / 'blend.png')
                job.update(source_path=str(source), source_digest=sha, source_info=info,
                           original_path=str(backup), input_path=str(folder/'input.png'),
                           mask_path=str(folder/'mask.png'), blend_path=str(folder/'blend.png'),
                           drawing=deepcopy(drawing), feather=feather,
                           target_width=info['width'], target_height=info['height'])
                del graph['5']
                graph['9'] = {'class_type': 'LoadImage', 'inputs': {'image': ''}}
                graph['16'] = {'class_type': 'LoadImage', 'inputs': {'image': ''}}
                graph['4']['inputs'].update(vae=['3', 0], **{'images.image_1': ['9', 0], 'images.image_2': ['16', 0]})
                graph['4']['inputs']['prompt'] = ('<image1> 是原图。<image2> 是对齐的黑白区域遮罩，不是图片内容。'
                    '只在白色区域执行修改，黑色区域保持原样，不要画出遮罩。保持原有构图、光照和未指定的细节。修改要求：'+prompt.strip())
                graph['6']['inputs']['latent_image'] = ['4', 2]
            job['workflow'] = graph
            self.save(job)
        except Exception as exc:
            job.update(state='failed', error=str(exc))
            self.save(job)
            raise
        return job

    def submit(self, job_id):
        job = self.get(job_id)
        if job['state'] == 'submitting' and job['mode'] == 'edit':
            try:
                client = ComfyClient(job['url'])
                for node, field in [('9', 'input_path'), ('16', 'mask_path')]:
                    with open(job[field], 'rb') as stream:
                        uploaded = client.request('POST', '/upload/image',
                            files={'image': (job['id']+'-'+node+'.png', stream, 'image/png')},
                            data={'subfolder': 'EidoryEdit', 'type': 'input', 'overwrite': 'false'})
                    job['workflow'][node]['inputs']['image'] = '/'.join(
                        p for p in (uploaded.get('subfolder'), uploaded['name']) if p)
                self.save(job)
            except Exception as exc:
                job.update(state='failed', error='准备编辑图片失败：'+str(exc))
                self.save(job)
                return job
        return super().submit(job_id)

    def reconcile(self, job_id):
        job = super().reconcile(job_id)
        if job['state'] == 'ready' and job['mode'] == 'edit' and not job.get('composited'):
            try:
                job = self.composite(job)
            except (OSError, ValueError) as exc:
                job['error'] = '合成编辑结果失败，候选保留，可重试：'+str(exc)
                self.save(job)
                raise
        return job

    def composite(self, job, analysis=None):
        candidate = Path(job['file_path'])
        stage = candidate.with_suffix('.composite.png')
        # Retain the decoded output so a crash before the job commit cannot
        # composite an already blended image a second time.
        raw = candidate.with_suffix('.raw.png')
        if not raw.exists():
            if job.get('composited'):
                raise ValueError('原始 AI 结果缓存缺失，不能重复调整已合成的图片。')
            raw_stage = raw.with_suffix('.part')
            shutil.copyfile(candidate, raw_stage)
            os.replace(raw_stage, raw)
        pending = job.get('tone_pending')
        options = tone_options(pending['options'] if pending else job.get('tone'))
        with Image.open(job['input_path']) as original, Image.open(job['blend_path']) as mask, Image.open(raw) as result, Image.open(job['mask_path']) as hard:
            analysis = cached_analysis(job, analysis)
            if options['match'] and options['strength'] and not analysis:
                analysis = dict(signature=analysis_signature(job), report=estimate_lightness(original, result, hard))
            edited = result.convert('RGB').resize(original.size, Image.Resampling.LANCZOS)
            edited, report = adjust_edit(edited, original, hard, mask, options, analysis['report'] if analysis else None)
            combined = Image.composite(edited, original.convert('RGB'), mask)
            if 'A' in original.getbands():
                combined.putalpha(original.getchannel('A'))
            combined.save(stage, icc_profile=original.info.get('icc_profile'))
            combined.thumbnail((480, 480))
            combined.convert('RGB').save(job['thumbnail_path'], 'WEBP', quality=85)
        os.replace(stage, candidate)
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            current = json.loads(conn.execute('SELECT payload FROM generated_reference_jobs WHERE id=?', (job['id'],)).fetchone()[0])
            # A newer slider request can arrive while a previous render runs.
            # Keep that request durable rather than overwriting it with old settings.
            if current.get('tone_pending') == pending:
                current.pop('tone_pending', None)
            current.update(composited=True, width=job['target_width'], height=job['target_height'],
                           tone=options, tone_report=report, render_id=str(uuid.uuid4()), error='')
            if analysis:
                current['tone_analysis'] = analysis
            self.save(current, conn)
        return current

    def queue_tone(self, job_id, options):
        options = tone_options(options)
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('SELECT payload FROM generated_reference_jobs WHERE id=?', (job_id,)).fetchone()
            if not row:
                raise ValueError('编辑结果不存在。')
            job = json.loads(row[0])
            if job['mode'] != 'edit' or job['state'] != 'ready' or not job.get('composited'):
                raise ValueError('只能调整尚未采用的编辑结果。')
            if not Path(job['file_path']).with_suffix('.raw.png').is_file():
                raise ValueError('原始 AI 结果缓存缺失，无法无损重调。')
            if options != tone_options(job.get('tone')) or job.get('tone_pending'):
                job['tone_pending'] = dict(id=str(uuid.uuid4()), options=options)
                self.save(job, conn)
        return job

    def render_tone(self, job_id, analysis=None):
        job = self.get(job_id)
        if job['mode'] != 'edit' or job['state'] != 'ready':
            raise ValueError('当前结果已采用或放弃，未修改。')
        return self.composite(job, analysis) if job.get('tone_pending') else job

    def _settled(self, job_id):
        job = self.get(job_id)
        return self.render_tone(job_id) if job.get('tone_pending') else job

    def replace(self, job_id):
        job = self._settled(job_id)
        if job['mode'] != 'edit' or not job.get('composited'):
            raise ValueError('只有已完成的编辑结果可以替换原图。')
        return ImageReplacement(self.store, self.paths).replace(job)

    def adopt(self, job_id, directory, collection_id=None):
        job = self._settled(job_id)
        if job['state'] == 'adopted':
            return self.store.get_image(job['saved_image_id'])
        if job['state'] != 'ready' or (job['mode'] == 'edit' and not job.get('composited')):
            raise ValueError('请先等待结果完成。')
        directory = Path(directory).expanduser().resolve()
        if directory == self.root or self.root in directory.parents:
            raise ValueError('请选择候选缓存之外的永久保存位置。')
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / ('AI-'+job['mode']+'-'+job['id']+'.png')
        source = Path(job['file_path'])
        if target.exists() and digest(target) != digest(source):
            raise ValueError('目标存在不同内容的同名文件，未覆盖。')
        if not target.exists():
            stage = target.with_suffix('.part')
            shutil.copyfile(source, stage)
            os.replace(stage, target)
        sidecar = target.with_suffix('.json')
        sidecar.write_text(json.dumps({k: job[k] for k in ('prompt', 'seed', 'workflow', 'created_at', 'mode', 'tone') if k in job}, ensure_ascii=False, indent=2), encoding='utf-8')
        folder_id, now = self.store.add_folder(str(directory)), utc_now_iso()
        stat = target.stat()
        with Image.open(target) as image:
            width, height = image.size
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            image_id, _ = self.store._upsert_image_on_connection(conn, now=now, folder_id=folder_id,
                file_path=str(target), file_size=stat.st_size, width=width, height=height,
                created_time_ns=getattr(stat, 'st_birthtime_ns', None), modified_time_ns=stat.st_mtime_ns, duration_ms=None)
            if collection_id:
                conn.execute('INSERT OR IGNORE INTO image_collections(image_id,collection_id,created_at) VALUES(?,?,?)', (image_id, collection_id, now))
            conn.execute('UPDATE images SET note=? WHERE id=?', ('AI 生图' if job['mode']=='generate' else 'AI 编辑副本', image_id))
            job.update(state='adopted', saved_image_id=image_id, saved_path=str(target))
            self.save(job, conn)
        try:
            thumbnail = Thumbnailer(self.paths.thumbnail_dir).generate(image_id, str(target))
            self.store.update_thumbnail(image_id, str(thumbnail), 'ready')
        except (OSError, ValueError):
            pass
        return self.store.get_image(image_id)
