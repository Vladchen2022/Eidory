"""Durable serial queue, explicit exports and conservative candidate retention."""
from __future__ import annotations

import json
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageCms
import requests

from eidory.core.comfy_reference import ACTIVE_STATES, GeneratedReferences
from eidory.core.image_loader import open_local_image, oriented_pixels
from eidory.core.image_upscale import UpscaleJobs, ImageReplacement, _replacement_exif, output_size, processing_input_size
from eidory.core.time_utils import utc_now_iso


def batch_size_plan(width, height, factor, *, reduce_input=False):
    """Prefer the requested scale, then 2×, then a reduced copy at 2×.

    This policy belongs to batch enqueueing; the single-image controls keep
    their explicit scale and reversible input-reduction behavior.
    """
    if not all(isinstance(n, int) and n > 0 for n in (width, height)):
        raise ValueError('无法读取有效的原图尺寸，请先修复图片索引。')
    if not math.isfinite(factor) or not 1 < factor <= 8:
        raise ValueError('放大倍数必须大于 1 且不超过 8。')
    source = (width, height)
    inputs = processing_input_size(width, height, reduce_input=reduce_input)
    actual_factor = factor
    if max(inputs) * actual_factor > 4096:
        actual_factor = 2
        if max(inputs) > 2048:
            inputs = processing_input_size(width, height, reduce_input=True)
    output = output_size(*inputs, actual_factor)
    reduced = inputs != source
    if reduced:
        adjustment = (f'临时输入缩至 {inputs[0]} × {inputs[1]} 后放大 {actual_factor:g} 倍'
                      '；保留原图，输出可能小于原图')
    elif actual_factor != factor:
        adjustment = f'{factor:g} 倍超过输出长边 4096 上限，自动改为 {actual_factor:g} 倍'
    else:
        adjustment = ''
    return dict(requested_factor=factor, factor=actual_factor, reduce_input=reduced,
                size_plan=dict(source=list(source), input=list(inputs), output=list(output)),
                adjustment_reason=adjustment)


class UpscaleWorkflow(UpscaleJobs):
    def waiting(self):
        return sorted((j for j in self.all() if j['state'] == 'waiting'), key=lambda j: j['queue_order'])

    def enqueue(self, image_ids, factor, url, *, reduce_input=False, retry_of=None):
        accepted, skipped = [], []
        previous = self.get(retry_of) if retry_of else {}
        batch_id = previous.get('batch_id') or str(uuid.uuid4())
        batch_created_at = previous.get('batch_created_at') or utc_now_iso()
        busy = {j['image_id'] for j in self.all() if j['state'] in ACTIVE_STATES | {'waiting', 'paused'}}
        for image_id in dict.fromkeys(image_ids):
            image = self.store.get_image(image_id)
            if not image or image.is_missing or image_id in busy:
                skipped.append((image_id, '图片缺失，或已有待处理任务'))
                continue
            try:
                plan = batch_size_plan(image.width, image.height, factor, reduce_input=reduce_input)
            except (ValueError, TypeError) as exc:
                skipped.append((image_id, str(exc)))
                continue
            job = dict(id=str(uuid.uuid4()), image_id=image_id, file_name=image.file_name,
                       source_path=image.file_path, created_at=utc_now_iso(), state='waiting',
                       queued_source=[image.file_path, image.file_size, image.modified_time_ns],
                       queue_order=time.time_ns(), **plan,
                       url=url, retry_of=retry_of, error='', stage='等待本机处理',
                       batch_id=batch_id, batch_created_at=batch_created_at,
                       batch_index=previous.get('batch_index', len(accepted)))
            accepted.append(job)
            busy.add(image_id)
        # A selection is one durable batch; interruption cannot leave half of
        # its accepted tasks in the queue.
        with self.store.connect() as conn:
            for job in accepted:
                self.save(job, conn)
        return accepted, skipped

    def batches(self):
        groups = {}
        # all() returns newest first. A retry takes the original slot in its
        # batch while older attempts remain available in task history.
        for job in self.all():
            if batch_id := job.get('batch_id'):
                groups.setdefault(batch_id, {}).setdefault(job['batch_index'], job)
        result = []
        for batch_id, slots in groups.items():
            jobs = [slots[i] for i in sorted(slots)]
            pending = sum(j['state'] in ACTIVE_STATES | {'waiting', 'paused'} for j in jobs)
            result.append(dict(id=batch_id, created_at=jobs[0]['batch_created_at'], jobs=jobs,
                total=len(jobs), pending=pending, complete=not pending,
                ready=sum(j['state'] == 'ready' for j in jobs),
                reviewed=sum(j['state'] in {'applied', 'exported', 'discarded', 'restored'} for j in jobs),
                failed=sum(j['state'] in {'failed', 'missing'} for j in jobs),
                cancelled=sum(j['state'] == 'cancelled' for j in jobs)))
        return sorted(result, key=lambda b: (b['created_at'], b['id']), reverse=True)

    def batch(self, batch_id):
        return next((b for b in self.batches() if b['id'] == batch_id), None)

    def start_next(self):
        if self.store.get_setting('upscale.queue_paused', '0') == '1':
            return None
        # Paused means the server's state is still unknown, not permission to
        # launch a second memory-intensive operation.
        if self.active() or GeneratedReferences(self.store, self.paths).active():
            return None
        if any(j['state'] == 'paused' for j in self.all()):
            return None
        waiting = self.waiting()
        if not waiting:
            return None
        job = waiting[0]
        try:
            self.create(job['image_id'], job['factor'], job['url'],
                        reduce_input=job['reduce_input'], queue_id=job['id'])
        except ValueError as exc:
            if isinstance(exc.__cause__, requests.RequestException):
                self.store.set_setting('upscale.queue_paused', '1')
            raise
        return self.submit(job['id'])

    def cancel(self, job_id):
        job = self.get(job_id)
        if job['state'] == 'waiting':
            job.update(state='cancelled', error='', stage='排队已取消')
            self.save(job)
            return job
        return super().cancel(job_id)

    def move(self, job_id, delta):
        jobs = self.waiting()
        index = next((i for i, j in enumerate(jobs) if j['id'] == job_id), -1)
        target = index + delta
        if index < 0 or not 0 <= target < len(jobs):
            return
        a, b = jobs[index], jobs[target]
        a['queue_order'], b['queue_order'] = b['queue_order'], a['queue_order']
        with self.store.connect() as conn:
            self.save(a, conn)
            self.save(b, conn)

    def retry(self, job_id):
        job = self.get(job_id)
        if job['state'] not in {'failed', 'cancelled'}:
            raise ValueError('仅已失败或已确认取消的任务可以重试；状态未确认的任务请先继续核对。')
        accepted, skipped = self.enqueue([job['image_id']], job.get('requested_factor', job['factor']), job['url'],
                                        reduce_input=job.get('reduce_input', job.get('input_reduced', False)), retry_of=job_id)
        if not accepted:
            raise ValueError(skipped[0][1])
        return accepted[0]

    def export_copy(self, job_id, destination):
        job = self.get(job_id)
        if job['state'] not in {'ready', 'applied', 'exported'} or job.get('purged_at'):
            raise ValueError('此任务没有可保存的候选。')
        target = Path(destination).expanduser().absolute()
        if target.suffix.lower() != '.png':
            raise ValueError('新图片使用 PNG 格式，请选择 .png 文件名。')
        if target.resolve().is_relative_to(self.root.resolve()):
            raise ValueError('请选择图库文件夹，不能保存在临时候选目录中。')
        if target.exists() or target.is_symlink():
            raise ValueError('目标文件已存在，请换一个文件名。')
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = target.with_name('.eidory-export-' + uuid.uuid4().hex + '.part')
        try:
            with open_local_image(job['file_path']) as result, open_local_image(job['original_path']) as original:
                if result.size != (job['target_width'], job['target_height']):
                    raise ValueError('候选尺寸不正确，未保存。')
                result = result.convert('RGB')
                if 'A' in original.getbands() or 'transparency' in original.info:
                    alpha = oriented_pixels(original).convert('RGBA').getchannel('A')
                    result.putalpha(alpha.resize(result.size, Image.Resampling.LANCZOS))
                exif, _ = _replacement_exif(original, result.size)
                options = {'icc_profile': ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()}
                if exif:
                    options['exif'] = exif
                result.save(stage, 'PNG', **options)
            with stage.open('rb') as stream:
                os.fsync(stream.fileno())
            # Exclusive link avoids an overwrite if the destination appears
            # while encoding. Both paths are on the destination filesystem.
            os.link(stage, target)
        finally:
            stage.unlink(missing_ok=True)
        stat = target.stat()
        folder_id = self.store.add_folder(str(target.parent))
        with self.store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            image_id, _ = self.store._upsert_image_on_connection(conn, now=utc_now_iso(), folder_id=folder_id,
                file_path=str(target), file_size=stat.st_size, width=job['target_width'], height=job['target_height'],
                created_time_ns=getattr(stat, 'st_birthtime_ns', None), modified_time_ns=stat.st_mtime_ns, duration_ms=None)
            conn.execute('INSERT OR IGNORE INTO image_collections SELECT ?,collection_id,? FROM image_collections WHERE image_id=?',
                         (image_id, utc_now_iso(), job['image_id']))
            conn.execute('INSERT OR IGNORE INTO image_tags SELECT ?,tag_id,source,confirmed_by_user,? FROM image_tags WHERE image_id=?',
                         (image_id, utc_now_iso(), job['image_id']))
            conn.execute('UPDATE images SET note=? WHERE id=?', (f'AI 放大副本 · 原图：{job["source_path"]}', image_id))
            job.setdefault('copies', []).append({'image_id': image_id, 'path': str(target), 'created_at': utc_now_iso()})
            if job['state'] == 'ready':
                job['state'] = 'exported'
            self.save(job, conn)
        ImageReplacement(self.store, self.paths)._thumbnail(image_id)
        return self.store.get_image(image_id)

    def storage_report(self, days=30):
        """Only finished discarded/cancelled/failed assets can be purged.

        Replacement backups, live candidates and exported library files are
        never cleanup targets. Exact path whitelists prevent path traversal.
        """
        totals = dict(sources=0, candidates=0, replacements=0)
        for category in totals:
            directory = self.root / category
            if directory.exists():
                for root, dirs, files in os.walk(directory, followlinks=False):
                    dirs[:] = [d for d in dirs if not (Path(root)/d).is_symlink()]
                    for name in files:
                        path = Path(root)/name
                        if not path.is_symlink():
                            try:
                                totals[category] += path.stat().st_size
                            except FileNotFoundError:
                                pass
        cutoff = time.time() - days * 86400
        eligible = []
        for job in self.all():
            if job['state'] not in {'discarded', 'cancelled', 'failed'} or job.get('purged_at'):
                continue
            try:
                age = datetime.fromisoformat(job['created_at']).replace(tzinfo=timezone.utc).timestamp()
            except (KeyError, ValueError):
                continue
            if age > cutoff:
                continue
            files = self._cleanup_files(job)
            eligible.append(dict(id=job['id'], files=[str(p) for p in files], bytes=sum(p.stat().st_size for p in files)))
        return dict(totals=totals, eligible=eligible, bytes=sum(j['bytes'] for j in eligible), days=days)

    def _cleanup_files(self, job):
        key = job['id']
        try:
            if str(uuid.UUID(key)) != key:
                return []
        except ValueError:
            return []
        root = self.root.resolve()
        paths = []
        source = self.root / 'sources' / key
        if source.is_dir() and not source.is_symlink():
            paths += list(source.iterdir())
        paths += [self.root/'candidates'/'0'/(key + suffix) for suffix in ('.png', '.webp')]
        return [p for p in paths if p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root)
                and (p.parent == source or p.parent == self.root/'candidates'/'0')]

    def cleanup(self, report):
        freed = 0
        for entry in report['eligible']:
            job = self.get(entry['id'])
            if job['state'] not in {'discarded', 'cancelled', 'failed'}:
                continue
            allowed = {str(p) for p in self._cleanup_files(job)}
            # Persist loss of undo before removing anything, including a crash
            # halfway through. Cleanup never removes the task's audit record.
            job['purged_at'] = utc_now_iso()
            self.save(job)
            for name in entry['files']:
                if name in allowed:
                    path = Path(name)
                    size = path.stat().st_size
                    path.unlink()
                    freed += size
        return freed
