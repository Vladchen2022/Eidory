"""SeedVR2 candidates and crash-recoverable, identity-preserving media replacement."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid

from PIL import Image, ImageCms, ImageOps
from PIL.PngImagePlugin import PngInfo

from eidory.core.comfy_reference import ACTIVE_STATES, ComfyClient, GeneratedReferences, local_url
from eidory.core.image_loader import open_local_image
from eidory.core.thumbnailer import Thumbnailer
from eidory.core.time_utils import utc_now_iso, timestamp_ns_to_iso

FORMATS = {'JPEG', 'PNG', 'WEBP'}


def digest(path):
    with open(path, 'rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def output_size(width, height, factor):
    if not math.isfinite(factor) or factor <= 1 or factor > 8:
        raise ValueError('目标尺寸必须大于原图，放大倍数最多 8 倍。')
    if max(width, height) * factor > 4096:
        raise ValueError('输出长边最多 4096 像素，请降低倍数或指定长边。')
    size = tuple(round(n * factor) // 2 * 2 for n in (width, height))
    if min(size) < 2 or max(size) <= max(width, height):
        raise ValueError('按偶数像素对齐后，目标尺寸必须大于原图。')
    return size


def source_info(path):
    if Path(path).is_symlink():
        raise ValueError('请使用源文件，暂不替换符号链接图片。')
    with open_local_image(path) as image:
        if image.format not in FORMATS or getattr(image, 'n_frames', 1) != 1:
            raise ValueError('目前支持单帧 JPG、PNG 和 WebP 图片。')
        w, h = image.size
        if image.getexif().get(274, 1) in (5, 6, 7, 8):
            w, h = h, w
        return {'width': w, 'height': h, 'format': image.format}


def seedvr_graph(input_name, factor, seed):
    def n(kind, **inputs):
        return {'class_type': kind, 'inputs': inputs}
    return {
        '1': n('LoadImage', image=input_name),
        '2': n('JoinImageWithAlpha', image=['1', 0], alpha=['1', 1]),
        '3': n('ResizeImageMaskNode', input=['2', 0], resize_type='scale by multiplier',
               **{'resize_type.multiplier': factor, 'scale_method': 'lanczos'}),
        '4': n('SeedVR2Preprocess', resized_images=['3', 0]),
        '5': n('UNETLoader', unet_name='seedvr2_7b_fp16.safetensors', weight_dtype='default'),
        '6': n('VAELoader', vae_name='seedvr2_ema_vae_fp16.safetensors'),
        '7': n('VAEEncodeTiled', pixels=['4', 0], vae=['6', 0], tile_size=512, overlap=128,
               temporal_size=4096, temporal_overlap=8),
        '8': n('SeedVR2Conditioning', model=['5', 0], vae_conditioning=['7', 0]),
        '9': n('KSampler', model=['5', 0], positive=['8', 0], negative=['8', 1], latent_image=['7', 0],
               seed=seed, steps=1, cfg=1.0, sampler_name='euler', scheduler='simple', denoise=1.0),
        '10': n('VAEDecodeTiled', samples=['9', 0], vae=['6', 0], tile_size=512, overlap=128,
                temporal_size=4096, temporal_overlap=8),
        '11': n('SeedVR2PostProcessing', images=['10', 0], original_resized_images=['3', 0], color_correction_method='lab'),
        '12': n('PreviewImage', images=['11', 0]),
    }


class UpscaleJobs(GeneratedReferences):
    def __init__(self, store, paths):
        self.store, self.paths = store, paths
        self.root = paths.data_dir / 'image-upscale'

    def all(self, image_id=None):
        sql, args = "SELECT json_remove(payload, '$.workflow') FROM image_upscale_jobs", ()
        if image_id is not None:
            sql, args = sql + ' WHERE image_id=?', (image_id,)
        with self.store.connect(readonly=True) as conn:
            return [json.loads(r[0]) for r in conn.execute(sql + ' ORDER BY created_at DESC, rowid DESC', args)]

    def active(self):
        with self.store.connect(readonly=True) as conn:
            row = conn.execute("SELECT payload FROM image_upscale_jobs WHERE json_extract(payload,'$.state') IN ('submitting','unknown','queued','running','downloading','offline') ORDER BY created_at LIMIT 1").fetchone()
        return json.loads(row[0]) if row else None

    def get(self, job_id):
        with self.store.connect(readonly=True) as conn:
            row = conn.execute('SELECT payload FROM image_upscale_jobs WHERE id=?', (job_id,)).fetchone()
        if not row:
            raise ValueError('放大任务不存在。')
        return json.loads(row[0])

    def save(self, job, conn=None):
        sql = '''INSERT INTO image_upscale_jobs(id,image_id,created_at,payload) VALUES(?,?,?,?)
                 ON CONFLICT(id) DO UPDATE SET payload=excluded.payload'''
        values = (job['id'], job['image_id'], job['created_at'], json.dumps(job, ensure_ascii=False))
        if conn is not None:
            conn.execute(sql, values)
        else:
            with self.store.connect() as conn:
                conn.execute(sql, values)

    def create(self, image_id, factor, url):
        if self.active() or GeneratedReferences(self.store, self.paths).active():
            raise ValueError('请等待当前生图或放大任务完成。')
        image = self.store.get_image(image_id)
        if not image or image.is_missing:
            raise ValueError('原图已不在图库中。')
        source = Path(image.file_path)
        info = source_info(source)
        width, height = output_size(info['width'], info['height'], factor)
        key = str(uuid.uuid4())
        directory = self.root / 'sources' / key
        directory.mkdir(parents=True, exist_ok=True)
        backup = directory / ('original' + source.suffix)
        shutil.copy2(source, backup)
        source_digest = digest(backup)
        if digest(source) != source_digest:
            raise ValueError('读取期间原图发生变化，请重试。')
        snapshot = directory / 'input.png'
        with open_local_image(backup) as original:
            decoded = ImageOps.exif_transpose(original)
            alpha = decoded.convert('RGBA').getchannel('A') if 'A' in decoded.getbands() or 'transparency' in decoded.info else None
            rgb = decoded.convert('RGB')
            profile = original.info.get('icc_profile')
            if profile:
                rgb = ImageCms.profileToProfile(rgb, ImageCms.ImageCmsProfile(io.BytesIO(profile)), ImageCms.createProfile('sRGB'), outputMode='RGB')
            if alpha is not None:
                rgb.putalpha(alpha)
            rgb.save(snapshot, 'PNG')
        job = dict(id=key, prompt_id=key, client_id='eidory-upscale-' + key, image_id=image_id,
                   project_id=0, source_path=str(source), source_digest=source_digest,
                   original_path=str(backup), input_path=str(snapshot), source_info=info,
                   target_width=width, target_height=height, factor=factor, seed=2026092701,
                   created_at=utc_now_iso(), submitted_at=time.time(), state='submitting', error='',
                   output_node='12', url=local_url(url))
        # Input upload is idempotent; the prompt identity is durable before POST /prompt.
        client = ComfyClient(job['url'])
        with snapshot.open('rb') as stream:
            uploaded = client.request('POST', '/upload/image', files={'image': (key + '.png', stream, 'image/png')},
                                      data={'type': 'input', 'subfolder': 'EidoryUpscale', 'overwrite': 'true'})
        name = '/'.join(filter(None, [uploaded.get('subfolder'), uploaded['name']]))
        job['workflow'] = seedvr_graph(name, factor, job['seed'])
        self.save(job)
        return job

    def reconcile(self, job_id):
        job = super().reconcile(job_id)
        if job['state'] == 'ready' and (job['width'], job['height']) != (job['target_width'], job['target_height']):
            job.update(state='failed', error='服务返回的尺寸与目标不符，结果未替换原图。')
            self.save(job)
        return job

    def discard(self, job_id, *, undo=False):
        job = self.get(job_id)
        if job['state'] != ('discarded' if undo else 'ready'):
            raise ValueError('只能丢弃尚未替换的结果。')
        # A recoverable discard, like reference candidates; never delete the source.
        job.update(state='ready' if undo else 'discarded', error='')
        self.save(job)
        return job


class ImageReplacement:
    def __init__(self, store, paths):
        self.store, self.paths = store, paths
        self.root = paths.data_dir / 'image-upscale' / 'replacements'

    def latest(self, image_id):
        with self.store.connect(readonly=True) as conn:
            row = conn.execute("SELECT payload FROM image_replacements WHERE image_id=? AND state='applied' ORDER BY rowid DESC LIMIT 1", (image_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def _save(self, record, conn):
        conn.execute('''INSERT INTO image_replacements(id,image_id,state,created_at,payload) VALUES(?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET state=excluded.state,payload=excluded.payload''',
            (record['id'], record['image_id'], record['state'], record['created_at'], json.dumps(record, ensure_ascii=False)))

    def replace(self, job):
        if job['state'] != 'ready':
            raise ValueError('放大结果尚未准备好。')
        image = self.store.get_image(job['image_id'])
        if not image or image.file_path != job['source_path']:
            raise ValueError('原图已移动或移出图库，未进行替换。')
        source = Path(image.file_path)
        if source.is_symlink() or digest(source) != job['source_digest']:
            raise ValueError('原图已在外部发生变化，未覆盖；放大结果仍保留。')
        key = str(uuid.uuid4())
        directory = self.root / key
        directory.mkdir(parents=True, exist_ok=True)
        prepared = directory / ('replacement' + source.suffix)
        with open_local_image(source) as original, open_local_image(job['file_path']) as result:
            if result.size != (job['target_width'], job['target_height']):
                raise ValueError('结果尺寸不正确。')
            result = result.convert('RGBA' if 'A' in result.getbands() else 'RGB')
            # Keep transparency from the actual source, including palette transparency.
            if 'A' in original.getbands() or 'transparency' in original.info:
                alpha = ImageOps.exif_transpose(original).convert('RGBA').getchannel('A')
                result.putalpha(alpha.resize(result.size, Image.Resampling.LANCZOS))
            fmt = job['source_info']['format']
            options = {}
            profile = original.info.get('icc_profile')
            if profile:
                alpha = result.getchannel('A') if 'A' in result.getbands() else None
                result = ImageCms.profileToProfile(result.convert('RGB'), ImageCms.createProfile('sRGB'),
                    ImageCms.ImageCmsProfile(io.BytesIO(profile)), outputMode='RGB')
                if alpha is not None:
                    result.putalpha(alpha)
                options['icc_profile'] = profile
            exif = original.getexif()
            if exif and fmt in {'JPEG', 'PNG', 'WEBP', 'TIFF'}:
                exif[274] = 1
                for field, value in ((256, result.width), (257, result.height), (40962, result.width), (40963, result.height)):
                    if field in exif:
                        exif[field] = value
                options['exif'] = exif.tobytes()
            if fmt == 'JPEG':
                result = result.convert('RGB')
                options.update(quality=98, subsampling=0)
            elif fmt == 'WEBP':
                options['lossless'] = True
                if original.info.get('xmp'):
                    options['xmp'] = original.info['xmp']
            if fmt == 'PNG':
                text = PngInfo()
                for name, value in original.info.items():
                    if isinstance(value, str):
                        text.add_text(name, value)
                options['pnginfo'] = text
            if original.info.get('dpi'):
                options['dpi'] = original.info['dpi']
            result.save(prepared, fmt, **options)
        return self._apply(image.id, prepared, job['source_digest'], key=key, job_id=job['id'])

    def restore(self, image_id):
        previous = self.latest(image_id)
        if not previous:
            raise ValueError('没有可恢复的原图。')
        return self._apply(image_id, Path(previous['backup']), previous['after_digest'], restore_of=previous['id'])

    def _apply(self, image_id, candidate, expected_digest, *, key=None, job_id=None, restore_of=None):
        image = self.store.get_image(image_id)
        if image is None:
            raise ValueError('图片已移出图库。')
        source = Path(image.file_path)
        if source.is_symlink() or digest(source) != expected_digest:
            raise ValueError('原图已在外部发生变化，操作已停止。')
        key = key or str(uuid.uuid4())
        directory = self.root / key
        directory.mkdir(parents=True, exist_ok=True)
        backup = directory / ('original' + source.suffix)
        shutil.copy2(source, backup)
        with backup.open('rb') as f:
            os.fsync(f.fileno())
        if digest(backup) != expected_digest:
            raise ValueError('备份期间原图发生变化，操作已停止。')
        # Stage on the destination filesystem; rename must be atomic, even across volumes.
        stage = source.with_name('.eidory-upscale-' + key + '.part')
        record = dict(id=key, image_id=image_id, source=str(source), backup=str(backup), stage=str(stage),
                      before_digest=expected_digest, after_digest=digest(candidate), state='prepared',
                      created_at=utc_now_iso(), job_id=job_id, restore_of=restore_of)
        try:
            shutil.copyfile(candidate, stage)
            shutil.copymode(source, stage)
            with stage.open('rb') as f:
                os.fsync(f.fileno())
            with open_local_image(stage) as im:
                record['width'], record['height'] = ImageOps.exif_transpose(im).size
            if restore_of:
                # Restoring original bytes also restores the original file timestamp.
                shutil.copystat(candidate, stage)
            with self.store.connect() as conn:
                conn.execute('PRAGMA synchronous=FULL')
                self._save(record, conn)  # Durable recovery intent, before touching the source.
            with self.store.connect() as conn:
                conn.execute('PRAGMA synchronous=FULL')
                conn.execute('BEGIN IMMEDIATE')
                current = conn.execute('SELECT file_path FROM images WHERE id=?', (image_id,)).fetchone()
                if not current or current[0] != str(source) or digest(source) != expected_digest:
                    raise ValueError('原图已改变，未替换。')
                processing = conn.execute("SELECT 1 FROM embeddings WHERE image_id=? AND status='processing' UNION ALL SELECT 1 FROM ai_vision_tags WHERE image_id=? AND status='processing' LIMIT 1", (image_id, image_id)).fetchone()
                if processing:
                    raise ValueError('此图正在建立 AI 索引，请等索引完成后再替换。')
                os.replace(stage, source)
                self._finish(record, conn)
        except Exception:
            # If rename succeeded but the DB commit did not, recovery finishes the recorded
            # update. If no rename happened the candidate and original both remain intact.
            self.recover()
            raise
        finally:
            stage.unlink(missing_ok=True)
        self._thumbnail(image_id)
        return self.store.get_image(image_id)

    def _finish(self, record, conn):
        source = Path(record['source'])
        stat = source.stat()
        conn.execute('''UPDATE images SET file_size=?,width=?,height=?,modified_time_ns=?,modified_at=?,
                        thumbnail_status='pending',thumbnail_path=NULL,is_missing=0 WHERE id=?''',
                     (stat.st_size, record['width'], record['height'], stat.st_mtime_ns,
                      timestamp_ns_to_iso(stat.st_mtime_ns), record['image_id']))
        conn.execute('DELETE FROM image_hashes WHERE image_id=?', (record['image_id'],))
        # Semantic vectors, AI/user tags, notes, links and board layouts deliberately untouched.
        record['state'] = 'restored' if record['restore_of'] else 'applied'
        self._save(record, conn)
        if record['restore_of']:
            row = conn.execute('SELECT payload FROM image_replacements WHERE id=?', (record['restore_of'],)).fetchone()
            previous = json.loads(row[0])
            previous['state'] = 'restored'
            self._save(previous, conn)
            if previous.get('job_id'):
                row = conn.execute('SELECT payload FROM image_upscale_jobs WHERE id=?', (previous['job_id'],)).fetchone()
                if row:
                    job = json.loads(row[0])
                    job.update(state='restored', error='')
                    UpscaleJobs(self.store, self.paths).save(job, conn)
        if record['job_id']:
            row = conn.execute('SELECT payload FROM image_upscale_jobs WHERE id=?', (record['job_id'],)).fetchone()
            if row:
                job = json.loads(row[0])
                job.update(state='applied', error='')
                UpscaleJobs(self.store, self.paths).save(job, conn)

    def _thumbnail(self, image_id):
        image = self.store.get_image(image_id)
        if image:
            try:
                thumb = Thumbnailer(self.paths.thumbnail_dir).generate(image_id, image.file_path)
                self.store.update_thumbnail(image_id, str(thumb), 'ready')
            except (OSError, ValueError):
                pass  # Pending thumbnail can be rebuilt without losing the replacement.

    def recover(self):
        """Run before scanners/workers start; resolves a crash between rename and DB commit."""
        with self.store.connect(readonly=True) as conn:
            pending = [json.loads(r[0]) for r in conn.execute("SELECT payload FROM image_replacements WHERE state='prepared'")]
        for record in pending:
            source = Path(record['source'])
            with self.store.connect() as conn:
                conn.execute('BEGIN IMMEDIATE')
                actual = digest(source) if source.is_file() else None
                row = conn.execute('SELECT file_path FROM images WHERE id=?', (record['image_id'],)).fetchone()
                if row and row[0] == str(source) and actual == record['after_digest']:
                    self._finish(record, conn)
                else:
                    record['state'] = 'aborted' if actual == record['before_digest'] else 'conflict'
                    self._save(record, conn)
            Path(record['stage']).unlink(missing_ok=True)
            if record['state'] in {'applied', 'restored'}:
                self._thumbnail(record['image_id'])
