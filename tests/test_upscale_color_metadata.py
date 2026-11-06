"""Regressions for GRAY ICC input and undefined camera EXIF rationals."""
import io
from pathlib import Path
import struct
from unittest.mock import patch

from PIL import Image, ImageChops, ImageCms, ImageOps
from PIL.TiffImagePlugin import IFDRational
from PySide6.QtTest import QTest
import pytest

from test_filter_workflow import app, library, window
from test_image_upscale import annotate, annotations
from eidory.core.comfy_reference import ComfyClient
from eidory.core.image_upscale import ImageReplacement, UpscaleJobs, digest, source_info
from eidory.core.image_loader import oriented_pixels
from eidory.core.scanner import ImageScanner
from eidory.core.thumbnailer import Thumbnailer


GRAY_PROFILE = (Path(__file__).parent / 'fixtures/gray-dot-gain20.icc').read_bytes()


def malformed_exif(orientation):
    exif = Image.Exif()
    exif.endian = '<'
    exif[271] = 'Regression camera'
    exif[274] = orientation
    exif[256], exif[257] = 80, 60
    exif[34665] = {
        37377: IFDRational(1), 37380: IFDRational(0), 37386: IFDRational(50),
        36867: '2012:10:31 16:00:24', 40962: 80, 40963: 60,
        40965: {1: 'R98'},
    }
    exif[34853] = {1: 'N', 2: (IFDRational(30), IFDRational(15), IFDRational(0))}
    # Reproduce the real file's -32768/0 shutter speed and 0/0 exposure bias.
    # Write a valid directory first, then patch its signed-rational payloads.
    raw = bytearray(exif.tobytes())
    root = struct.unpack_from('<I', raw, 10)[0] + 6
    for entry in range(struct.unpack_from('<H', raw, root)[0]):
        tag, _, _, offset = struct.unpack_from('<HHII', raw, root + 2 + entry * 12)
        if tag == 34665:
            sub = offset + 6
            break
    for entry in range(struct.unpack_from('<H', raw, sub)[0]):
        position = sub + 2 + entry * 12
        tag, _, _, offset = struct.unpack_from('<HHII', raw, position)
        if tag in (37377, 37380):
            struct.pack_into('<H', raw, position + 2, 10)  # SRATIONAL
            struct.pack_into('<ii', raw, offset + 6, -32768 if tag == 37377 else 0, 0)
    return bytes(raw)


def import_job(library, source):
    paths, store, _ = library
    iid = ImageScanner(store, Thumbnailer(paths.thumbnail_dir)).import_files([str(source)]).image_ids[0]
    image = store.get_image(iid)
    repo = UpscaleJobs(store, paths)
    with patch.object(ComfyClient, 'request', return_value={'name': 'input.png'}) as upload:
        job = repo.create(iid, 2, 'http://127.0.0.1:8188')
    assert upload.call_count == 1
    with Image.open(job['input_path']) as model_input:
        candidate = model_input.resize((job['target_width'], job['target_height']))
        candidate.save(source.parent / 'candidate.png')
    job.update(state='ready', file_path=str(source.parent / 'candidate.png'))
    repo.save(job)
    return repo, image, job


@pytest.mark.parametrize('fmt,alpha', [('JPEG', False), ('PNG', False), ('PNG', True), ('WEBP', False)])
def test_gray_icc_input_replace_and_full_restore(library, tmp_path, fmt, alpha):
    source = tmp_path / ('gray.' + {'JPEG': 'jpg', 'PNG': 'png', 'WEBP': 'webp'}[fmt])
    pixels = Image.linear_gradient('L').resize((80, 60))
    if alpha:
        pixels.putalpha(96)
    pixels.save(source, fmt, icc_profile=GRAY_PROFILE)
    before = digest(source)
    repo, image, job = import_job(library, source)
    with Image.open(source) as original, Image.open(job['input_path']) as actual:
        expected = ImageCms.profileToProfile(original.convert('L'),
            ImageCms.ImageCmsProfile(io.BytesIO(GRAY_PROFILE)), ImageCms.createProfile('sRGB'), outputMode='RGB')
        assert ImageChops.difference(actual.convert('RGB'), expected).getbbox() is None
        assert ImageCms.ImageCmsProfile(io.BytesIO(actual.info['icc_profile'])).profile.xcolor_space.strip() == 'RGB'
        assert 'exif' not in actual.info
        if alpha:
            assert actual.getchannel('A').getextrema() == (96, 96)
    assert digest(source) == before
    replacement = ImageReplacement(repo.store, repo.paths)
    replacement.replace(job)
    with Image.open(source) as actual:
        assert actual.size == (160, 120)
        assert actual.format == fmt
        assert actual.mode == ('RGB' if fmt == 'WEBP' else 'LA' if alpha else 'L')
        if fmt != 'WEBP':
            assert actual.info['icc_profile'] == GRAY_PROFILE
            # Verify that the saved pixels and embedded profile form a usable transform.
            ImageCms.profileToProfile(actual.convert('L'), ImageCms.ImageCmsProfile(io.BytesIO(GRAY_PROFILE)),
                                      ImageCms.createProfile('sRGB'), outputMode='RGB')
        else:
            assert ImageCms.ImageCmsProfile(io.BytesIO(actual.info['icc_profile'])).profile.xcolor_space.strip() == 'RGB'
        if alpha:
            assert actual.getchannel('A').getextrema() == (96, 96)
    replacement.restore(image.id)
    assert digest(source) == before


@pytest.mark.parametrize('fmt', ['JPEG', 'PNG', 'WEBP'])
@pytest.mark.parametrize('orientation', [1, 6])
def test_undefined_exif_rationals_do_not_block_prepare_replace_or_restore(library, tmp_path, fmt, orientation):
    source = tmp_path / ('camera.' + {'JPEG': 'jpg', 'PNG': 'png', 'WEBP': 'webp'}[fmt])
    pixels = Image.new('RGB', (80, 60), 'teal')
    pixels.save(source, fmt, exif=malformed_exif(orientation))
    before = digest(source)
    with Image.open(source) as original:
        assert original.getexif().get_ifd(34665)[37377].denominator == 0
        with pytest.raises(struct.error):
            original.getexif().tobytes()
    repo, image, job = import_job(library, source)
    annotate(repo.store, image)
    semantics = annotations(repo.store, image.id)
    replacement = ImageReplacement(repo.store, repo.paths)
    replacement.replace(job)
    with Image.open(source) as actual:
        exif = actual.getexif()
        details = exif.get_ifd(34665)
        assert (37377 not in details) and (37380 not in details)
        assert details[37386] == 50
        assert details[36867] == '2012:10:31 16:00:24'
        assert exif[271] == 'Regression camera'
        assert exif[274] == 1
        assert (exif[256], exif[257]) == actual.size
        assert (details[40962], details[40963]) == actual.size
        assert exif.get_ifd(34853)[2] == (30, 15, 0)
        assert exif.get_ifd(40965)[1] == 'R98'
        assert actual.size == ((120, 160) if orientation == 6 else (160, 120))
    assert set(replacement.latest(image.id)['metadata_omissions']) == {'IFD0/34665/37377', 'IFD0/34665/37380'}
    assert annotations(repo.store, image.id) == semantics
    replacement.restore(image.id)
    assert digest(source) == before
    assert annotations(repo.store, image.id) == semantics
    info = source_info(source)
    assert (repo.store.get_image(image.id).width, repo.store.get_image(image.id).height) == (info['width'], info['height'])
    assert repo.store.get_image(image.id).thumbnail_status == 'ready'


def test_replacement_reports_skipped_camera_parameters_in_panel(window, app, library, tmp_path):
    source = tmp_path / 'camera.jpg'
    Image.new('RGB', (80, 60), 'teal').save(source, exif=malformed_exif(1))
    repo, image, job = import_job(library, source)
    window.grid_view.set_images([image])
    window._open_image_preview(image)
    preview = next(iter(window._preview_dialogs))
    preview.upscale_button.click()
    QTest.qWait(30)
    panel = window.image_upscaler.panel
    assert panel.replace.isEnabled()
    panel.replace.click()
    QTest.qWait(30)
    assert '已跳过 2 项无效拍摄参数' in panel.status.text()
    assert panel.restore.isVisible()
    assert repo.get(job['id'])['state'] == 'applied'
    panel.restore.click()
    QTest.qWait(30)
    assert '已跳过' not in panel.status.text()
    assert digest(source) == job['source_digest']


def test_valid_signed_exif_rationals_are_retained(library, tmp_path):
    source = tmp_path / 'valid.jpg'
    exif = Image.Exif()
    exif[34665] = {37377: IFDRational(-1, 2), 37380: IFDRational(-1, 3)}
    Image.new('RGB', (80, 60), 'teal').save(source, exif=exif)
    repo, image, job = import_job(library, source)
    replacement = ImageReplacement(repo.store, repo.paths)
    replacement.replace(job)
    with Image.open(source) as actual:
        details = actual.getexif().get_ifd(34665)
        assert float(details[37377]) == -.5
        assert float(details[37380]) == pytest.approx(-1/3)
    assert not replacement.latest(image.id)['metadata_omissions']


@pytest.mark.parametrize('fmt', ['JPEG', 'PNG', 'WEBP'])
def test_rgb_profile_round_trip_remains_usable(library, tmp_path, fmt):
    source = tmp_path / ('rgb.' + {'JPEG': 'jpg', 'PNG': 'png', 'WEBP': 'webp'}[fmt])
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    Image.new('RGBA' if fmt == 'PNG' else 'RGB', (80, 60), 'teal').save(source, fmt, icc_profile=profile)
    before = digest(source)
    repo, image, job = import_job(library, source)
    replacement = ImageReplacement(repo.store, repo.paths)
    replacement.replace(job)
    with Image.open(source) as actual:
        assert actual.info['icc_profile'] == profile
        assert actual.mode == ('RGBA' if fmt == 'PNG' else 'RGB')
    replacement.restore(image.id)
    assert digest(source) == before


@pytest.mark.parametrize('orientation', range(1, 9))
def test_pixel_orientation_matches_pillow_without_carrying_source_metadata(orientation):
    image = Image.linear_gradient('L').resize((80, 60))
    exif = Image.Exif()
    exif[274] = orientation
    exif[271] = 'Camera metadata stays in source'
    image.info['exif'] = exif.tobytes()
    actual = oriented_pixels(image)
    expected = ImageOps.exif_transpose(image)
    assert actual.size == expected.size
    assert ImageChops.difference(actual, expected).getbbox() is None
    assert not actual.info
    assert image.getexif()[274] == orientation
