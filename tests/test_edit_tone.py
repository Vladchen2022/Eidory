from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from test_filter_workflow import app, library, window
from test_preview_ai import suspend_qt_wrapper_cycle_collection
from eidory.core.comfy_reference import ComfyClient
from eidory.core.edit_tone import adjust_edit, estimate_lightness, tone_options
from eidory.core.image_upscale import ImageReplacement, digest
from eidory.core.preview_ai import PreviewAIJobs, raster_mask

DRAWING = [dict(points=[[.5,.5]], diameter=.5)]


def tone_job(library, *, color=(140,140,140)):
    paths, store, images = library
    image = images[0]
    Image.new('RGB',(120,80),(160,160,160)).save(image.file_path)
    from eidory.core.scanner import ImageScanner
    from eidory.core.thumbnailer import Thumbnailer
    ImageScanner(store,Thumbnailer(paths.thumbnail_dir)).scan_folder(str(Path(image.file_path).parent))
    image = store.get_image(image.id)
    repo = PreviewAIJobs(store, paths)
    job = repo.create(image.id,'edit','remove text','http://127.0.0.1:8188',drawing=DRAWING,feather=4)
    target = repo.root/'candidates'/'0'/(job['id']+'.png')
    target.parent.mkdir(parents=True,exist_ok=True)
    Image.new('RGB',(120,80),color).save(target)
    job.update(state='ready',file_path=str(target),thumbnail_path=str(target.with_suffix('.webp')),tone=tone_options())
    repo.save(job)
    return repo, repo.composite(job), image


def lab_l(image):
    return cv2.cvtColor(np.array(image.convert('RGB'),dtype=np.float32)/255,cv2.COLOR_RGB2LAB)[:,:,0]


def test_automatic_match_corrects_dark_patch_without_changing_unmasked_pixels(library):
    repo, job, image = tone_job(library)
    source_sha, raw_sha = digest(image.file_path), digest(Path(job['file_path']).with_suffix('.raw.png'))
    before = np.array(Image.open(job['file_path']))
    repo.queue_tone(job['id'],tone_options(new=True))
    with patch.object(ComfyClient,'request',side_effect=AssertionError('must not invoke ComfyUI')):
        job = repo.render_tone(job['id'])
    result = Image.open(job['file_path'])
    assert abs(float(lab_l(result)[40,60])-float(lab_l(Image.open(image.file_path))[40,60])) < .8
    mask = np.asarray(raster_mask(DRAWING,(120,80)))
    assert np.array_equal(np.array(result)[mask==0],before[mask==0])
    assert digest(image.file_path)==source_sha
    assert digest(Path(job['file_path']).with_suffix('.raw.png'))==raw_sha
    assert job['tone_report']['samples']>=64


def test_match_samples_background_not_intentionally_recolored_object():
    original=Image.new('RGB',(120,80),(160,160,160))
    edited=Image.new('RGB',(120,80),(140,140,140))
    hard=raster_mask(DRAWING,original.size)
    report1=estimate_lightness(original,edited,hard)
    edited.paste(Image.new('RGB',original.size,'red'),mask=hard)
    report2=estimate_lightness(original,edited,hard)
    assert report1['offset']==pytest.approx(report2['offset'])


def test_full_mask_disables_auto_matching_but_manual_adjustments_still_work():
    source=Image.new('RGB',(100,80),'#aaaaaa')
    edited=Image.new('RGB',source.size,'#888888')
    mask=Image.new('L',source.size,255)
    result,report=adjust_edit(edited,source,mask,mask,tone_options(dict(match=True,lightness=10)))
    assert report['offset']==0 and report['reason']
    assert np.mean(lab_l(result)) > np.mean(lab_l(edited))


def test_white_background_can_match_without_treating_it_as_missing_context():
    source=Image.new('RGB',(120,80),'white')
    edited=Image.new('RGB',source.size,(235,235,235))
    hard=raster_mask(DRAWING,source.size)
    result,report=adjust_edit(edited,source,hard,hard,tone_options(new=True))
    assert report['samples']>=64 and not report['reason']
    assert min(result.getpixel((60,40)))>=254


def test_neutral_preserves_pixels_and_saturation_zero_removes_chroma():
    source=Image.new('RGB',(80,60),'gray')
    edited=Image.new('RGB',source.size,(160,70,55))
    mask=Image.new('L',source.size,255)
    neutral,_=adjust_edit(edited,source,mask,mask,tone_options())
    assert neutral.tobytes()==edited.tobytes()
    gray,_=adjust_edit(edited,source,mask,mask,tone_options(dict(saturation=0)))
    rgb=np.array(gray).astype(int)
    assert np.max(rgb.max(axis=2)-rgb.min(axis=2)) <= 1
    assert abs(float(lab_l(gray).mean())-float(lab_l(edited).mean())) < .5


def test_repeated_adjustments_do_not_accumulate_and_reset_restores_raw_composite(library):
    repo,job,_=tone_job(library,color=(180,100,80))
    neutral=digest(job['file_path'])
    options=tone_options(dict(lightness=12,saturation=140))
    repo.queue_tone(job['id'],options); repo.render_tone(job['id'])
    first=digest(job['file_path'])
    repo.queue_tone(job['id'],tone_options(dict(lightness=-5,saturation=50))); repo.render_tone(job['id'])
    repo.queue_tone(job['id'],options); repo.render_tone(job['id'])
    assert digest(job['file_path'])==first
    repo.queue_tone(job['id'],tone_options()); repo.render_tone(job['id'])
    assert digest(job['file_path'])==neutral


def test_latest_slider_request_is_preserved_during_older_render(library):
    repo,job,_=tone_job(library)
    first=tone_options(dict(lightness=5));latest=tone_options(dict(lightness=18,saturation=80))
    repo.queue_tone(job['id'],first)
    original_adjust=adjust_edit
    def queue_newer(*args):
        result=original_adjust(*args)
        repo.queue_tone(job['id'],latest)
        return result
    with patch('eidory.core.preview_ai.adjust_edit',side_effect=queue_newer):
        intermediate=repo.render_tone(job['id'])
    assert intermediate['tone']==first
    assert intermediate['tone_pending']['options']==latest
    final=repo.render_tone(job['id'])
    assert final['tone']==latest and 'tone_pending' not in final


def test_pending_adjustment_survives_reopen_and_is_saved_before_adoption(library,tmp_path):
    repo,job,image=tone_job(library)
    source_sha=digest(image.file_path)
    options=tone_options(dict(lightness=15))
    repo.queue_tone(job['id'],options)
    reopened=PreviewAIJobs(repo.store,repo.paths)
    adopted=reopened.adopt(job['id'],tmp_path/'saved')
    assert reopened.get(job['id'])['tone']==options
    assert digest(adopted.file_path)==digest(job['file_path'])
    assert digest(image.file_path)==source_sha
    with pytest.raises(ValueError,match='尚未采用'):
        reopened.queue_tone(job['id'],tone_options())


def test_pending_adjustment_is_replaced_and_can_restore_original(library):
    repo,job,image=tone_job(library)
    sha=digest(image.file_path)
    repo.queue_tone(job['id'],tone_options(dict(lightness=15)))
    repo.replace(job['id'])
    assert lab_l(Image.open(image.file_path))[40,60] > 65
    ImageReplacement(repo.store,repo.paths).restore(image.id)
    assert digest(image.file_path)==sha


@pytest.mark.parametrize('bad',[dict(lightness=float('nan')),dict(lightness=51),dict(saturation=201),dict(strength=-1),dict(match='yes')])
def test_invalid_tone_values_are_rejected(bad):
    with pytest.raises(ValueError):tone_options(bad)


def test_transparency_is_preserved_after_adjustment(library):
    repo,job,image=tone_job(library)
    pixels=Image.open(job['input_path']).convert('RGBA')
    pixels.putalpha(Image.new('L',pixels.size,123));pixels.save(job['input_path'])
    repo.queue_tone(job['id'],tone_options(dict(lightness=10)))
    repo.render_tone(job['id'])
    assert Image.open(job['file_path']).getchannel('A').getextrema()==(123,123)


def test_ui_sliders_persist_latest_values_and_disable_save_until_applied(window,library,app):
    repo,job,image=tone_job(library)
    window.preview_ai.open(image,'edit',job_id=job['id'])
    app.processEvents()
    panel=window.preview_ai.panel
    assert panel.save.isEnabled()
    panel.tone_spins['lightness'].setValue(15)
    panel.tone_spins['saturation'].setValue(75)
    assert not panel.save.isEnabled()
    QTest.qWait(450);app.processEvents()
    assert repo.get(job['id'])['tone']==tone_options(dict(lightness=15,saturation=75))
    assert panel.save.isEnabled()
    panel.reset_tone()
    QTest.qWait(450);app.processEvents()
    assert repo.get(job['id'])['tone']==tone_options()
    panel.close()


def test_closing_during_debounce_preserves_adjustment_for_background_processing(window,library,app):
    repo,job,image=tone_job(library)
    window.preview_ai.open(image,'edit',job_id=job['id']);app.processEvents()
    panel=window.preview_ai.panel
    panel.tone_spins['lightness'].setValue(11)
    panel.close();app.processEvents()
    assert repo.get(job['id'])['tone']['lightness']==11
    assert window.preview_ai.panel is None


def test_tone_controls_expand_sliders_and_fit_small_window(window,library,app):
    repo,job,image=tone_job(library)
    window.preview_ai.open(image,'edit',job_id=job['id']);app.processEvents()
    panel=window.preview_ai.panel
    panel.resize(1100,720);app.processEvents()
    assert panel.tone_sliders['lightness'].width()>=240
    assert panel.minimumSizeHint().height()<=740
    assert not panel.tone_sliders['strength'].isEnabled()
    panel.tone_match.setChecked(True)
    assert panel.tone_sliders['strength'].isEnabled()
    panel.close();app.processEvents()


def test_pending_tone_and_raw_output_survive_workspace_migration(library,tmp_path):
    from eidory.config import AppPaths
    from eidory.core.metadata_store import MetadataStore
    from eidory.core.workspace_package import export_workspace,restore_workspace
    repo,job,_=tone_job(library)
    options=tone_options(dict(lightness=9,saturation=130))
    repo.queue_tone(job['id'],options)
    export_workspace(repo.paths,tmp_path/'package',include_model=False)
    data,_=restore_workspace(tmp_path/'package',tmp_path/'restored')
    # The test fixture uses db.sqlite3 rather than the app's canonical name.
    restored=MetadataStore(data/'eidory.sqlite3')
    paths=AppPaths(data,data/'thumbnails',data/'eidory.sqlite3',data/'logs')
    moved=PreviewAIJobs(restored,paths)
    pending=moved.get(job['id'])
    assert pending['tone_pending']['options']==options
    assert Path(pending['file_path']).with_suffix('.raw.png').exists()
    rendered=moved.render_tone(job['id'])
    assert rendered['tone']==options and 'tone_pending' not in rendered
