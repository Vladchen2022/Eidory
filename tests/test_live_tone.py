import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF
from PySide6.QtTest import QTest

from test_filter_workflow import app, library, window
from test_preview_ai import suspend_qt_wrapper_cycle_collection
from test_edit_tone import tone_job
from eidory.core.edit_tone import TonePreview, adjust_edit, estimate_lightness, tone_options
from eidory.core.image_upscale import digest
from eidory.ui.tone_preview import LatestTonePreview


def test_cached_preview_matches_full_resolution_processing(library):
    repo,job,_=tone_job(library,color=(180,100,80))
    cache=TonePreview.from_job(job)
    for options in [tone_options(),tone_options(dict(lightness=10,saturation=70)),tone_options(new=True)]:
        repo.queue_tone(job['id'],options);result=repo.render_tone(job['id'],cache.analysis)
        expected=np.array(Image.open(result['file_path']).convert('RGBA'))
        preview=np.array(cache.render(options))
        assert np.max(np.abs(expected.astype(int)-preview.astype(int)))<=1


def test_preview_frames_reuse_lab_and_analysis_without_file_io(library):
    repo,job,_=tone_job(library)
    cache=TonePreview.from_job(job)
    with patch('PIL.Image.open',side_effect=AssertionError('frame performed file I/O')), \
         patch('eidory.core.edit_tone.estimate_lightness',side_effect=AssertionError('repeated analysis')), \
         patch('eidory.core.edit_tone._lab',side_effect=AssertionError('repeated RGB to Lab')):
        for value in range(-10,11):cache.render(tone_options(dict(match=True,lightness=value)))


def test_screen_cache_is_bounded_and_preserves_transparency_and_unselected_pixels():
    original=Image.new('RGBA',(4000,2400),(160,160,160,93))
    raw=Image.new('RGB',original.size,(120,90,60))
    hard=Image.new('L',original.size)
    from PIL import ImageDraw
    ImageDraw.Draw(hard).rectangle((1000,600,3000,1800),fill=255)
    cache=TonePreview(original,raw,hard,hard)
    assert cache.original.width*cache.original.height<=TonePreview.MAX_PIXELS
    assert cache.nbytes<25*1024*1024
    result=np.array(cache.render(tone_options(dict(lightness=10,saturation=50))))
    assert np.all(result[:,:,3]==93)
    assert tuple(result[0,0])==cache.original.getpixel((0,0))


def test_latest_value_queue_does_not_accumulate_or_starve_during_drag(app):
    waiting=[];painted=[];rendered=[]
    class Cache:
        def render(self,options):
            rendered.append(options['lightness'])
            return Image.new('RGBA',(2,2),'white')
    def start(fn):waiting.append(fn);return True
    actor=LatestTonePreview(start)
    actor.ready.connect(lambda request,image:painted.append(request[1]['lightness']))
    cache=Cache()
    actor.request('page',cache,dict(lightness=0));actor.launch()
    for v in range(1,101):actor.request('page',cache,dict(lightness=v))
    assert len(waiting)==1 and actor.running
    waiting.pop(0)()
    assert painted==[0]  # Continuous pointer movement must not starve all frames.
    QTest.qWait(20);app.processEvents()
    assert len(waiting)==1
    waiting.pop(0)()
    assert rendered==[0,100] and painted[-1]==100
    actor.close()


def test_switching_page_or_closing_discards_old_preview_frames(app):
    waiting=[];painted=[]
    actor=LatestTonePreview(lambda fn:(waiting.append(fn) or True))
    actor.ready.connect(lambda request,image:painted.append(request[0]))
    class Cache:
        def render(self,options):return Image.new('RGBA',(2,2),'white')
    actor.request('old',Cache(),{});actor.launch()
    actor.request('new',Cache(),{})
    waiting.pop(0)()
    assert not painted
    QTest.qWait(20);app.processEvents()
    actor.close()
    waiting.pop(0)()
    assert not painted


def test_drag_changes_preview_before_release_without_saving_files(window,library,app):
    repo,job,image=tone_job(library)
    window.preview_ai.open(image,'edit',job_id=job['id']);app.processEvents()
    panel=window.preview_ai.panel
    sha=digest(job['file_path'])
    slider=panel.tone_sliders['lightness']
    with patch.object(panel,'refresh_controls',side_effect=AssertionError('heavy refresh on mouse move')), \
         patch.object(window.preview_ai.repo,'queue_tone') as queue:
        slider.setSliderDown(True);slider.setValue(10)
        QTest.qWait(35);app.processEvents()
        first=panel.comparison.after_preview.copy()
        slider.setValue(20);QTest.qWait(35);app.processEvents()
        assert not first.isNull() and first!=panel.comparison.after_preview
        assert digest(job['file_path'])==sha
        queue.assert_not_called()
    slider.setSliderDown(False);QTest.qWait(100);app.processEvents()
    assert repo.get(job['id'])['tone']['lightness']==20
    assert panel.comparison.after_preview.isNull()
    assert panel.save.isEnabled()
    panel.close()


def test_preview_keeps_native_zoom_pan_and_comparison_dimensions(window,library,app):
    repo,job,image=tone_job(library)
    window.preview_ai.open(image,'edit',job_id=job['id']);app.processEvents()
    panel=window.preview_ai.panel
    view=panel.comparison
    view.actual();view.pan=QPointF(23,17)
    assert not view.before_preview.isNull()
    dimensions=view.dimensions()
    panel.tone_sliders['lightness'].setSliderDown(True)
    panel.tone_spins['lightness'].setValue(10)
    QTest.qWait(35);app.processEvents()
    assert view.scale==1 and view.pan==QPointF(23,17) and view.dimensions()==dimensions
    panel.close();app.processEvents()
    assert window.preview_ai.panel is None


def test_old_full_save_cannot_flash_over_newer_live_adjustment(window,library,app):
    repo,job,image=tone_job(library)
    controller=window.preview_ai
    controller.open(image,'edit',job_id=job['id']);app.processEvents()
    panel=controller.panel;waiting=[]
    def start(fn,**kw):
        if kw.get('name')=='tone-preview':fn()
        else:waiting.append(fn)
        return True
    with patch.object(window,'_start_background_task',side_effect=start):
        panel.tone_spins['lightness'].setValue(3)
        controller.adjust_tone(panel)
        assert len(waiting)==1
        panel.tone_sliders['lightness'].setSliderDown(True)
        panel.tone_spins['lightness'].setValue(20)
        QTest.qWait(35);app.processEvents()
        newest=panel.comparison.after_preview.copy()
        waiting.pop(0)();app.processEvents()
        assert not newest.isNull() and panel.comparison.after_preview==newest
        assert panel.tone_spins['lightness'].value()==20
        panel.tone_sliders['lightness'].setSliderDown(False)
        QTest.qWait(25);app.processEvents()
        assert len(waiting)==1
        waiting.pop(0)();app.processEvents()
        assert panel.comparison.after_preview.isNull()
        assert repo.get(job['id'])['tone']['lightness']==20
    panel.close()


def test_matching_analysis_is_reused_and_invalidated_when_input_changes(library):
    repo,job,_=tone_job(library)
    cache=TonePreview.from_job(job)
    repo.queue_tone(job['id'],tone_options(new=True))
    with patch('eidory.core.preview_ai.estimate_lightness',side_effect=AssertionError('analysis not cached')):
        repo.render_tone(job['id'],cache.analysis)
    repo.queue_tone(job['id'],tone_options(dict(match=True,lightness=3)))
    with patch('eidory.core.preview_ai.estimate_lightness',side_effect=AssertionError('analysis not reused')):
        repo.render_tone(job['id'])
    with Image.open(job['input_path']) as im:im.convert('RGB').save(job['input_path'])
    repo.queue_tone(job['id'],tone_options(dict(match=True,lightness=5)))
    with patch('eidory.core.preview_ai.estimate_lightness',wraps=estimate_lightness) as analysis:
        repo.render_tone(job['id'],cache.analysis)
    assert analysis.call_count==1


def test_opening_unchanged_source_reuses_full_resolution_pixels(window,library,app):
    from eidory.ui.preview_ai import read_image
    repo,job,image=tone_job(library)
    with patch('eidory.ui.preview_ai.read_image',wraps=read_image) as read:
        window.preview_ai.open(image,'edit',job_id=job['id']);app.processEvents()
    assert read.call_count==2
    panel=window.preview_ai.panel
    assert panel.canvas.before.cacheKey()==panel.comparison.before.cacheKey()
    panel.close()


def test_background_save_progress_does_not_move_sliders_or_resize_image(window,library,app):
    repo,job,image=tone_job(library)
    window.preview_ai.open(image,'edit',job_id=job['id']);app.processEvents()
    panel=window.preview_ai.panel
    before=[widget.geometry() for widget in (panel.comparison,*panel.tone_sliders.values())]
    panel.progress.setVisible(True);app.processEvents()
    assert [widget.geometry() for widget in (panel.comparison,*panel.tone_sliders.values())]==before
    panel.progress.setVisible(False);app.processEvents()
    assert [widget.geometry() for widget in (panel.comparison,*panel.tone_sliders.values())]==before
    panel.close()
