import threading
import time
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest

from test_filter_workflow import app, library, window
from test_board_inspector import board_project
from eidory.core.vector_index import VectorIndex
from eidory.core.thumbnailer import Thumbnailer
from eidory.core.search_service import SearchService


def wait_until(condition, timeout=5):
    deadline=time.monotonic()+timeout
    while not condition() and time.monotonic()<deadline:
        QTest.qWait(5)
    assert condition()


def test_board_loading_is_nonblocking_and_stale_results_never_replace_new_project(window, library):
    w=window
    board=w.project_board_view
    first, second=library[2][:2]
    # No synchronous reader may run while setting up a scene.
    with patch.object(board, '_pixmap_for', side_effect=AssertionError('GUI decode')):
        board.set_images([first])
        board.set_images([second])
    wait_until(lambda: not board._pending_image_loads)
    assert list(board._image_items)==[second.id]
    actual=board._image_items[second.id].source_pixmap()
    assert not actual.isNull()
    assert actual.width()==second.width
    assert actual.toImage().pixelColor(1,1) == QImage(second.thumbnail_path).pixelColor(1,1)


def test_loading_keeps_user_edits_and_can_release_high_resolution(window, library):
    board=window.project_board_view
    image=library[2][0]
    window.center_result_stack.setCurrentWidget(board)
    board.set_images([image])
    item=board._image_items[image.id]
    item.setPos(712,413); item.set_pinned(True); item.set_grayscale(True)
    layout=board.layout_payload()
    wait_until(lambda: not board._pending_image_loads)
    assert board.layout_payload()==layout
    assert item.pos().x()==712 and item.pos().y()==413
    assert item.is_pinned() and item.is_grayscale()
    board._focus_quality_image_ids.add(image.id)
    window.center_result_stack.setCurrentWidget(window.grid_view)
    assert image.id not in board._focus_quality_image_ids


def test_board_batch_metadata_never_reads_individual_decisions(window, library):
    w=window
    _,_,_,inspector=board_project(w,library[2])
    with patch.object(w.workflow_upgrade.state,'decision',side_effect=AssertionError('N+1 read')):
        board=w.project_board_view
        board._scene.blockSignals(True)
        for item in board._image_items.values(): item.setSelected(True)
        board._scene.blockSignals(False)
        board._selected_image_ids_cache=list(board._image_items)
        inspector.refresh()
        inspector.save(status='primary')
    assert all('主参考' in item._badges for item in board._image_items.values())


def test_unreadable_board_image_is_named_without_retry_loop(window, library):
    board=window.project_board_view
    image=replace(library[2][0],file_path='/missing/performance.png',thumbnail_path='',file_name='missing.png')
    window.center_result_stack.setCurrentWidget(board)
    board.set_images([image])
    wait_until(lambda: not board._pending_image_loads)
    assert image.id in board._failed_image_ids
    item=board._image_items[image.id]
    assert [child.toPlainText() for child in item.childItems()]==['missing.png']
    for _ in range(3):board._refresh_visible_quality()
    assert not board._pending_image_loads
    # A file can become available again (for example a reconnected volume).
    # Successful explicit reload must also clear its failure label/state.
    board._images_by_id[image.id] = library[2][0]
    from eidory.ui.project_board import MAX_BOARD_THUMBNAIL_SIDE
    board._request_image(image.id, MAX_BOARD_THUMBNAIL_SIDE)
    wait_until(lambda: not board._pending_image_loads)
    assert image.id not in board._failed_image_ids
    assert not item.childItems()
    assert item.toolTip() == library[2][0].file_path


def test_note_burst_is_one_write_and_switch_flushes_to_original_image(window,library):
    w=window
    _,a,_,inspector=board_project(w,library[2])
    first,second=library[2][:2]
    w.project_board_view.select_image_id(first.id)
    store=w.creative_workspace.store
    with patch.object(store,'save_reference_note',wraps=store.save_reference_note) as save:
        for text in ('只','只参','只参考','只参考轮廓'):
            inspector.note.setPlainText(text)
        assert save.call_count==0
        w.project_board_view.select_image_id(second.id)
        assert save.call_count==1
    assert store.reference_note(a.id,first.id)=='只参考轮廓'
    assert store.reference_note(a.id,second.id)==''


def test_close_flushes_notes_and_waits_for_inflight_work_without_blocking_event_loop(window,library):
    w=window
    _,a,_,inspector=board_project(w,library[2])
    iid=library[2][0].id
    w.project_board_view.select_image_id(iid)
    inspector.note.setPlainText('退出前最后一次编辑')
    release=threading.Event()
    done=threading.Event()
    # The normal fixture makes background work synchronous; exercise the real implementation.
    from eidory.ui.main_window import MainWindow
    MainWindow._start_background_task(w,lambda:(release.wait(3),done.set()))
    QTimer.singleShot(100,release.set)
    w.close()
    assert w.isVisible() and not w._shutdown_ready
    wait_until(lambda:w._shutdown_ready)
    assert done.is_set() and not w._file_watch_thread.isRunning()
    assert w.creative_workspace.store.reference_note(a.id,iid)=='退出前最后一次编辑'


def test_exact_search_incremental_updates_removals_and_compaction():
    rng=np.random.default_rng(93)
    vectors=rng.normal(size=(400,32)).astype(np.float32)
    class Store:
        calls=0
        def embeddings_for_model(self,**kwargs):
            self.calls+=1
            return np.arange(1,401),vectors.copy()
    store=Store();index=VectorIndex(store,model_name='test',model_revision='r',embedding_dim=32)
    index.reload()
    expected={i+1:v/np.linalg.norm(v) for i,v in enumerate(vectors)}
    for iid in range(200,700):
        v=rng.normal(size=32).astype(np.float32)
        expected[iid]=v/np.linalg.norm(v); index.upsert(iid,v)
    for iid in (12,210,580): expected.pop(iid); index.remove(iid)
    q=rng.normal(size=32).astype(np.float32);q/=np.linalg.norm(q)
    for scope in (None,set(range(180,245)),{210,580},{698,699}):
        wanted=sorted(((i,float(v@q)) for i,v in expected.items() if scope is None or i in scope),key=lambda pair:-pair[1])
        actual=index.search(q,10000,scope)
        assert [i for i,_ in actual]==[i for i,_ in wanted]
        assert np.allclose([s for _,s in actual],[s for _,s in wanted],atol=2e-7)
    assert store.calls==1
    assert not index._matrix.flags.writeable


def test_text_vector_cache_is_bounded_and_model_specific():
    class Provider:
        model_name='m';model_revision='1';dim=2;calls=0
        def encode_text(self,q):
            self.calls+=1
            return np.array([self.calls,1],dtype=np.float32)
    provider=Provider()
    service=SearchService(store=None,embedding_provider=provider,vector_index=None)
    a=service._encode_query('机器人');a[0]=999
    assert service._encode_query('机器人')[0]==1
    assert provider.calls==1
    provider.model_revision='2'
    assert service._encode_query('机器人')[0]==2
    for i in range(140):service._encode_query(str(i))
    assert len(service._query_vectors)==128


def test_failed_thumbnail_write_preserves_previous_good_file(tmp_path):
    thumbnailer=Thumbnailer(tmp_path/'thumbs')
    source=tmp_path/'source.png'
    Image.new('RGB',(100,100),'blue').save(source)
    target=thumbnailer.generate(1,str(source));before=target.read_bytes()
    with patch.object(Image.Image,'save',side_effect=RuntimeError('encoder failed')):
        try:thumbnailer.generate(1,str(source))
        except RuntimeError:pass
        else:raise AssertionError('Expected encoder failure')
    assert target.read_bytes()==before
    assert list(target.parent.glob('.thumb-*'))==[]


def test_rotated_photo_keeps_display_orientation_after_async_loading(window, library, tmp_path):
    source = tmp_path / 'rotated.jpg'
    photo = Image.new('RGB', (300, 160), 'red')
    exif = photo.getexif()
    exif[274] = 6
    photo.save(source, exif=exif)
    from eidory.core.scanner import ImageScanner
    assert ImageScanner._read_dimensions(str(source)) == (160, 300)
    thumbnail = Thumbnailer(tmp_path / 'thumbs').generate(998, str(source))
    image = replace(library[2][0], id=998, file_path=str(source),
                    thumbnail_path=str(thumbnail), width=300, height=160)
    board = window.project_board_view
    window.center_result_stack.setCurrentWidget(board)
    layout = {'items': {str(image.id): {'x': 300, 'y': 200, 'width': 160, 'height': 300}}}
    board.set_images([image], layout_payload=layout)
    wait_until(lambda: not board._pending_image_loads)
    item = board._image_items[image.id]
    assert abs(item.display_width / item.display_height - 160 / 300) < .001
    assert item.pos().x() == 300 and item.pos().y() == 200
    # A second opening can use cached, already-oriented thumbnail pixels.
    board.set_images([])
    board.set_images([image], layout_payload=layout)
    item = board._image_items[image.id]
    assert abs(item.display_width / item.display_height - 160 / 300) < .001
