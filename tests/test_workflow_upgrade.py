from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication
from PIL import Image

from eidory.core.automatic_backup import automatic_backup, database_summary
from eidory.core.creative_tasks import node_task
from eidory.core.filter_engine import balanced_rank
from eidory.core.metadata_store import MetadataStore
from eidory.core.scanner import ImageScanner
from eidory.core.thumbnailer import Thumbnailer
from eidory.core.search_targets import resolved_targets, target_signature
from eidory.core.readiness import check_model_service
from eidory.core.search_filters import SearchFilter
from eidory.ui.project_board import ProjectBoardView


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def images(tmp_path):
    store = MetadataStore(tmp_path / 'db.sqlite3'); store.initialize()
    folder = tmp_path / 'media'; folder.mkdir()
    Image.new('RGB', (120, 80), 'orange').save(folder / 'one.png')
    scan = ImageScanner(store, Thumbnailer(tmp_path / 'thumbs')).scan_folder(str(folder))
    im = store.get_image(scan.image_ids[0])
    return store, [replace(im, id=i+1) for i in range(60)]


@pytest.mark.parametrize('count', [10, 30, 60])
def test_sections_move_restore_and_preserve_locked_layout(app, images, count):
    _, ims = images
    board = ProjectBoardView(); board.resize(1300, 800)
    badges = {im.id: ['物件' if im.id % 2 else '人物'] for im in ims}
    badges[1].append('主参考')
    board.set_images(ims[:count], badges_by_image_id=badges)
    board.arrange_by_labels()
    assert len(board._section_items) == 2
    a, b = board._section_items
    assert not a.sceneBoundingRect().intersects(b.sceneBoundingRect())
    old = board._image_items[a.image_ids[0]].pos()
    a.moveBy(200, 100)
    assert board._image_items[a.image_ids[0]].pos() == old + QPointF(200, 100)
    a.locked = True
    frozen = {str(i): board.layout_payload()['items'][str(i)] for i in a.image_ids}
    board.arrange_by_labels()
    assert all(board.layout_payload()['items'][i] == position for i, position in frozen.items())
    payload = board.layout_payload()
    board.set_images(ims[:count], layout_payload=payload, badges_by_image_id=badges)
    assert board.layout_payload() == payload
    if count < 60:
        board.set_images(ims[:count+1], layout_payload=payload, badges_by_image_id=badges)
        assert all(board.layout_payload()['items'][i] == position for i, position in frozen.items())
    board.close()


def test_balanced_rank_independent_of_condition_order(images):
    _, ims = images
    a, b = SearchFilter('semantic', '工程师'), SearchFilter('semantic', '机器人')
    raw = {a: [replace(im, score=1-i*.01) for i, im in enumerate(ims[:20])],
           b: [replace(im, score=1-i*.01) for i, im in enumerate(ims[20:40])]}
    result = balanced_rank(ims, [a,b], raw)
    assert {result[0].id, result[1].id} == {1,21}
    assert [im.id for im in result] == [im.id for im in balanced_rank(ims, [b,a], raw)]
    assert len({im.id for im in result}) == 60


def test_backup_retention_and_preview(images, tmp_path):
    store, _ = images
    root = tmp_path/'backups'; root.mkdir()
    (root/'manual.sqlite3').write_bytes(b'keep')
    for i in range(1,10):
        result = automatic_backup(store.database_path, root, day=f'202609{i:02}')
    assert len(list(root.glob('auto-*'))) == 7
    assert database_summary(result)['images'] == 1
    assert (root/'manual.sqlite3').read_bytes() == b'keep'
    assert automatic_backup(store.database_path, root, day='20260909') == result


def test_role_specific_labels():
    assert node_task('人物')[0] == '完善人物方案'
    assert node_task('地点')[0] == '补充空间组成'
    assert node_task('世界观')[0] == '完善说明'
    assert node_task('氛围')[0] == '完善说明'


def test_manual_targets_survive_checkbox_changes():
    from types import SimpleNamespace
    node = SimpleNamespace(note='工程师', search_query='工程师', title='人物')
    entries = {'base_note': '三位工程师', 'base_query': '工程师', 'rows': [{'id':'r1','name':'技师','enabled':True}]}
    saved = {'signature':target_signature(node,entries), 'rows':[{'id':'manual','kind':'手动目标','name':'手部','query':'精细维修手势','enabled':True}]}
    entries['rows'][0]['enabled'] = False
    result = resolved_targets(node, entries, {'search_targets_json':json.dumps(saved)})
    assert result[-1]['query'] == '精细维修手势'
    assert not next(r for r in result if r['id']=='r1')['enabled']


@pytest.mark.parametrize('code', [401,403,404,429,500])
def test_readiness_rejects_http_errors(code):
    from urllib.error import HTTPError
    with patch('urllib.request.urlopen', side_effect=HTTPError('http://local',code,'error',{},None)):
        ok, message = check_model_service('http://local/v1')
    assert not ok and str(code) in message
