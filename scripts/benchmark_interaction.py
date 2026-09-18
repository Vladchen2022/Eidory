"""Repeatable local performance probe; use an isolated EIDORY_DATA_DIR copy.

No models or network requests are needed. Results include time to return control
and time until all board images arrive, so async work is not counted as free.
"""
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QThreadPool, QTimer, QEventLoop
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import numpy as np

from eidory.config import AppPaths
from eidory.core.metadata_store import MetadataStore
from eidory.core.vector_index import VectorIndex
from eidory.ui.accessibility import disable_qt_accessibility
from eidory.ui.main_window import MainWindow
from eidory.ui import project_board as pb


def main():
    if not os.environ.get('EIDORY_DATA_DIR'):
        raise SystemExit('Set EIDORY_DATA_DIR to an isolated database copy')
    disable_qt_accessibility()
    app = QApplication([])
    disable_qt_accessibility()
    paths = AppPaths.default()
    paths.ensure()
    store = MetadataStore(paths.database_path)
    report = {}
    start = time.perf_counter()
    with patch.object(MainWindow, '_run_startup_self_check'), patch.object(MainWindow, '_refresh_file_watcher'), patch.object(MainWindow, '_cleanup_missing_active_scan_roots', return_value=(0, 0)):
        w = MainWindow(paths=paths, store=store)
        w.resize(1500, 950)
        w.show()
        app.processEvents()
        report['window_ready_ms'] = (time.perf_counter()-start)*1000
        images = store.list_images(limit=500)
        board = w.project_board_view
        w.center_result_stack.setCurrentWidget(board)
        for count in (54, 200, 500):
            pb._BOARD_PIXMAP_CACHE.clear()
            pb._BOARD_PIXMAP_CACHE_BYTES.clear()
            pb._BOARD_PIXMAP_CACHE_TOTAL_BYTES = 0
            start = time.perf_counter()
            board.set_images(images[:count], title=f'Benchmark {count}')
            sync_ms = (time.perf_counter()-start)*1000
            deadline = time.monotonic()+60
            if getattr(board, '_pending_image_loads', {}):
                loop=QEventLoop(); timer=QTimer(); timer.setInterval(5)
                def ready():
                    if not board._pending_image_loads or time.monotonic()>deadline:
                        loop.quit()
                timer.timeout.connect(ready);timer.start();loop.exec();timer.stop()
            app.processEvents()
            ready_ms = (time.perf_counter()-start)*1000
            samples=[]
            for _ in range(20):
                t=time.perf_counter(); board.viewport().repaint(); samples.append((time.perf_counter()-t)*1000)
            report[f'board_{count}']={'return_ms':sync_ms,'ready_ms':ready_ms,'paint_p95_ms':float(np.percentile(samples,95))}
        with store.connect() as conn:
            row=conn.execute('SELECT n.project_id,n.id FROM creative_nodes n WHERE n.parent_id IS NULL ORDER BY (SELECT count(*) FROM creative_node_images l JOIN creative_nodes ch ON ch.id=l.node_id WHERE ch.project_id=n.project_id) DESC LIMIT 1').fetchone()
        w._load_creative_project(int(row['project_id']), select_node_id=int(row['id']))
        w.creative_flow.show_project_board()
        board._scene.blockSignals(True)
        for item in board._image_items.values(): item.setSelected(True)
        board._scene.blockSignals(False)
        board._selected_image_ids_cache=list(board._image_items)
        query_count=0
        connect=store.connect
        @contextmanager
        def counted(*args, **kwargs):
            nonlocal query_count
            query_count+=1
            with connect(*args,**kwargs) as conn: yield conn
        samples=[]
        with patch.object(store,'connect',counted):
            for _ in range(10):
                t=time.perf_counter();w.creative_flow.board_inspector.refresh();samples.append((time.perf_counter()-t)*1000)
        report['inspector']={'images':len(board._image_items),'p95_ms':float(np.percentile(samples,95)), 'connections_per_refresh':query_count/10}
        w.close()
        for pool in w.findChildren(QThreadPool): pool.waitForDone(60000)
        QTest.qWait(300)
    rng=np.random.default_rng(1718)
    matrix=rng.normal(size=(23500,512)).astype(np.float32)
    class VStore:
        calls=0
        def embeddings_for_model(self,**kwargs):
            self.calls+=1
            return np.arange(1,len(matrix)+1), matrix.copy()
    vs=VStore(); index=VectorIndex(vs,model_name='bench',model_revision='1',embedding_dim=512)
    query=rng.normal(size=512).astype(np.float32)
    index.reload()
    for size in (200,23500):
        allowed=set(range(1,size+1)); samples=[]
        for _ in range(25):
            t=time.perf_counter(); index.search(query,size,allowed);samples.append((time.perf_counter()-t)*1000)
        report[f'vector_scope_{size}_p95_ms']=float(np.percentile(samples,95))
    t=time.perf_counter()
    for i in range(30):
        index.upsert(24000+i,query);index.search(query,200,set(range(1,201)))
    report['index_insert_search_30_ms']=(time.perf_counter()-t)*1000
    report['index_store_reloads']=vs.calls
    Path(sys.argv[1]).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
