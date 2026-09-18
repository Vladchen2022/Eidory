"""Long-running Qt event-loop regression on an isolated library copy.

Usage: EIDORY_DATA_DIR=/isolated/copy python scripts/soak_performance.py 3600 report.json
This automated stress probe complements (does not replace) native screen tests.
"""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from eidory.config import AppPaths
from eidory.core.metadata_store import MetadataStore
from eidory.ui.main_window import MainWindow
from eidory.ui.accessibility import disable_qt_accessibility


def main():
    if not os.environ.get('EIDORY_DATA_DIR'):
        raise SystemExit('An isolated data directory is required')
    seconds=int(sys.argv[1]);report_path=Path(sys.argv[2])
    disable_qt_accessibility();app=QApplication([]);disable_qt_accessibility()
    paths=AppPaths.default();paths.ensure();store=MetadataStore(paths.database_path)
    with patch.object(MainWindow,'_run_startup_self_check'),patch.object(MainWindow,'_refresh_file_watcher'),patch.object(MainWindow,'_cleanup_missing_active_scan_roots',return_value=(0,0)):
        w=MainWindow(paths=paths,store=store)
        w.resize(1500,950);w.show()
        images=store.list_images(limit=500)
        pid=w._create_creative_project_from_seed(title='自动长时性能验收',brief='科研站的机器人维护',extra='测试副本')
        nodes=store.list_creative_nodes(pid)
        node=next(n for n in nodes if n.title=='物件')
        store.add_images_to_creative_node(node.id,[i.id for i in images])
        w._load_creative_project(pid,select_node_id=node.id)
        w.creative_flow.show_project_board()
        started=time.monotonic();ticks=0;latencies=[];rss=[];errors=[];heartbeat=[]
        last=time.monotonic()
        def pulse():
            nonlocal last
            now=time.monotonic();heartbeat.append((now-last)*1000);last=now
        heartbeat_timer=QTimer(w);heartbeat_timer.setInterval(50);heartbeat_timer.timeout.connect(pulse);heartbeat_timer.start()
        def record():
            report={'elapsed_seconds':round(time.monotonic()-started,1),'cycles':ticks,
                    'rss_mib':rss,'max_action_ms':max(latencies,default=0),'max_heartbeat_gap_ms':max(heartbeat,default=0),
                    'errors':errors,'runtime_errors':w.error_log_messages}
            report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False))
        def step():
            nonlocal ticks
            try:
                t=time.monotonic();board=w.project_board_view;phase=ticks%8
                if phase==0:
                    w.creative_flow.show_project_board()
                elif phase==1:
                    board.select_image_id(images[ticks%len(images)].id);board.toggle_fit_selection_or_restore()
                elif phase==2:
                    board.fit_all_images();w.creative_flow.board_inspector.save(status='primary' if ticks%16==2 else 'candidate')
                elif phase==3:
                    board.select_image_id(images[(ticks+1)%len(images)].id)
                    w.creative_flow.board_inspector.note.setPlainText(f'性能验收第 {ticks} 步')
                elif phase==4:
                    board.toggle_node_sections()
                elif phase==5:
                    w.creative_flow.show_planning()
                elif phase==6:
                    w.creative_flow.show_project_board()
                    board._zoom_by(1.2);board._zoom_by(1/1.2)
                else:
                    w._start_background_task(lambda:store.list_images(limit=500),name='soak-read')
                    w.resize(1450+(ticks%3)*50,920+(ticks%2)*30)
                latencies.append((time.monotonic()-t)*1000);ticks+=1
                if ticks%12==0:
                    memory=int(subprocess.check_output(['ps','-o','rss=','-p',str(os.getpid())]).strip())/1024
                    rss.append([round(time.monotonic()-started),round(memory,1)])
                    record()
                if time.monotonic()-started>=seconds:
                    timer.stop();record();w.close()
            except Exception as exc:
                errors.append(repr(exc));record();timer.stop();w.close()
        timer=QTimer(w);timer.setInterval(5000);timer.timeout.connect(step);timer.start()
        app.exec()
        with store.connect() as conn:
            check=conn.execute('PRAGMA quick_check').fetchone()[0]
        record()
        result=json.loads(report_path.read_text());result['database_check']=check;result['shutdown_ready']=w._shutdown_ready
        report_path.write_text(json.dumps(result,indent=2,ensure_ascii=False))
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
