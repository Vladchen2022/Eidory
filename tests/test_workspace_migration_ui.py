import json
from pathlib import Path
from unittest.mock import patch

from PySide6.QtWidgets import QMessageBox
from PySide6.QtTest import QTest

from test_filter_workflow import app, library, window
from eidory.core.workspace_package import MANIFEST, export_workspace
from eidory.ui.workspace_migration import WorkspaceMigrationDialog, MigrationProgress


def test_export_ui_runs_real_worker_with_database_maintenance(window, library, tmp_path, app):
    dialog = WorkspaceMigrationDialog(window)
    dialog.include_model.setChecked(False)
    target = tmp_path/'ui-package'
    dialog.export_path.setText(str(target))
    with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes), \
         patch.object(QMessageBox, 'warning') as warning:
        dialog.export()
    assert not warning.called
    assert json.loads((target/MANIFEST).read_text())['complete']
    assert '导出完成' in dialog.export_status.text()
    assert not window._database_maintenance_active
    assert window.poll_timer.isActive()
    assert list((library[0].data_dir/'backups').glob('*.sqlite3'))
    dialog.close()


def test_reference_package_ui_shows_mapping_and_invalidates_old_selection(window, library, tmp_path):
    package=tmp_path/'package'
    export_workspace(library[0], package, include_model=False, include_media=False)
    dialog=WorkspaceMigrationDialog(window)
    dialog.package_path.setText(str(package))
    dialog.inspect()
    assert dialog.restore_button.isEnabled()
    assert len(dialog.mapping_fields)==1
    assert '逐张校验内容' in dialog.restore_summary.text()
    dialog.package_path.setText(str(tmp_path/'different'))
    assert not dialog.restore_button.isEnabled() and dialog.manifest is None
    dialog.close()


def test_progress_cancel_waits_for_worker_and_preserves_main_window(window, app):
    from eidory.core.workspace_package import MigrationCancelled
    from PySide6.QtCore import QTimer
    import time
    def task(progress, cancelled):
        while not cancelled():
            time.sleep(.002)
        raise MigrationCancelled('paused')
    dialog=MigrationProgress(window,'test',task)
    QTimer.singleShot(20,dialog.reject)
    try:
        dialog.run()
        assert False
    except MigrationCancelled:
        pass
    assert not dialog.worker.isRunning()
    assert not window._closing


def test_restore_ui_creates_workspace_but_does_not_activate_without_decision(window,library,tmp_path):
    package=tmp_path/'package'
    export_workspace(library[0],package,include_model=False)
    dialog=WorkspaceMigrationDialog(window)
    dialog.package_path.setText(str(package));dialog.inspect()
    dialog.restore_path.setText(str(tmp_path));dialog.restore_name.setText('ui-restored')
    with patch.object(QMessageBox,'exec',return_value=0), patch.object(QMessageBox,'warning') as warning:
        dialog.restore()
    assert not warning.called
    assert (tmp_path/'ui-restored/data/eidory.sqlite3').is_file()
    assert '恢复完成' in dialog.restore_summary.text()
    assert not window._closing
    assert not getattr(app,'_eidory_next_workspace',None)
    dialog.close()
