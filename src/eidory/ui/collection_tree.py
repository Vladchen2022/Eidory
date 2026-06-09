from __future__ import annotations

from PySide6.QtCore import QByteArray, Qt, Signal
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem


IMAGE_IDS_MIME = "application/x-eidory-image-ids"


class CollectionTreeWidget(QTreeWidget):
    treeReordered = Signal(object)
    imagesDropped = Signal(int, object)
    filesDropped = Signal(int, object)
    rootFilesDropped = Signal(object)

    def __init__(self):
        super().__init__()
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)
        self.setDragDropMode(QTreeWidget.DragDropMode.DragDrop)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat(IMAGE_IDS_MIME) or event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasFormat(IMAGE_IDS_MIME):
            item = self.itemAt(event.position().toPoint())
            if item is not None and item.data(0, Qt.ItemDataRole.UserRole) is not None:
                event.setDropAction(Qt.DropAction.CopyAction)
                event.accept()
            else:
                event.ignore()
            return
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        if event.mimeData().hasFormat(IMAGE_IDS_MIME):
            item = self.itemAt(event.position().toPoint())
            if item is None:
                event.ignore()
                return
            collection_id = item.data(0, Qt.ItemDataRole.UserRole)
            if collection_id is None:
                event.ignore()
                return
            image_ids = self._decode_image_ids(event.mimeData().data(IMAGE_IDS_MIME))
            if image_ids:
                self.imagesDropped.emit(int(collection_id), image_ids)
                event.setDropAction(Qt.DropAction.CopyAction)
                event.accept()
            else:
                event.ignore()
            return

        if event.mimeData().hasUrls():
            item = self.itemAt(event.position().toPoint())
            paths = self._local_paths(event.mimeData().urls())
            if paths:
                collection_id = (
                    item.data(0, Qt.ItemDataRole.UserRole)
                    if item is not None
                    else None
                )
                if collection_id is None:
                    self.rootFilesDropped.emit(paths)
                else:
                    self.filesDropped.emit(int(collection_id), paths)
                event.setDropAction(Qt.DropAction.CopyAction)
                event.accept()
            else:
                event.ignore()
            return

        super().dropEvent(event)
        self.treeReordered.emit(self._tree_updates())

    def _tree_updates(self) -> list[tuple[int, int | None, int]]:
        updates: list[tuple[int, int | None, int]] = []

        def visit(parent: QTreeWidgetItem | None, parent_id: int | None) -> None:
            child_count = self.topLevelItemCount() if parent is None else parent.childCount()
            for order in range(child_count):
                child = self.topLevelItem(order) if parent is None else parent.child(order)
                collection_id = child.data(0, Qt.ItemDataRole.UserRole)
                if collection_id is None:
                    visit(child, None)
                    continue
                updates.append((int(collection_id), parent_id, order))
                visit(child, int(collection_id))

        visit(None, None)
        return updates

    @staticmethod
    def encode_image_ids(image_ids: list[int]) -> QByteArray:
        return QByteArray(",".join(str(image_id) for image_id in image_ids).encode("utf-8"))

    @staticmethod
    def _decode_image_ids(data: QByteArray) -> list[int]:
        text = bytes(data).decode("utf-8", errors="ignore")
        image_ids: list[int] = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                image_ids.append(int(part))
            except ValueError:
                continue
        return image_ids

    @staticmethod
    def _local_paths(urls) -> list[str]:
        paths: list[str] = []
        for url in urls:
            if url.isLocalFile():
                path = url.toLocalFile()
                if path:
                    paths.append(path)
        return paths
