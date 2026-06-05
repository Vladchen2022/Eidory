from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel


def payload_supports_mime_data(mime_data) -> bool:
    return (
        mime_data.hasUrls()
        or mime_data.hasImage()
        or mime_data.hasHtml()
        or mime_data.hasText()
    )


def payload_from_mime_data(mime_data) -> dict[str, object]:
    urls = list(mime_data.urls()) if mime_data.hasUrls() else []
    return {
        "local_paths": [
            url.toLocalFile()
            for url in urls
            if url.isLocalFile() and url.toLocalFile()
        ],
        "urls": [
            url.toString()
            for url in urls
            if not url.isLocalFile() and url.toString()
        ],
        "html": mime_data.html() if mime_data.hasHtml() else "",
        "text": mime_data.text() if mime_data.hasText() else "",
        "image": mime_data.imageData() if mime_data.hasImage() else None,
    }


class DropImportBox(QLabel):
    payloadDropped = Signal(object)

    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self._drop_enabled = True
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumHeight(118)
        self.setStyleSheet(
            "border: 1px dashed #7b8490; border-radius: 6px; "
            "background: #252b33; color: #d8dee9; padding: 14px;"
        )

    def set_drop_enabled(self, enabled: bool, text: str | None = None) -> None:
        self._drop_enabled = enabled
        self.setEnabled(enabled)
        if text is not None:
            self.setText(text)
        opacity = "1.0" if enabled else "0.45"
        self.setStyleSheet(
            "border: 1px dashed #7b8490; border-radius: 6px; "
            "background: #252b33; color: #d8dee9; padding: 14px; "
            f"opacity: {opacity};"
        )

    @staticmethod
    def _supports_mime(mime_data) -> bool:
        return payload_supports_mime_data(mime_data)

    def dragEnterEvent(self, event) -> None:
        if self._drop_enabled and self._supports_mime(event.mimeData()):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        if self._drop_enabled and self._supports_mime(event.mimeData()):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            return
        event.ignore()

    def dropEvent(self, event) -> None:
        if not self._drop_enabled:
            event.ignore()
            return
        mime_data = event.mimeData()
        if not self._supports_mime(mime_data):
            event.ignore()
            return

        self.payloadDropped.emit(payload_from_mime_data(mime_data))
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()
