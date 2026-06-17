from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImageReader, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from eidory.core.duplicate_detection import NearDuplicateCandidate


class NearDuplicateDecision:
    REPLACE = "replace"
    SKIP = "skip"
    IMPORT = "import"


class NearDuplicateDialog(QDialog):
    def __init__(
        self,
        source_path: str | Path,
        candidates: list[NearDuplicateCandidate],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.source_path = Path(source_path)
        self.candidates = list(candidates)
        self.decision = NearDuplicateDecision.SKIP
        self._selected_index = 0

        self.setWindowTitle("发现近似图片")
        self.setModal(True)
        self.resize(980, 620)

        root = QVBoxLayout(self)
        root.setSpacing(12)

        intro = QLabel("导入图片与图库中已有图片高度相似，或已经在图库中。请选择后续操作。")
        intro.setWordWrap(True)
        root.addWidget(intro)

        previews = QHBoxLayout()
        previews.setSpacing(16)
        self.source_preview = self._preview_panel("新导入图片")
        self.existing_preview = self._preview_panel("图库内近似图片")
        previews.addWidget(self.source_preview["container"], 1)
        previews.addWidget(self.existing_preview["container"], 1)
        root.addLayout(previews, 1)

        self.candidate_list = QListWidget()
        self.candidate_list.setMaximumHeight(110)
        for candidate in self.candidates:
            item = QListWidgetItem(self._candidate_title(candidate))
            self.candidate_list.addItem(item)
        if self.candidates:
            self.candidate_list.setCurrentRow(0)
        self.candidate_list.currentRowChanged.connect(self._select_candidate)
        root.addWidget(self.candidate_list)
        self.candidate_list.setVisible(len(self.candidates) > 1)

        actions = QHBoxLayout()
        self.replace_button = QPushButton("替换已有图片")
        self.skip_button = QPushButton("放弃导入")
        self.import_button = QPushButton("仍然导入")
        self.replace_button.clicked.connect(lambda: self._finish(NearDuplicateDecision.REPLACE))
        self.skip_button.clicked.connect(lambda: self._finish(NearDuplicateDecision.SKIP))
        self.import_button.clicked.connect(lambda: self._finish(NearDuplicateDecision.IMPORT))
        actions.addStretch(1)
        actions.addWidget(self.replace_button)
        actions.addWidget(self.skip_button)
        actions.addWidget(self.import_button)
        root.addLayout(actions)

        self._set_source_preview()
        self._select_candidate(0)

    def selected_candidate(self) -> NearDuplicateCandidate | None:
        if 0 <= self._selected_index < len(self.candidates):
            return self.candidates[self._selected_index]
        return None

    def reject(self) -> None:
        self.decision = NearDuplicateDecision.SKIP
        super().reject()

    def _finish(self, decision: str) -> None:
        self.decision = decision
        self.accept()

    def _preview_panel(self, title: str) -> dict[str, QLabel | QWidget]:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setMinimumSize(360, 280)
        image_label.setStyleSheet("background: #20262e; border: 1px solid #56616e;")
        meta_label = QLabel()
        meta_label.setWordWrap(True)
        meta_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(title_label)
        layout.addWidget(image_label, 1)
        layout.addWidget(meta_label)
        return {"container": container, "image": image_label, "meta": meta_label}

    def _set_source_preview(self) -> None:
        self._set_pixmap(self.source_preview["image"], self._source_preview_path())
        self.source_preview["meta"].setText(self._path_meta(self.source_path))

    def _select_candidate(self, index: int) -> None:
        if not self.candidates:
            return
        if index < 0:
            index = 0
        self._selected_index = min(index, len(self.candidates) - 1)
        candidate = self.candidates[self._selected_index]
        preview_path = Path(candidate.hash_source)
        if not preview_path.is_file():
            preview_path = Path(candidate.image.file_path)
        self._set_pixmap(self.existing_preview["image"], preview_path)
        meta = (
            f"{candidate.image.file_name}\n"
            f"{candidate.image.width or '-'} x {candidate.image.height or '-'} / "
            f"{candidate.image.file_size:,} bytes\n"
            f"相似度 {candidate.similarity:.0%}，哈希距离 {candidate.distance}\n"
            f"{self._same_path_note(candidate)}{candidate.image.file_path}"
        )
        self.existing_preview["meta"].setText(meta)

    def _source_preview_path(self) -> Path:
        for candidate in self.candidates:
            try:
                if self.source_path.resolve() != Path(candidate.image.file_path).resolve():
                    continue
            except OSError:
                continue
            preview_path = Path(candidate.hash_source)
            if preview_path.is_file():
                return preview_path
        return self.source_path

    @staticmethod
    def _set_pixmap(label: QLabel, path: Path) -> None:
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        source_size = reader.size()
        if source_size.isValid():
            reader.setScaledSize(
                source_size.scaled(
                    QSize(420, 300),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
            )
        image = reader.read()
        if image.isNull():
            label.setText("无法预览")
            label.setPixmap(QPixmap())
            return
        pixmap = QPixmap.fromImage(image)
        label.setText("")
        label.setPixmap(pixmap)

    @staticmethod
    def _path_meta(path: Path) -> str:
        try:
            stat = path.stat()
            size = stat.st_size
        except OSError:
            size = 0
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        image_size = reader.size()
        dimensions = (
            "-"
            if not image_size.isValid()
            else f"{image_size.width()} x {image_size.height()}"
        )
        return f"{path.name}\n{dimensions} / {size:,} bytes\n{path}"

    @staticmethod
    def _candidate_title(candidate: NearDuplicateCandidate) -> str:
        image = candidate.image
        return (
            f"{candidate.similarity:.0%} | {image.file_name} | "
            f"{image.width or '-'} x {image.height or '-'} | {image.file_path}"
        )

    def _same_path_note(self, candidate: NearDuplicateCandidate) -> str:
        try:
            if self.source_path.resolve() == Path(candidate.image.file_path).resolve():
                return "已在图库中：同一路径\n"
        except OSError:
            pass
        return ""
