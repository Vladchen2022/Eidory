from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

from eidory.core.embedding_provider import EmbeddingProvider
from eidory.core.metadata_store import MetadataStore
from eidory.core.vector_index import VectorIndex
from eidory.models import ImageItem


@dataclass(frozen=True)
class EmbeddingProgress:
    image_id: int | None
    file_name: str | None
    status: str
    message: str


EmbeddingProgressCallback = Callable[[EmbeddingProgress], None]


class EmbeddingWorker(threading.Thread):
    def __init__(
        self,
        *,
        store: MetadataStore,
        provider: EmbeddingProvider,
        vector_index: VectorIndex,
        batch_size: int = 4,
        idle_sleep_seconds: float = 2.0,
        on_progress: EmbeddingProgressCallback | None = None,
    ):
        super().__init__(daemon=True)
        self.store = store
        self.provider = provider
        self.vector_index = vector_index
        self.batch_size = batch_size
        self.idle_sleep_seconds = idle_sleep_seconds
        self.on_progress = on_progress
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.set()

    def pause(self) -> None:
        self._pause_event.clear()
        self._emit(None, None, "paused", "embedding paused")

    def resume_work(self) -> None:
        self._pause_event.set()
        self._emit(None, None, "running", "embedding resumed")

    def run(self) -> None:
        self._emit(None, None, "running", "embedding worker started")
        while not self._stop_event.is_set():
            self._pause_event.wait()
            if self._stop_event.is_set():
                break
            jobs = self.store.next_embedding_jobs(
                model_name=self.provider.model_name,
                model_revision=self.provider.model_revision,
                embedding_dim=self.provider.dim,
                limit=self.batch_size,
            )
            if not jobs:
                self._emit(None, None, "idle", "no pending embedding jobs")
                time.sleep(self.idle_sleep_seconds)
                continue
            for image in jobs:
                if self._stop_event.is_set():
                    break
                self._pause_event.wait()
                self._process_image(image)
        self._emit(None, None, "stopped", "embedding worker stopped")

    def _process_image(self, image: ImageItem) -> None:
        self.store.mark_embedding_processing(
            image_id=image.id,
            model_name=self.provider.model_name,
            model_revision=self.provider.model_revision,
            embedding_dim=self.provider.dim,
        )
        self._emit(image.id, image.file_name, "processing", "encoding image")
        try:
            vector = self.provider.encode_image(image.file_path)
            self.store.upsert_embedding_success(
                image_id=image.id,
                model_name=self.provider.model_name,
                model_revision=self.provider.model_revision,
                vector=vector,
            )
            self.vector_index.invalidate()
            self._emit(image.id, image.file_name, "ready", "embedding ready")
        except Exception as exc:
            self.store.mark_embedding_failed(
                image_id=image.id,
                model_name=self.provider.model_name,
                model_revision=self.provider.model_revision,
                embedding_dim=self.provider.dim,
                error_message=str(exc),
            )
            self._emit(image.id, image.file_name, "failed", str(exc))

    def _emit(
        self,
        image_id: int | None,
        file_name: str | None,
        status: str,
        message: str,
    ) -> None:
        if self.on_progress is not None:
            self.on_progress(EmbeddingProgress(image_id, file_name, status, message))
