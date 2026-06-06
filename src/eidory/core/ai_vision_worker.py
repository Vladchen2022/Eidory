from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from eidory.core.ai_vision import AI_VISION_PROMPT_VERSION, AIVisionAnalysis, AIVisionProvider
from eidory.core.metadata_store import MetadataStore
from eidory.models import ImageItem


@dataclass(frozen=True)
class AIVisionProgress:
    image_id: int | None
    file_name: str | None
    status: str
    message: str


AIVisionProgressCallback = Callable[[AIVisionProgress], None]


class AIVisionWorker(threading.Thread):
    def __init__(
        self,
        *,
        store: MetadataStore,
        provider: AIVisionProvider,
        idle_sleep_seconds: float = 2.0,
        on_progress: AIVisionProgressCallback | None = None,
    ):
        super().__init__(daemon=True)
        self.store = store
        self.provider = provider
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
        self._emit(None, None, "paused", "AI vision paused")

    def resume_work(self) -> None:
        self._pause_event.set()
        self._emit(None, None, "running", "AI vision resumed")

    def run(self) -> None:
        model_name = self.provider.resolved_model_name()
        self._emit(None, None, "running", "AI vision worker started")
        while not self._stop_event.is_set():
            self._pause_event.wait()
            if self._stop_event.is_set():
                break
            jobs = self.store.next_ai_vision_jobs(
                provider_name=self.provider.provider_name,
                model_name=model_name,
                prompt_version=AI_VISION_PROMPT_VERSION,
                limit=1,
            )
            if not jobs:
                self._emit(None, None, "idle", "no pending AI vision jobs")
                time.sleep(self.idle_sleep_seconds)
                continue
            for image in jobs:
                if self._stop_event.is_set():
                    break
                self._pause_event.wait()
                self._process_image(image, model_name=model_name)
        self._emit(None, None, "stopped", "AI vision worker stopped")

    def _process_image(self, image: ImageItem, *, model_name: str) -> None:
        if image.is_missing or not Path(image.file_path).is_file():
            self.store.mark_image_missing(image.id)
            message = f"源文件不存在：{image.file_path}"
            self.store.mark_ai_vision_failed(
                image_id=image.id,
                provider_name=self.provider.provider_name,
                model_name=model_name,
                prompt_version=AI_VISION_PROMPT_VERSION,
                error_message=message,
            )
            self._emit(image.id, image.file_name, "failed", message)
            return
        self.store.mark_ai_vision_processing(
            image_id=image.id,
            provider_name=self.provider.provider_name,
            model_name=model_name,
            prompt_version=AI_VISION_PROMPT_VERSION,
        )
        self._emit(image.id, image.file_name, "processing", "analyzing image")
        try:
            analysis = self._analyze_with_retries(image)
            self.store.upsert_ai_vision_success(
                image_id=image.id,
                provider_name=self.provider.provider_name,
                model_name=model_name,
                prompt_version=AI_VISION_PROMPT_VERSION,
                analysis=analysis,
                source_modified_time_ns=image.modified_time_ns,
            )
            self._emit(image.id, image.file_name, "ready", "AI vision ready")
        except Exception as exc:
            self.store.mark_ai_vision_failed(
                image_id=image.id,
                provider_name=self.provider.provider_name,
                model_name=model_name,
                prompt_version=AI_VISION_PROMPT_VERSION,
                error_message=str(exc),
            )
            self._emit(image.id, image.file_name, "failed", str(exc))

    def _analyze_with_retries(self, image: ImageItem) -> AIVisionAnalysis:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return self.provider.analyze_image(image.file_path)
            except Exception as exc:
                last_error = exc
                if attempt >= 2 or not self._is_retryable_error(exc):
                    break
                delay = 1.5 * (attempt + 1)
                self._emit(
                    image.id,
                    image.file_name,
                    "retrying",
                    f"AI vision retrying after transient error: {exc}",
                )
                time.sleep(delay)
        if last_error is None:
            raise RuntimeError("AI vision failed without an error")
        raise last_error

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        message = str(exc)
        retryable_fragments = (
            "Connection aborted",
            "Connection reset",
            "RemoteDisconnected",
            "Read timed out",
            "请求失败",
            "视觉模型 JSON 无效",
            "视觉模型返回了空内容",
        )
        return any(fragment in message for fragment in retryable_fragments)

    def _emit(
        self,
        image_id: int | None,
        file_name: str | None,
        status: str,
        message: str,
    ) -> None:
        if self.on_progress is not None:
            self.on_progress(AIVisionProgress(image_id, file_name, status, message))
