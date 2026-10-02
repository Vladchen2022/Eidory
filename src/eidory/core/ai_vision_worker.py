from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from eidory.core.ai_vision import (
    AI_VISION_ANALYSIS_SCENE,
    AIVisionAnalysis,
    AIVisionProvider,
    ai_vision_prompt_version,
    normalize_ai_vision_analysis_type,
)
from eidory.core.metadata_store import MetadataStore
from eidory.models import ImageItem


@dataclass(frozen=True)
class AIVisionProgress:
    image_id: int | None
    file_name: str | None
    status: str
    message: str
    analysis_type: str = AI_VISION_ANALYSIS_SCENE


AIVisionProgressCallback = Callable[[AIVisionProgress], None]


class AIVisionWorker(threading.Thread):
    def __init__(
        self,
        *,
        store: MetadataStore,
        provider: AIVisionProvider,
        analysis_type: str = AI_VISION_ANALYSIS_SCENE,
        idle_sleep_seconds: float = 2.0,
        on_progress: AIVisionProgressCallback | None = None,
    ):
        super().__init__(daemon=True)
        self.store = store
        self.provider = provider
        self.analysis_type = normalize_ai_vision_analysis_type(analysis_type)
        self.prompt_version = ai_vision_prompt_version(self.analysis_type)
        self.idle_sleep_seconds = idle_sleep_seconds
        self.on_progress = on_progress
        self._stop_event = threading.Event()
        self.provider.cancel_event = self._stop_event
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._generation_ready = threading.Event()
        self._generation_ready.set()

    def set_generation_hold(self, hold: bool) -> None:
        if hold:
            self._generation_ready.clear()
        else:
            self._generation_ready.set()

    def _wait_ready(self) -> bool:
        while not self._stop_event.is_set():
            if self._pause_event.wait(.1) and self._generation_ready.wait(.1):
                return not self._stop_event.is_set()
        return False

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.set()
        self._generation_ready.set()

    def pause(self) -> None:
        self._pause_event.clear()
        self._emit(None, None, "paused", "AI vision paused")

    def resume_work(self) -> None:
        self._pause_event.set()
        self._emit(None, None, "running", "AI vision resumed")

    def run(self) -> None:
        try:
            model_name = self.provider.resolved_model_name()
        except Exception:
            if self._stop_event.is_set():
                return
            raise
        self._emit(None, None, "running", "AI vision worker started")
        while not self._stop_event.is_set():
            if not self._wait_ready():
                break
            jobs = self.store.next_ai_vision_jobs(
                provider_name=self.provider.provider_name,
                model_name=model_name,
                prompt_version=self.prompt_version,
                analysis_type=self.analysis_type,
                limit=1,
            )
            if not jobs:
                self._stop_event.wait(self.idle_sleep_seconds)
                continue
            for image in jobs:
                if self._stop_event.is_set():
                    break
                if not self._wait_ready():
                    break
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
                prompt_version=self.prompt_version,
                analysis_type=self.analysis_type,
                error_message=message,
            )
            self._emit(image.id, image.file_name, "failed", message)
            return
        self.store.mark_ai_vision_processing(
            image_id=image.id,
            provider_name=self.provider.provider_name,
            model_name=model_name,
            prompt_version=self.prompt_version,
            analysis_type=self.analysis_type,
        )
        self._emit(image.id, image.file_name, "processing", "analyzing image")
        try:
            analysis = self._analyze_with_retries(image)
            if self._stop_event.is_set():
                self.store.mark_ai_vision_interrupted(image.id, analysis_type=self.analysis_type)
                return
            self.store.upsert_ai_vision_success(
                image_id=image.id,
                provider_name=self.provider.provider_name,
                model_name=model_name,
                prompt_version=self.prompt_version,
                analysis_type=self.analysis_type,
                analysis=analysis,
                source_modified_time_ns=image.modified_time_ns,
            )
            self._emit(image.id, image.file_name, "ready", "AI vision ready")
        except Exception as exc:
            if self._stop_event.is_set():
                self.store.mark_ai_vision_interrupted(image.id, analysis_type=self.analysis_type)
                return
            self.store.mark_ai_vision_failed(
                image_id=image.id,
                provider_name=self.provider.provider_name,
                model_name=model_name,
                prompt_version=self.prompt_version,
                analysis_type=self.analysis_type,
                error_message=str(exc),
            )
            self._emit(image.id, image.file_name, "failed", str(exc))

    def _analyze_with_retries(self, image: ImageItem) -> AIVisionAnalysis:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return self.provider.analyze_image(
                    image.file_path,
                    analysis_type=self.analysis_type,
                )
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
                if self._stop_event.wait(delay):
                    break
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
            self.on_progress(
                AIVisionProgress(
                    image_id,
                    file_name,
                    status,
                    message,
                    self.analysis_type,
                )
            )
