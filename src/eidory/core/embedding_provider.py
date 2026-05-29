from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_revision(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def encode_image(self, image_path: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        raise NotImplementedError


class JinaClipV2Provider(EmbeddingProvider):
    def __init__(
        self,
        *,
        model_name: str = "jinaai/jina-clip-v2",
        model_revision: str = "main",
        dim: int = 512,
        prefer_mps: bool = True,
    ):
        self._model_name = model_name
        self._model_revision = model_revision
        self._dim = dim
        self._prefer_mps = prefer_mps
        self._model: Any | None = None
        self._lock = threading.RLock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_revision(self) -> str:
        return self._model_revision

    @property
    def dim(self) -> int:
        return self._dim

    def encode_image(self, image_path: str) -> np.ndarray:
        with self._lock:
            model = self._load_model()
            vector = model.encode([image_path], normalize_embeddings=True)[0]
        return self._as_vector(vector)

    def encode_text(self, text: str) -> np.ndarray:
        with self._lock:
            model = self._load_model()
            try:
                vector = model.encode(
                    [text],
                    prompt_name="retrieval.query",
                    normalize_embeddings=True,
                )[0]
            except TypeError:
                vector = model.encode([text], normalize_embeddings=True)[0]
        return self._as_vector(vector)

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        first_device = "cpu"
        if self._prefer_mps:
            try:
                import torch

                if torch.backends.mps.is_available():
                    first_device = "mps"
            except Exception:
                first_device = "cpu"

        try:
            self._model = SentenceTransformer(
                self._model_name,
                revision=self._model_revision,
                trust_remote_code=True,
                truncate_dim=self._dim,
                device=first_device,
            )
        except Exception:
            if first_device == "cpu":
                raise
            self._model = SentenceTransformer(
                self._model_name,
                revision=self._model_revision,
                trust_remote_code=True,
                truncate_dim=self._dim,
                device="cpu",
            )
        return self._model

    def _as_vector(self, vector: Any) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("embedding provider returned a non-vector result")
        if arr.shape[0] != self._dim:
            raise ValueError(f"expected dim {self._dim}, got {arr.shape[0]}")
        norm = np.linalg.norm(arr)
        if norm == 0:
            raise ValueError("embedding provider returned a zero vector")
        return arr / norm
