"""WeSpeaker ResNet293-LM speaker embedding extractor."""

from __future__ import annotations

import asyncio
import gc
import os
import threading
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import torchaudio.compliance.kaldi as kaldi
from huggingface_hub import hf_hub_download

from ..logging_db import logger
from ..settings import get_effective_hf_token
from .audio import resample
from .model_store import WESPEAKER_FILENAME, WESPEAKER_REPO_ID
from .protocols import VoiceprintVector
from .scoring import normalize_vector

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

WESPEAKER_MODEL_TAG = f"{WESPEAKER_REPO_ID}/{WESPEAKER_FILENAME}"

_EMBED_LOCK = threading.Lock()
_CACHED_SESSIONS: dict[tuple[str, int], tuple[ort.InferenceSession, str, int]] = {}
_CACHED_DEFAULT_PATH: str | None = None


def _default_embedding_threads() -> int:
    env_val = os.environ.get("OLIVE_EMBEDDING_THREADS")
    if env_val:
        try:
            parsed = int(env_val)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    cores = os.cpu_count() or 4
    return max(2, min(8, cores // 2))


class WeSpeakerEmbeddingExtractor:
    """Extracts 256-dim L2-normalized speaker embeddings using WeSpeaker ResNet293-LM ONNX."""

    def __init__(self, options: dict[str, Any] | int | None = None, **kwargs: Any) -> None:
        if isinstance(options, int):
            self.model_path: str | None = None
            self.num_threads: int = _default_embedding_threads()
            self._dim: int = options
        else:
            opts = {**(options or {}), **kwargs}
            self.model_path = opts.get("modelPath") or opts.get("model_path")
            self.num_threads = int(
                opts.get("numThreads") or opts.get("num_threads") or _default_embedding_threads()
            )
            self._dim = 256

        self._session: ort.InferenceSession | None = None
        self._input_name: str = "feats"

    @property
    def dim(self) -> int:
        return self._dim

    def _ensure_session_sync(self) -> ort.InferenceSession:
        global _CACHED_DEFAULT_PATH
        if self._session is not None:
            return self._session

        with _EMBED_LOCK:
            if self._session is not None:
                return self._session

            path = self.model_path or _CACHED_DEFAULT_PATH
            if not path:
                tok = get_effective_hf_token()
                try:
                    path = hf_hub_download(
                        repo_id=WESPEAKER_REPO_ID,
                        filename=WESPEAKER_FILENAME,
                        token=tok,
                    )
                except Exception:
                    path = hf_hub_download(
                        repo_id=WESPEAKER_REPO_ID,
                        filename=WESPEAKER_FILENAME,
                        token=None,
                    )
                _CACHED_DEFAULT_PATH = path
            self.model_path = path

            cache_key = (path, self.num_threads)
            if cache_key in _CACHED_SESSIONS:
                sess, in_name, dim = _CACHED_SESSIONS[cache_key]
                self._session = sess
                self._input_name = in_name
                self._dim = dim
                return sess

            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = self.num_threads
            sess_opts.inter_op_num_threads = 1
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            sess = ort.InferenceSession(
                path,
                sess_opts,
                providers=["CPUExecutionProvider"],
            )
            in_name = sess.get_inputs()[0].name
            out_shape = sess.get_outputs()[0].shape
            dim = int(out_shape[1]) if (len(out_shape) >= 2 and isinstance(out_shape[1], int)) else 256

            _CACHED_SESSIONS[cache_key] = (sess, in_name, dim)
            self._session = sess
            self._input_name = in_name
            self._dim = dim
            return sess

    def _extract_sync(self, samples: np.ndarray, sample_rate: int = 16000) -> VoiceprintVector:
        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1).astype(np.float32)
        if sample_rate != 16000:
            arr = resample(arr, sample_rate, 16000)
        if arr.size < 400:
            return [0.0] * self._dim

        try:
            sess = self._ensure_session_sync()
            waveform = torch.from_numpy(arr).unsqueeze(0) * 32768.0
            feats = kaldi.fbank(
                waveform,
                num_mel_bins=80,
                frame_length=25.0,
                frame_shift=10.0,
                dither=0.0,
                sample_frequency=16000.0,
                window_type="hamming",
                use_energy=False,
            )
            if feats.shape[0] == 0:
                return [0.0] * self._dim
            feats = feats - feats.mean(dim=0, keepdim=True)
            feat_np = feats.unsqueeze(0).numpy().astype(np.float32)
            ort_out: Any = sess.run(None, {self._input_name: feat_np})
            raw_emb = np.asarray(ort_out[0])[0]
            return normalize_vector(raw_emb)
        except Exception as exc:
            # Keep the zero-vector contract (callers treat it as "no voiceprint"),
            # but never fail silently: a zero vector scores 0.0 against everyone.
            logger.error(
                "WeSpeaker embedding failed; returning zero vector",
                {
                    "category": "transcription",
                    "error": f"{type(exc).__name__}: {exc}",
                    "samples": int(arr.size),
                    "modelPath": self.model_path,
                },
            )
            return [0.0] * self._dim

    async def extract(
        self,
        samples: np.ndarray | list[float],
        sample_rate: int = 16000,
    ) -> VoiceprintVector:
        arr = np.asarray(samples, dtype=np.float32)
        if arr.size == 0:
            return [0.0] * self._dim
        return await asyncio.to_thread(self._extract_sync, arr, sample_rate)

    def release(self, clear_cache: bool = False) -> None:
        with _EMBED_LOCK:
            self._session = None
            if clear_cache:
                _CACHED_SESSIONS.clear()
                gc.collect()


SherpaSpeakerEmbeddingExtractor = WeSpeakerEmbeddingExtractor
AcousticFeatureEmbeddingExtractor = WeSpeakerEmbeddingExtractor
