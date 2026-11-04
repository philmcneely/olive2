"""Cohere Transcribe 03-2026 local ASR engine (INT8 ONNX default + PyTorch transformers)."""

from __future__ import annotations

import asyncio
import gc
import os
import re
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from ..logging_db import logger
from ..schemas import TranscriptWord
from ..settings import DEFAULT_LOCAL_ASR_MODEL, get_effective_hf_token, resolve_paths
from .align import proportional_word_timestamps
from .audio import resample
from .model_store import (
    _COHERE_STAGED_MIN_SIZES,  # noqa: F401  # re-exported (tests import it from here)
    COHERE_INT8_ONNX_REPO,
    COHERE_PROCESSOR_REPO,
    is_cohere_staged_complete,
)

_COHERE_STAGE_LOCK = threading.Lock()


def stage_cohere_int8_model_dir(hf_token: str | None = None) -> Path:
    """Stages Cohere INT8 ONNX files into a flat non-symlink directory for ONNX Runtime 1.25+."""
    tok = hf_token if hf_token is not None else get_effective_hf_token()
    cohere_dir = Path(resolve_paths().models_dir) / "cohere-int8"
    with _COHERE_STAGE_LOCK:
        cohere_dir.mkdir(parents=True, exist_ok=True)
        # Fast path: fully staged from a previous boot — skip all HF Hub calls.
        if is_cohere_staged_complete(cohere_dir):
            return cohere_dir
        for onnx_name, data_name, fixed_name in (
            (
                "cohere-encoder.int8.onnx",
                "cohere-encoder.int8.onnx.data",
                "cohere-encoder.int8.fixed.onnx.data",
            ),
            (
                "cohere-decoder.int8.onnx",
                "cohere-decoder.int8.onnx.data",
                "cohere-decoder.int8.fixed.onnx.data",
            ),
        ):
            src_onnx = os.path.realpath(
                hf_hub_download(
                    repo_id=COHERE_INT8_ONNX_REPO,
                    filename=onnx_name,
                    token=tok,
                )
            )
            src_data = os.path.realpath(
                hf_hub_download(
                    repo_id=COHERE_INT8_ONNX_REPO,
                    filename=data_name,
                    token=tok,
                )
            )
            dst_onnx = cohere_dir / onnx_name
            if not dst_onnx.exists() or dst_onnx.is_symlink():
                dst_onnx.unlink(missing_ok=True)
                shutil.copy2(src_onnx, dst_onnx)
            for target_name in (data_name, fixed_name):
                dst_data = cohere_dir / target_name
                if not dst_data.exists() or dst_data.is_symlink():
                    dst_data.unlink(missing_ok=True)
                    try:
                        os.link(src_data, dst_data)
                    except OSError:
                        shutil.copy2(src_data, dst_data)

        src_tok = os.path.realpath(
            hf_hub_download(
                repo_id=COHERE_INT8_ONNX_REPO,
                filename="tokens.txt",
                token=tok,
            )
        )
        dst_tok = cohere_dir / "tokens.txt"
        if not dst_tok.exists() or dst_tok.is_symlink():
            dst_tok.unlink(missing_ok=True)
            shutil.copy2(src_tok, dst_tok)

    return cohere_dir


COHERE_TORCH_REPO = "CohereLabs/cohere-transcribe-03-2026"
COHERE_TORCH_UNGATED_REPO = "evewashere/cohere-transcribe-03-2026-ungated"

LANG_TOKEN_ID: dict[str, int] = {
    "ar": 28,
    "zh": 50,
    "nl": 60,
    "en": 62,
    "fr": 69,
    "de": 76,
    "it": 97,
    "ja": 98,
    "ko": 110,
    "pl": 148,
    "pt": 149,
    "ru": 155,
    "es": 169,
    "tr": 186,
    "vi": 194,
}
DECODER_START_ID = 13764
EOS_ID = 3


class CohereAsrEngine:
    """Cohere Transcribe 03-2026 local ASR engine implementing AsrBackend."""

    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        cfg = {**(config or {}), **kwargs}
        self.model_id: str = str(cfg.get("modelId") or cfg.get("model_id") or DEFAULT_LOCAL_ASR_MODEL)
        self.dtype: str = str(cfg.get("dtype") or "int8")
        self.device: str = str(cfg.get("device") or "cpu")
        self.language: str = str(cfg.get("language") or "en")
        self.num_threads: int = int(cfg.get("numThreads") or max(2, min(8, (os.cpu_count() or 4) // 2)))

        self._processor: Any = None
        self._encoder_sess: ort.InferenceSession | None = None
        self._decoder_sess: ort.InferenceSession | None = None
        self._id_to_token: dict[int, str] | None = None
        self._torch_model: Any = None

    def _ensure_int8_onnx_sync(self) -> None:
        if (
            self._processor is not None
            and self._encoder_sess is not None
            and self._decoder_sess is not None
            and self._id_to_token is not None
        ):
            return

        from transformers import AutoProcessor

        hf_token = get_effective_hf_token()
        logger.info(
            "Loading Cohere Transcribe 03-2026 INT8 ONNX model on CPU",
            {"repo": COHERE_INT8_ONNX_REPO, "threads": self.num_threads},
        )
        self._processor = AutoProcessor.from_pretrained(COHERE_PROCESSOR_REPO, token=hf_token)

        # Stage non-symlink ONNX + .data files so ONNX Runtime 1.25 path validation succeeds
        cohere_dir = stage_cohere_int8_model_dir(hf_token)
        enc_path = str(cohere_dir / "cohere-encoder.int8.onnx")
        dec_path = str(cohere_dir / "cohere-decoder.int8.onnx")
        tok_path = str(cohere_dir / "tokens.txt")

        id_to_token: dict[int, str] = {}
        with open(tok_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split(" ")
                if len(parts) >= 2:
                    try:
                        id_to_token[int(parts[-1])] = " ".join(parts[:-1])
                    except ValueError:
                        pass
        self._id_to_token = id_to_token

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._encoder_sess = ort.InferenceSession(enc_path, opts, providers=["CPUExecutionProvider"])
        self._decoder_sess = ort.InferenceSession(dec_path, opts, providers=["CPUExecutionProvider"])

    def _detokenize(self, token_ids: list[int]) -> str:
        if not self._id_to_token:
            return ""
        pieces: list[str] = []
        for tid in token_ids:
            tok = self._id_to_token.get(tid, "")
            if tok.startswith("<|") and tok.endswith("|>"):
                continue
            pieces.append(tok)
        return "".join(pieces).replace("▁", " ").strip()

    def _transcribe_chunk_int8_sync(
        self,
        chunk_16k: np.ndarray,
        language: str = "en",
        max_new_tokens: int = 256,
    ) -> str:
        self._ensure_int8_onnx_sync()
        assert self._processor is not None
        assert self._encoder_sess is not None
        assert self._decoder_sess is not None

        feat = self._processor.feature_extractor(
            chunk_16k.astype(np.float32),
            sampling_rate=16000,
            return_tensors="np",
        )["input_features"].astype(np.float32)

        # vignettes export expects (batch, 128, T) if input shape has 128 on axis 1
        enc_in = self._encoder_sess.get_inputs()[0]
        if len(enc_in.shape) == 3 and enc_in.shape[1] == 128 and feat.shape[-1] == 128:
            mel = np.transpose(feat, (0, 2, 1))
        else:
            mel = feat

        enc_out: Any = self._encoder_sess.run(
            ["n_layer_cross_k", "n_layer_cross_v"],
            {enc_in.name: mel},
        )
        cross_k = np.asarray(enc_out[0], dtype=np.float32)
        cross_v = np.asarray(enc_out[1], dtype=np.float32)
        n_layers = int(cross_k.shape[0])
        self_k = np.zeros((n_layers, 1, 8, 1024, 128), dtype=np.float32)
        self_v = np.zeros((n_layers, 1, 8, 1024, 128), dtype=np.float32)

        lang_id = LANG_TOKEN_ID.get(language.lower()[:2], 62)
        tokens = np.array([[DECODER_START_ID, lang_id]], dtype=np.int64)
        offset = np.array(0, dtype=np.int64)
        generated: list[int] = []

        for _ in range(max_new_tokens):
            dec_out: Any = self._decoder_sess.run(
                ["logits", "out_n_layer_self_k_cache", "out_n_layer_self_v_cache"],
                {
                    "tokens": tokens,
                    "in_n_layer_self_k_cache": self_k,
                    "in_n_layer_self_v_cache": self_v,
                    "n_layer_cross_k": cross_k,
                    "n_layer_cross_v": cross_v,
                    "offset": offset,
                },
            )
            logits = np.asarray(dec_out[0])
            self_k = np.asarray(dec_out[1], dtype=np.float32)
            self_v = np.asarray(dec_out[2], dtype=np.float32)
            next_token = int(np.argmax(logits[0, -1, :]))
            if next_token == EOS_ID:
                break
            generated.append(next_token)
            offset = np.array(int(offset) + tokens.shape[1], dtype=np.int64)
            tokens = np.array([[next_token]], dtype=np.int64)

        return self._detokenize(generated)

    def _transcribe_sync(
        self,
        samples: np.ndarray,
        sample_rate: int = 16000,
        language: str = "en",
        start_ms: int = 0,
        strict: bool = False,
        on_chunk: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1).astype(np.float32)
        if sample_rate != 16000:
            arr = resample(arr, sample_rate, 16000)

        duration_ms = int(round((len(arr) / 16000.0) * 1000))
        if arr.size == 0:
            return {"text": ""}

        try:
            # Chunk audio in 30s windows with 0s overlap
            chunk_samples = 30 * 16000
            texts: list[str] = []
            total_chunks = -(-len(arr) // chunk_samples)
            for idx, offset in enumerate(range(0, len(arr), chunk_samples), start=1):
                sub = arr[offset : offset + chunk_samples]
                if len(sub) >= 800:
                    t = self._transcribe_chunk_int8_sync(sub, language=language)
                    if t:
                        texts.append(t)
                if on_chunk is not None:
                    # Liveness signal for the caller's stall watchdog; must never fail ASR.
                    try:
                        on_chunk(idx, total_chunks)
                    except Exception:
                        pass
            full_text = " ".join(texts).strip()
            words = proportional_word_timestamps(full_text, start_ms, duration_ms)
            return {
                "text": full_text,
                "words": [w.model_dump(exclude_none=True) for w in words] if words else None,
            }
        except Exception as exc:
            if strict:
                raise
            logger.warn("Segment transcription encountered fallback", {"error": str(exc)})
            return {"text": f"[Audio segment ({duration_ms}ms)]"}

    async def transcribe_segment(
        self,
        samples: np.ndarray | list[float],
        sample_rate: int = 16000,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        opts = options or {}
        arr = np.asarray(samples, dtype=np.float32)
        if arr.size == 0:
            return {"text": ""}

        model_id = str(opts.get("modelId") or opts.get("model_id") or self.model_id)
        start_ms = int(opts.get("startMs") or opts.get("start_ms") or 0)
        duration_ms = int(round((len(arr) / float(sample_rate)) * 1000))

        if model_id in ("mock", "mock-asr"):
            half = duration_ms // 2
            return {
                "text": "Transcribed speech",
                "words": [
                    TranscriptWord(
                        word="Transcribed",
                        startMs=start_ms,
                        endMs=start_ms + half,
                    ),
                    TranscriptWord(
                        word="speech",
                        startMs=start_ms + half,
                        endMs=start_ms + duration_ms,
                    ),
                ],
            }

        language = str(opts.get("language") or self.language or "en")
        strict = bool(opts.get("strict"))
        on_chunk = opts.get("onChunk")
        return await asyncio.to_thread(
            self._transcribe_sync,
            arr,
            sample_rate,
            language,
            start_ms,
            strict,
            on_chunk if callable(on_chunk) else None,
        )

    transcribeSegment = transcribe_segment

    def release(self) -> None:
        if (
            self._encoder_sess is not None
            or self._decoder_sess is not None
            or self._processor is not None
            or self._torch_model is not None
        ):
            self._encoder_sess = None
            self._decoder_sess = None
            self._processor = None
            self._torch_model = None
            gc.collect()


TransformersAsrEngine = CohereAsrEngine


class MockLocalAsrEngine:
    """Deterministic Mock / In-Memory ASR Engine for unit testing and offline test suites."""

    def __init__(
        self,
        phrases: list[str] | None = None,
    ) -> None:
        self.phrases = phrases or [
            "Hello everyone.",
            "Welcome to the meeting.",
            "Let's review the items.",
        ]
        self.default_text = self.phrases[0] if self.phrases else "Transcribed speech"
        self.call_count = 0

    async def transcribe_segment(
        self,
        samples: np.ndarray | list[float],
        sample_rate: int = 16000,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        opts = options or {}
        phrase = self.phrases[self.call_count % len(self.phrases)] if self.phrases else self.default_text
        self.call_count += 1

        base_start_ms = int(opts.get("startMs") or opts.get("start_ms") or 0)
        duration_ms = int(round((len(samples) / float(sample_rate)) * 1000))
        word_list = [w for w in re.split(r"\s+", phrase.strip()) if w]
        word_dur = duration_ms // max(1, len(word_list))

        words = [
            TranscriptWord(
                word=w,
                startMs=base_start_ms + idx * word_dur,
                endMs=base_start_ms + (idx + 1) * word_dur,
            )
            for idx, w in enumerate(word_list)
        ]
        return {"text": phrase, "words": words}

    transcribeSegment = transcribe_segment

    def release(self) -> None:
        pass
