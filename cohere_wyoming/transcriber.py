"""Cohere transcription backend independent from HTTP/Wyoming transport."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from .audio import is_effectively_silent, read_audio_to_numpy
from .vad import SileroVoiceActivityDetector, VadConfig


LOGGER = logging.getLogger("cohere-wyoming.transcriber")

SUPPORTED_LANGUAGES = {
    "ar",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "it",
    "ja",
    "ko",
    "nl",
    "pl",
    "pt",
    "vi",
    "zh",
}

LANGUAGE_ALIASES = {
    "arabic": "ar",
    "chinese": "zh",
    "dutch": "nl",
    "english": "en",
    "french": "fr",
    "german": "de",
    "greek": "el",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "polish": "pl",
    "portuguese": "pt",
    "spanish": "es",
    "vietnamese": "vi",
}


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration: float
    processing_time: float

    def asdict(self) -> dict:
        return asdict(self)


class CohereTranscriber:
    """Lazy-loading wrapper around Cohere Transcribe."""

    def __init__(
        self,
        *,
        model_name: str = "CohereLabs/cohere-transcribe-03-2026",
        default_language: str = "en",
        prefer_device: Optional[str] = None,
        vad_config: Optional[VadConfig] = None,
    ) -> None:
        self.model_name = model_name
        self.default_language = self.resolve_language(default_language)
        self.prefer_device = prefer_device
        self.backend = "native"
        self.model = None
        self.processor = None
        self.device = None
        self.vad_detector = SileroVoiceActivityDetector(vad_config or VadConfig.from_env())

    def resolve_language(self, language: Optional[str]) -> str:
        """Resolve a requested language or fall back to the configured default."""
        if language is None:
            return self.default_language

        resolved = language.strip().lower()
        if resolved == "auto":
            return self.default_language

        resolved = LANGUAGE_ALIASES.get(resolved, resolved)
        if resolved not in SUPPORTED_LANGUAGES:
            LOGGER.warning(
                "Language '%s' not supported by Cohere Transcribe. Falling back to '%s'.",
                resolved,
                self.default_language,
            )
            return self.default_language

        return resolved

    def set_default_language(self, language: str) -> None:
        self.default_language = self.resolve_language(language)

    def set_model_name(self, model_name: str) -> None:
        self.model_name = model_name

    def set_vad_config(self, vad_config: VadConfig) -> None:
        self.vad_detector.update_config(vad_config)

    def _select_device(self) -> torch.device:
        if self.prefer_device == "cpu":
            LOGGER.info("Using forced CPU device")
            return torch.device("cpu")

        if self.prefer_device and self.prefer_device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but CUDA is not available")
            LOGGER.info("Using requested CUDA device: %s", self.prefer_device)
            return torch.device(self.prefer_device)

        if torch.cuda.is_available():
            LOGGER.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
            return torch.device("cuda:0")

        LOGGER.info("Using CPU device")
        return torch.device("cpu")

    def load_model_artifacts(self, model_name: str):
        """Load processor/model preferring local Hugging Face cache first."""
        local_only_kwargs = {"trust_remote_code": False, "local_files_only": True}
        remote_allowed_kwargs = {"trust_remote_code": False}

        try:
            LOGGER.info("Trying to load model artifacts from local cache first")
            processor = AutoProcessor.from_pretrained(model_name, **local_only_kwargs)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, **local_only_kwargs)
            LOGGER.info("Loaded model artifacts from local cache")
            return processor, model
        except Exception as local_error:
            LOGGER.info(
                "Local cache load failed, retrying with network access: %s",
                local_error,
            )

        processor = AutoProcessor.from_pretrained(model_name, **remote_allowed_kwargs)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, **remote_allowed_kwargs)
        LOGGER.info("Loaded model artifacts with network access")
        return processor, model

    def load(self, model_name: Optional[str] = None) -> None:
        """Load the model and only swap instance state after success."""
        if model_name:
            self.model_name = model_name

        LOGGER.info("Loading model: %s (backend=%s)", self.model_name, self.backend)
        start_time = time.time()
        next_processor, next_model = self.load_model_artifacts(self.model_name)

        if torch.cuda.is_available() and self.prefer_device != "cpu":
            gpu_count = torch.cuda.device_count()
            primary_gpu_name = torch.cuda.get_device_name(0)
            LOGGER.info(
                "CUDA is available (%s visible GPU%s). Using primary device cuda:0 (%s).",
                gpu_count,
                "" if gpu_count == 1 else "s",
                primary_gpu_name,
            )
            try:
                next_device = self._select_device()
                next_model = next_model.to(next_device)
            except (RuntimeError, torch.OutOfMemoryError) as error:
                LOGGER.warning(
                    "Falling back to CPU because loading the model on %s failed: %s",
                    next_device,
                    error,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                next_device = torch.device("cpu")
                next_model = next_model.to(next_device)
        else:
            next_device = self._select_device()
            next_model = next_model.to(next_device)

        next_model.eval()

        self.processor = next_processor
        self.model = next_model
        self.device = next_device

        elapsed = time.time() - start_time
        LOGGER.info("Model loaded in %.1fs using backend=%s", elapsed, self.backend)

    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def health_payload(self) -> dict:
        return {
            "status": "ok" if self.is_loaded() else "loading",
            "ready": self.is_loaded(),
            "model": self.model_name or None,
            "device": str(self.device) if self.device is not None else None,
            "backend": self.backend,
            "vad": self.vad_detector.status_payload(),
        }

    def transcribe_pcm(
        self,
        audio_data: np.ndarray,
        *,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        temperature: float = 0.0,
    ) -> TranscriptionResult:
        """Transcribe normalized PCM audio."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        resolved_language = self.resolve_language(language)
        duration_s = len(audio_data) / sample_rate

        if is_effectively_silent(audio_data):
            LOGGER.info("No speech detected above silence threshold; returning empty transcription")
            return TranscriptionResult(
                text="",
                language=resolved_language,
                duration=round(duration_s, 2),
                processing_time=0.0,
            )

        vad_decision = self.vad_detector.detect_speech(audio_data, sample_rate=sample_rate)
        if not vad_decision.has_speech:
            LOGGER.info(
                "Silero VAD rejected audio as non-speech (reason=%s, segments=%s, total_ms=%s, max_ms=%s, speech_rms=%.6f, noise_rms=%.6f, snr=%.3f)",
                vad_decision.reason,
                vad_decision.speech_segments,
                vad_decision.total_speech_ms,
                vad_decision.max_segment_ms,
                vad_decision.speech_rms,
                vad_decision.noise_rms,
                vad_decision.speech_to_noise_ratio,
            )
            return TranscriptionResult(
                text="",
                language=resolved_language,
                duration=round(duration_s, 2),
                processing_time=0.0,
            )

        start_time = time.time()

        inputs = self.processor(
            audio_data,
            sampling_rate=sample_rate,
            return_tensors="pt",
            language=resolved_language,
        )
        audio_chunk_index = inputs.get("audio_chunk_index")
        inputs.to(self.model.device, dtype=self.model.dtype)

        generate_kwargs = {"max_new_tokens": 256, "do_sample": False}
        if temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        if audio_chunk_index is not None:
            text = self.processor.decode(
                outputs,
                skip_special_tokens=True,
                audio_chunk_index=audio_chunk_index,
                language=resolved_language,
            )
        else:
            text = self.processor.decode(outputs, skip_special_tokens=True)

        if isinstance(text, list):
            text = text[0]

        elapsed = time.time() - start_time
        rtfx = duration_s / elapsed if elapsed > 0 else 0
        LOGGER.info(
            "Transcribed %.1fs audio in %.1fs (RTFx: %.1fx) lang=%s backend=%s",
            duration_s,
            elapsed,
            rtfx,
            resolved_language,
            self.backend,
        )

        return TranscriptionResult(
            text=text.strip() if isinstance(text, str) else str(text).strip(),
            language=resolved_language,
            duration=round(duration_s, 2),
            processing_time=round(elapsed, 2),
        )

    def transcribe_file_bytes(
        self,
        file_bytes: bytes,
        *,
        filename: str = "audio",
        language: Optional[str] = None,
        temperature: float = 0.0,
    ) -> TranscriptionResult:
        audio_data, sample_rate = read_audio_to_numpy(file_bytes, filename)
        return self.transcribe_pcm(
            audio_data,
            sample_rate=sample_rate,
            language=language,
            temperature=temperature,
        )
