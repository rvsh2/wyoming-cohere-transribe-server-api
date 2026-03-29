import asyncio
import io
import json
import sys
import unittest
import wave
from unittest.mock import patch

from fastapi import HTTPException
import numpy as np

import server


def make_wav_bytes(
    duration_s: float = 0.1,
    sample_rate: int = 16000,
    channels: int = 1,
) -> bytes:
    """Create a tiny valid WAV fixture for endpoint tests."""
    frames = int(duration_s * sample_rate)
    t = np.arange(frames, dtype=np.float32) / sample_rate
    signal = 0.25 * np.sin(2 * np.pi * 440.0 * t)
    pcm = np.clip(signal * 32767.0, -32768, 32767).astype(np.int16)
    if channels > 1:
        pcm = np.repeat(pcm[:, None], channels, axis=1)
    audio = io.BytesIO()
    with wave.open(audio, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return audio.getvalue()


class FakeUploadFile:
    """Minimal UploadFile-like object for direct endpoint testing."""

    def __init__(self, content: bytes, filename: str = "test.wav") -> None:
        self._content = content
        self.filename = filename

    async def read(self) -> bytes:
        return self._content


def make_upload_file(content: bytes, filename: str = "test.wav") -> FakeUploadFile:
    """Build a minimal upload object matching handler expectations."""
    return FakeUploadFile(content=content, filename=filename)


class CohereTranscribeApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.audio_bytes = make_wav_bytes()
        self.model_patcher = patch.object(server, "model", object())
        self.processor_patcher = patch.object(server, "processor", object())
        self.model_patcher.start()
        self.processor_patcher.start()

    def tearDown(self) -> None:
        self.processor_patcher.stop()
        self.model_patcher.stop()

    def run_async(self, coro):
        return asyncio.run(coro)

    def decode_json_response(self, response) -> dict:
        return json.loads(response.body.decode("utf-8"))

    def call_inference(self, **overrides):
        params = {
            "file": make_upload_file(self.audio_bytes),
            "temperature": 0.0,
            "temperature_inc": 0.2,
            "response_format": "json",
            "language": None,
            "encode": True,
            "no_timestamps": False,
            "prompt": None,
            "translate": False,
        }
        params.update(overrides)
        return self.run_async(server.inference(**params))

    def call_openai_transcriptions(self, **overrides):
        params = {
            "file": make_upload_file(self.audio_bytes),
            "model_name": None,
            "language": None,
            "response_format": "json",
            "temperature": 0.0,
            "prompt": None,
        }
        params.update(overrides)
        return self.run_async(server.openai_transcriptions(**params))

    def test_index_page_includes_compatibility_notes(self):
        response = self.run_async(server.index())

        self.assertIn("Compatibility Notes", response)
        self.assertIn("/inference", response)

    def test_inference_returns_json_response(self):
        with patch.object(server, "transcribe_audio", return_value={
            "text": "hello world",
            "language": "en",
            "duration": 0.1,
            "processing_time": 0.01,
        }):
            response = self.call_inference()

        self.assertEqual(self.decode_json_response(response), {"text": "hello world"})

    def test_inference_supports_text_verbose_json_srt_and_vtt(self):
        mocked_result = {
            "text": "hello world",
            "language": "en",
            "duration": 1.25,
            "processing_time": 0.02,
        }

        with patch.object(server, "transcribe_audio", return_value=mocked_result):
            text_response = self.call_inference(response_format="text")
            verbose_response = self.call_inference(response_format="verbose_json")
            srt_response = self.call_inference(response_format="srt")
            vtt_response = self.call_inference(response_format="vtt")

        self.assertEqual(text_response.body.decode("utf-8"), "hello world\n")
        self.assertEqual(
            self.decode_json_response(verbose_response)["segments"][0]["end"], 1.25
        )
        self.assertIn("00:00:01,250", srt_response.body.decode("utf-8"))
        self.assertTrue(vtt_response.body.decode("utf-8").startswith("WEBVTT"))

    def test_openai_endpoint_returns_verbose_json(self):
        with patch.object(server, "transcribe_audio", return_value={
            "text": "openai shape",
            "language": "pl",
            "duration": 0.2,
            "processing_time": 0.03,
        }):
            response = self.call_openai_transcriptions(
                model_name="ignored-model",
                response_format="verbose_json",
            )

        payload = self.decode_json_response(response)
        self.assertEqual(payload["text"], "openai shape")
        self.assertEqual(payload["segments"][0]["start"], 0.0)

    def test_load_endpoint_returns_ok_on_success(self):
        with patch.object(server, "load_model") as load_model:
            response = self.run_async(server.load(model_path="CohereLabs/mock-model"))

        self.assertEqual(
            self.decode_json_response(response),
            {"status": "ok", "model": "CohereLabs/mock-model"},
        )
        load_model.assert_called_once()

    def test_empty_upload_returns_400(self):
        with self.assertRaises(HTTPException) as ctx:
            self.run_async(
                server.inference(
                    file=make_upload_file(b"", "empty.wav"),
                    temperature=0.0,
                    temperature_inc=0.2,
                    response_format="json",
                    language=None,
                    encode=True,
                    no_timestamps=False,
                    prompt=None,
                    translate=False,
                )
            )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, "Empty audio file")

    def test_invalid_audio_returns_400(self):
        with patch.object(server, "read_audio_to_numpy", side_effect=ValueError("bad audio")):
            with self.assertRaises(HTTPException) as ctx:
                self.call_inference()

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, "bad audio")

    def test_unsupported_language_falls_back_to_default_language(self):
        with patch.object(server, "transcribe_audio", side_effect=lambda audio, sr, language, temperature: {
            "text": f"lang={language}",
            "language": language,
            "duration": 0.1,
            "processing_time": 0.01,
        }):
            response = self.call_inference(language="xx")

        self.assertEqual(self.decode_json_response(response)["text"], "lang=en")

    def test_missing_model_returns_503(self):
        with patch.object(server, "model", None), patch.object(server, "processor", None):
            with self.assertRaises(HTTPException) as ctx:
                self.call_inference()

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail, "Model not loaded")

    def test_transcription_failure_returns_500(self):
        with patch.object(server, "transcribe_audio", side_effect=RuntimeError("boom")):
            with self.assertRaises(HTTPException) as ctx:
                self.call_inference()

        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("Transcription failed: boom", ctx.exception.detail)

    def test_load_failure_returns_500(self):
        with patch.object(server, "load_model", side_effect=RuntimeError("cannot load")):
            with self.assertRaises(HTTPException) as ctx:
                self.run_async(server.load(model_path="broken-model"))

        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("Failed to load model: cannot load", ctx.exception.detail)

    def test_compatibility_only_parameters_do_not_break_inference(self):
        with patch.object(server, "transcribe_audio", return_value={
            "text": "compat ok",
            "language": "en",
            "duration": 0.1,
            "processing_time": 0.01,
        }):
            response = self.call_inference(
                temperature_inc=0.5,
                prompt="hello",
                encode=False,
                no_timestamps=True,
                translate=True,
            )

        self.assertEqual(self.decode_json_response(response)["text"], "compat ok")

    def test_audio_preprocessing_resamples_mono_and_stereo_wav(self):
        stereo_44k = make_wav_bytes(sample_rate=44100, channels=2)
        mono_22k = make_wav_bytes(sample_rate=22050, channels=1)

        stereo_audio, stereo_sr = server.read_audio_to_numpy(stereo_44k, "stereo.wav")
        mono_audio, mono_sr = server.read_audio_to_numpy(mono_22k, "mono.wav")

        self.assertEqual(stereo_sr, 16000)
        self.assertEqual(mono_sr, 16000)
        self.assertEqual(stereo_audio.ndim, 1)
        self.assertEqual(mono_audio.ndim, 1)
        self.assertGreater(len(stereo_audio), 0)
        self.assertGreater(len(mono_audio), 0)

    def test_cli_no_longer_exposes_trust_remote_code_flags(self):
        argv = ["server.py"]
        with patch.object(sys, "argv", argv):
            args = server.parse_args()

        self.assertFalse(hasattr(args, "trust_remote_code"))
        self.assertFalse(hasattr(args, "no_trust_remote_code"))


if __name__ == "__main__":
    unittest.main()
