#!/usr/bin/env python3
"""
Cohere Transcribe API Server
=============================
A whisper.cpp-compatible HTTP server that uses CohereLabs/cohere-transcribe-03-2026
for speech recognition. Exposes the same API as whisper.cpp server so existing
clients (curl scripts, benchmarks, etc.) work without modification.

Endpoints:
  POST /inference                  — whisper.cpp compatible transcription
  POST /v1/audio/transcriptions   — OpenAI / vLLM compatible transcription
  POST /load                      — hot-reload model
  GET  /                          — server info page

Usage:
  python server.py --host 0.0.0.0 --port 8080
  python server.py --model CohereLabs/cohere-transcribe-03-2026 --language pl
"""

import argparse
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cohere-transcribe-server")

# ---------------------------------------------------------------------------
# Globals – filled on startup
# ---------------------------------------------------------------------------
model = None
processor = None
device = None
model_id: str = ""
default_language: str = "en"
model_backend: str = "native"
IGNORED_WHISPER_PARAMS = ("temperature_inc", "prompt", "encode", "no_timestamps")

# Cohere Transcribe supported languages
SUPPORTED_LANGUAGES = {
    "en", "fr", "de", "it", "es", "pt", "el", "nl", "pl",
    "zh", "ja", "ko", "vi", "ar",
}

# Whisper language codes mapping (whisper uses some different codes)
LANGUAGE_ALIASES = {
    "english": "en", "french": "fr", "german": "de", "italian": "it",
    "spanish": "es", "portuguese": "pt", "greek": "el", "dutch": "nl",
    "polish": "pl", "chinese": "zh", "japanese": "ja", "korean": "ko",
    "vietnamese": "vi", "arabic": "ar",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_name: str):
    """Load the Cohere Transcribe model and processor."""
    global model, processor, device, model_id, model_backend

    logger.info(f"Loading model: {model_name} (backend=native)")
    start = time.time()

    # Determine device
    if torch.cuda.is_available():
        next_device = torch.device("cuda:0")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        next_device = torch.device("cpu")
        logger.info("Using CPU device")

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    next_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=False)
    next_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, trust_remote_code=False
    ).to(next_device)
    next_model.eval()

    model_id = model_name
    model_backend = "native"
    device = next_device
    processor = next_processor
    model = next_model

    elapsed = time.time() - start
    logger.info(f"Model loaded in {elapsed:.1f}s using backend={model_backend}")


# ---------------------------------------------------------------------------
# Audio processing helpers
# ---------------------------------------------------------------------------
def read_audio_to_numpy(file_bytes: bytes, filename: str = "audio") -> tuple[np.ndarray, int]:
    """
    Read audio bytes into a numpy array resampled to 16kHz mono.
    Supports WAV, MP3, FLAC, OGG, etc. via soundfile and librosa.
    """
    try:
        # Try soundfile first (fast, handles WAV/FLAC/OGG)
        audio_io = io.BytesIO(file_bytes)
        audio_data, sr = sf.read(audio_io, dtype="float32")

        # Convert to mono if stereo
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            sr = 16000

        return audio_data, sr
    except Exception:
        pass

    try:
        # Fallback to librosa (handles MP3 and more formats)
        audio_io = io.BytesIO(file_bytes)
        audio_data, sr = librosa.load(audio_io, sr=16000, mono=True)
        return audio_data, sr
    except Exception as e:
        raise ValueError(
            f"Could not read audio file '{filename}'. "
            f"Supported formats: WAV, MP3, FLAC, OGG. Error: {e}"
        )


async def read_upload_audio(file: UploadFile) -> tuple[np.ndarray, int]:
    """Read and decode an uploaded audio file into a mono 16kHz numpy array."""
    try:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")

        return read_audio_to_numpy(file_bytes, file.filename or "audio")
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading audio: {e}")


def resolve_language(lang: Optional[str]) -> str:
    """Resolve language code, handling whisper.cpp aliases."""
    if lang is None:
        return default_language

    lang = lang.strip().lower()

    if lang == "auto":
        return default_language

    # Check aliases (e.g. "english" -> "en")
    if lang in LANGUAGE_ALIASES:
        lang = LANGUAGE_ALIASES[lang]

    if lang not in SUPPORTED_LANGUAGES:
        logger.warning(
            f"Language '{lang}' not supported by Cohere Transcribe. "
            f"Falling back to '{default_language}'. "
            f"Supported: {sorted(SUPPORTED_LANGUAGES)}"
        )
        return default_language

    return lang


def ensure_model_loaded() -> None:
    """Raise a 503 if the model is not available yet."""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")


def log_ignored_whisper_options(
    *,
    temperature_inc: float,
    prompt: Optional[str],
    encode: Optional[bool],
    no_timestamps: Optional[bool],
    translate: Optional[bool],
) -> None:
    """Log compatibility-only whisper.cpp parameters that do not affect decoding."""
    ignored: list[str] = []

    if temperature_inc != 0.2:
        ignored.append("temperature_inc")
    if prompt:
        ignored.append("prompt")
    if encode is not None and encode is not True:
        ignored.append("encode")
    if no_timestamps:
        ignored.append("no_timestamps")

    if translate:
        logger.warning("Translate mode is not supported by Cohere Transcribe — ignoring")

    if ignored:
        logger.info(
            "Accepted whisper.cpp compatibility parameters without applying them: %s",
            ", ".join(ignored),
        )


def build_segments(text: str, duration: float) -> list[dict]:
    """Return a synthetic single-segment payload for verbose formats."""
    return [
        {
            "id": 0,
            "start": 0.0,
            "end": duration,
            "text": text,
        }
    ]


def format_timestamp(duration: float, *, srt: bool) -> str:
    """Format a duration as an SRT or WebVTT timestamp."""
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    millis = int((duration % 1) * 1000)
    separator = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


def format_whisper_response(response_format: str, result: dict):
    """Format transcription output for whisper.cpp-compatible responses."""
    text = result["text"]
    duration = result["duration"]

    if response_format == "text":
        return PlainTextResponse(text + "\n")

    if response_format == "srt":
        srt_content = (
            "1\n"
            f"00:00:00,000 --> {format_timestamp(duration, srt=True)}\n"
            f"{text}\n\n"
        )
        return PlainTextResponse(srt_content, media_type="text/plain")

    if response_format == "vtt":
        vtt_content = (
            "WEBVTT\n\n"
            f"00:00:00.000 --> {format_timestamp(duration, srt=False)}\n"
            f"{text}\n\n"
        )
        return PlainTextResponse(vtt_content, media_type="text/vtt")

    if response_format == "verbose_json":
        return JSONResponse(
            {
                "task": "transcribe",
                "language": result["language"],
                "duration": duration,
                "text": text,
                "segments": build_segments(text, duration),
            }
        )

    return JSONResponse({"text": text})


def format_openai_response(response_format: str, result: dict):
    """Format transcription output for the OpenAI-compatible endpoint."""
    text = result["text"]

    if response_format == "text":
        return PlainTextResponse(text)

    if response_format == "verbose_json":
        return JSONResponse(
            {
                "task": "transcribe",
                "language": result["language"],
                "duration": result["duration"],
                "text": text,
                "segments": build_segments(text, result["duration"]),
            }
        )

    return JSONResponse({"text": text})


def health_payload() -> dict:
    """Return the current service health snapshot."""
    ready = model is not None and processor is not None
    return {
        "status": "ok" if ready else "loading",
        "ready": ready,
        "model": model_id or None,
        "device": str(device) if device is not None else None,
        "backend": model_backend,
    }


def run_transcription_request(
    *,
    audio_data: np.ndarray,
    sr: int,
    language: Optional[str],
    temperature: float,
) -> dict:
    """Resolve request defaults and execute a transcription."""
    lang = resolve_language(language)

    try:
        return transcribe_audio(audio_data, sr, lang, temperature)
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------
def transcribe_audio(
    audio_data: np.ndarray,
    sr: int = 16000,
    language: str = "en",
    temperature: float = 0.0,
) -> dict:
    """
    Transcribe audio using Cohere Transcribe model.
    Returns a dict with text, language, duration.
    """
    global model, processor, device, model_backend

    if model is None or processor is None:
        raise RuntimeError("Model not loaded")

    duration_s = len(audio_data) / sr
    start = time.time()

    inputs = processor(
        audio_data,
        sampling_rate=sr,
        return_tensors="pt",
        language=language,
    )
    audio_chunk_index = inputs.get("audio_chunk_index", None)
    inputs.to(model.device, dtype=model.dtype)

    generate_kwargs = {"max_new_tokens": 256}
    if temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
    else:
        generate_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    if audio_chunk_index is not None:
        text = processor.decode(
            outputs,
            skip_special_tokens=True,
            audio_chunk_index=audio_chunk_index,
            language=language,
        )
    else:
        text = processor.decode(outputs, skip_special_tokens=True)

    if isinstance(text, list):
        text = text[0]

    elapsed = time.time() - start
    rtfx = duration_s / elapsed if elapsed > 0 else 0

    logger.info(
        f"Transcribed {duration_s:.1f}s audio in {elapsed:.1f}s "
        f"(RTFx: {rtfx:.1f}x) lang={language} backend={model_backend}"
    )

    return {
        "text": text.strip() if isinstance(text, str) else str(text).strip(),
        "language": language,
        "duration": round(duration_s, 2),
        "processing_time": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    yield


app = FastAPI(
    title="Cohere Transcribe Server",
    description="whisper.cpp-compatible API powered by CohereLabs/cohere-transcribe-03-2026",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# GET / — Info page (matches whisper.cpp behavior)
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    compatibility_notes = (
        "<li><strong>Full request/response compatibility:</strong> basic whisper.cpp "
        "multipart requests and JSON/text/subtitle response shapes</li>"
        "<li><strong>Compatibility-only parameters:</strong> "
        "<code>temperature_inc</code>, <code>prompt</code>, <code>encode</code>, "
        "<code>no_timestamps</code></li>"
        "<li><strong>Not supported:</strong> translation, auto language detection, "
        "native per-segment timestamps</li>"
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cohere Transcribe Server</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, system-ui, sans-serif;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            color: #e0e0e0; min-height: 100vh; padding: 2rem;
        }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        h1 {{
            font-size: 2.5rem; font-weight: 700;
            background: linear-gradient(90deg, #7f5af0, #2cb67d);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }}
        .card {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px; padding: 1.5rem; margin: 1rem 0;
            backdrop-filter: blur(10px);
        }}
        .card h2 {{ color: #7f5af0; margin-bottom: 0.5rem; font-size: 1.2rem; }}
        code {{
            background: rgba(127,90,240,0.15); color: #2cb67d;
            padding: 2px 6px; border-radius: 4px; font-size: 0.9rem;
        }}
        pre {{
            background: rgba(0,0,0,0.4); border-radius: 8px;
            padding: 1rem; overflow-x: auto; margin: 0.5rem 0;
            border: 1px solid rgba(255,255,255,0.08);
        }}
        pre code {{ background: none; color: #a0e4b0; padding: 0; }}
        .badge {{
            display: inline-block; padding: 4px 12px;
            background: rgba(127,90,240,0.2); border: 1px solid #7f5af0;
            border-radius: 20px; font-size: 0.8rem; color: #7f5af0;
            margin: 2px 4px;
        }}
        .transcribe-form {{
            display: grid;
            gap: 1rem;
        }}
        .form-row {{
            display: grid;
            gap: 0.5rem;
        }}
        label {{
            font-weight: 600;
            color: #fffffe;
        }}
        input[type="file"], select {{
            width: 100%;
            padding: 0.9rem 1rem;
            background: rgba(0,0,0,0.25);
            color: #fffffe;
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 10px;
        }}
        button {{
            border: 0;
            border-radius: 999px;
            padding: 0.95rem 1.4rem;
            background: linear-gradient(135deg, #7f5af0, #2cb67d);
            color: #fffffe;
            font-weight: 700;
            cursor: pointer;
        }}
        button:disabled {{
            opacity: 0.6;
            cursor: wait;
        }}
        .hint {{
            color: #94a1b2;
            font-size: 0.92rem;
        }}
        .result {{
            min-height: 96px;
            white-space: pre-wrap;
            line-height: 1.6;
        }}
        .status {{ color: #2cb67d; font-weight: 600; }}
    </style>
</head>
<body>
<div class="container">
    <h1>🎙️ Cohere Transcribe Server</h1>
    <p>whisper.cpp-compatible API powered by <strong>Cohere Transcribe</strong></p>

    <div class="card">
        <h2>📊 Status</h2>
        <p>Model: <code>{model_id}</code></p>
        <p>Device: <code>{device}</code></p>
        <p>Backend: <code>{model_backend}</code></p>
        <p>Status: <span class="status">● Running</span></p>
    </div>

    <div class="card">
        <h2>🌐 Supported Languages</h2>
        <div>
            {"".join(f'<span class="badge">{l}</span>' for l in sorted(SUPPORTED_LANGUAGES))}
        </div>
    </div>

    <div class="card">
        <h2>🔗 API Endpoints</h2>
        <p><code>POST /inference</code> — whisper.cpp compatible</p>
        <p><code>POST /v1/audio/transcriptions</code> — OpenAI compatible</p>
        <p><code>POST /load</code> — hot-reload model</p>
        <p><code>GET /health</code> — readiness and liveness check</p>
    </div>

    <div class="card">
        <h2>🎧 Quick Transcription</h2>
        <form id="transcribe-form" class="transcribe-form">
            <div class="form-row">
                <label for="audio-file">Audio File</label>
                <input id="audio-file" name="file" type="file" accept="audio/*" required>
            </div>
            <div class="form-row">
                <label for="language">Language</label>
                <select id="language" name="language">
                    {"".join(
                        f'<option value="{lang}"{" selected" if lang == default_language else ""}>{lang}</option>'
                        for lang in sorted(SUPPORTED_LANGUAGES)
                    )}
                </select>
                <div class="hint">Language is a hint for the model. Clear audio may still transcribe correctly even if a different language is selected.</div>
            </div>
            <button id="submit-button" type="submit">Transcribe</button>
        </form>
        <div class="form-row" style="margin-top: 1rem;">
            <label for="transcription-result">Result</label>
            <pre id="transcription-result" class="result"><code>Choose an audio file and click Transcribe.</code></pre>
        </div>
    </div>

    <div class="card">
        <h2>⚠️ Compatibility Notes</h2>
        <ul>{compatibility_notes}</ul>
    </div>

    <div class="card">
        <h2>📋 Example (curl)</h2>
        <pre><code>curl http://127.0.0.1:8080/inference \\
  -H "Content-Type: multipart/form-data" \\
  -F file="@audio.wav" \\
  -F temperature="0.0" \\
  -F response_format="json" \\
  -F language="en"</code></pre>
    </div>
</div>
<script>
const form = document.getElementById("transcribe-form");
const button = document.getElementById("submit-button");
const result = document.getElementById("transcription-result");

form.addEventListener("submit", async (event) => {{
    event.preventDefault();

    const fileInput = document.getElementById("audio-file");
    const languageInput = document.getElementById("language");

    if (!fileInput.files.length) {{
        result.textContent = "Please choose an audio file first.";
        return;
    }}

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("language", languageInput.value);
    formData.append("response_format", "json");
    formData.append("temperature", "0.0");

    button.disabled = true;
    button.textContent = "Transcribing...";
    result.textContent = "Processing audio...";

    try {{
        const response = await fetch("/inference", {{
            method: "POST",
            body: formData,
        }});
        const payload = await response.json();

        if (!response.ok) {{
            throw new Error(payload.detail || "Transcription failed");
        }}

        result.textContent = payload.text || "";
    }} catch (error) {{
        result.textContent = `Error: ${{error.message}}`;
    }} finally {{
        button.disabled = false;
        button.textContent = "Transcribe";
    }}
}});
</script>
</body>
</html>"""


@app.get("/health")
async def health():
    """Container-friendly liveness/readiness endpoint."""
    return JSONResponse(health_payload())


# ---------------------------------------------------------------------------
# POST /inference — whisper.cpp compatible endpoint
# ---------------------------------------------------------------------------
@app.post("/inference")
async def inference(
    file: UploadFile = File(...),
    temperature: float = Form(0.0),
    temperature_inc: float = Form(0.2),
    response_format: str = Form("json"),
    language: Optional[str] = Form(None),
    # Additional whisper.cpp params (accepted but some may be ignored)
    encode: Optional[bool] = Form(True),
    no_timestamps: Optional[bool] = Form(False),
    prompt: Optional[str] = Form(None),
    translate: Optional[bool] = Form(False),
):
    """
    Transcribe audio — whisper.cpp server compatible endpoint.

    Accepts the same multipart/form-data parameters as whisper.cpp server.
    """
    ensure_model_loaded()
    log_ignored_whisper_options(
        temperature_inc=temperature_inc,
        prompt=prompt,
        encode=encode,
        no_timestamps=no_timestamps,
        translate=translate,
    )
    audio_data, sr = await read_upload_audio(file)
    result = run_transcription_request(
        audio_data=audio_data,
        sr=sr,
        language=language,
        temperature=temperature,
    )
    return format_whisper_response(response_format, result)


# ---------------------------------------------------------------------------
# POST /v1/audio/transcriptions — OpenAI / vLLM compatible endpoint
# ---------------------------------------------------------------------------
@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None, alias="model"),
    language: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: float = Form(0.0),
    prompt: Optional[str] = Form(None),
):
    """
    OpenAI-compatible /v1/audio/transcriptions endpoint.
    Same format as used by vLLM serving Cohere Transcribe.
    """
    ensure_model_loaded()
    audio_data, sr = await read_upload_audio(file)
    result = run_transcription_request(
        audio_data=audio_data,
        sr=sr,
        language=language,
        temperature=temperature,
    )
    return format_openai_response(response_format, result)


# ---------------------------------------------------------------------------
# POST /load — hot-reload model (whisper.cpp compatible)
# ---------------------------------------------------------------------------
@app.post("/load")
async def load(
    model_path: Optional[str] = Form(None, alias="model"),
):
    """
    Load a new model. whisper.cpp compatible endpoint.
    Accepts a model path or HuggingFace model ID.
    """
    if model_path is None or model_path.strip() == "":
        raise HTTPException(status_code=400, detail="No model path provided")

    try:
        load_model(model_path.strip())
        return JSONResponse({
            "status": "ok",
            "model": model_path.strip(),
        })
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Cohere Transcribe Server — whisper.cpp compatible API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server options
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Hostname/IP address for the server")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port number for the server")

    # Model options
    parser.add_argument("-m", "--model", type=str,
                        default="CohereLabs/cohere-transcribe-03-2026",
                        help="HuggingFace model ID or local path")
    # Transcription defaults
    parser.add_argument("-l", "--language", type=str, default="en",
                        help="Default spoken language (ISO 639-1 code)")

    # whisper.cpp compatible flags (accepted for compatibility)
    parser.add_argument("-t", "--threads", type=int, default=4,
                        help="Number of threads (sets torch threads)")
    parser.add_argument("-ng", "--no-gpu", action="store_true", default=False,
                        help="Disable GPU, use CPU only")

    return parser.parse_args()


def main():
    global default_language

    args = parse_args()

    # Set default language
    default_language = resolve_language(args.language)

    # Set torch threads
    torch.set_num_threads(args.threads)

    # Force CPU if requested
    if args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Print banner
    print(r"""
  ╔═══════════════════════════════════════════════════════╗
  ║   🎙️  Cohere Transcribe Server                       ║
  ║   whisper.cpp-compatible API                         ║
  ╚═══════════════════════════════════════════════════════╝
    """)
    print(f"  Model:    {args.model}")
    print(f"  Language: {default_language}")
    print(f"  Host:     {args.host}:{args.port}")
    print(f"  Threads:  {args.threads}")
    print(f"  GPU:      {'disabled' if args.no_gpu else 'auto'}")
    print()

    # Load model
    load_model(args.model)

    # Start server
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
