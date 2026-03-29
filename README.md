# wyoming-transcribe

Speech recognition server for Home Assistant built around the Wyoming protocol and the `CohereLabs/cohere-transcribe-03-2026` model.

In practice, this is a self-hosted speech-to-text service in the same general category as Whisper-based setups, but using Cohere Transcribe instead.

The main interface is the Wyoming server. `server.py` is still available as a small HTTP debug server with basic `whisper.cpp`-style compatibility.

## What Is In This Repository

- `cohere_wyoming/` - shared runtime, transcription backend, and Wyoming handler
- `python -m cohere_wyoming` - main server for Home Assistant
- `python server.py` - optional HTTP debug server
- HTTP frontend for file upload and microphone recording
- Docker and Compose setup
- unit tests for backend, HTTP, and Wyoming handler flows

## Requirements

- Python 3.11+
- `uv` as the primary dependency manager
- GPU is preferred, but CPU fallback is supported
- `HF_TOKEN` if you are using the gated Hugging Face model

## Quick Start

### 1. Install with uv

```bash
cp .env.example .env
UV_CACHE_DIR=/tmp/uv-cache uv venv
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

Set `HF_TOKEN` in `.env` before the first model download.

### 2. Start the Wyoming server

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m cohere_wyoming \
  --uri tcp://0.0.0.0:10300 \
  --language pl
```

The default port for Home Assistant integration is `10300`.

### 3. Optional HTTP debug server

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python server.py --host 0.0.0.0 --port 8080 --language pl
```

The HTTP mode is intended as a developer tool. The frontend supports file upload, browser microphone recording, and formats decoded through the `ffmpeg` fallback.

## Docker

The simplest option:

```bash
docker compose up --build -d
```

`compose.yml` runs the Wyoming service on port `10300` and includes a ready-to-use VAD preset for Home Assistant.

Manual container start is also possible:

```bash
docker build -t wyoming-transcribe .
docker run --gpus all -p 10300:10300 \
  -e HF_TOKEN=hf_your_token_here \
  wyoming-transcribe \
  --uri tcp://0.0.0.0:10300 \
  --language pl
```

The Docker image uses `uv.lock`, so builds stay aligned with the locked dependency set.

## Home Assistant

Typical setup:

1. Start the Wyoming server.
2. Add it in Home Assistant through the `Wyoming Protocol` integration.
3. In Home Assistant, enter the host and port `10300`.

Currently supported events:

- `describe`
- `transcribe`
- `audio-start`
- `audio-chunk`
- `audio-stop`

Transcription runs after the full utterance is received, on `audio-stop`.

## Silence and Noise Handling

The backend has several layers to reduce hallucinations on silence:

- prefer local Hugging Face cache before network download
- fall back from CUDA to CPU if GPU model loading fails
- detect effective silence with a fast energy-based filter (`RMS/peak`)
- use `silero-vad` as a more precise speech detector
- apply additional `speech RMS` and `speech-to-noise ratio` checks to reject very quiet sounds close to background noise

If `silero-vad` cannot be loaded, the server falls back to the simpler silence detector and keeps running.

Important environment variables:

- `VAD_ENABLED=true`
- `VAD_THRESHOLD=0.5`
- `VAD_MIN_SPEECH_DURATION_MS=250`
- `VAD_MIN_SILENCE_DURATION_MS=100`
- `VAD_SPEECH_PAD_MS=30`
- `VAD_MIN_TOTAL_SPEECH_MS=60`
- `VAD_MIN_MAX_SEGMENT_MS=40`
- `VAD_MIN_SPEECH_RMS=0.012`
- `VAD_MIN_SPEECH_TO_NOISE_RATIO=3.0`
- `VAD_USE_ONNX=false`

Useful CLI options:

- `--disable-vad`
- `--vad-threshold 0.6`

Recommended starting preset for Home Assistant:

```env
VAD_ENABLED=true
VAD_THRESHOLD=0.54
VAD_MIN_SPEECH_DURATION_MS=180
VAD_MIN_SILENCE_DURATION_MS=120
VAD_SPEECH_PAD_MS=50
VAD_MIN_TOTAL_SPEECH_MS=70
VAD_MIN_MAX_SEGMENT_MS=45
VAD_MIN_SPEECH_RMS=0.014
VAD_MIN_SPEECH_TO_NOISE_RATIO=2.6
```

Practical tuning guidance:

- if you still get hallucinations on silence or noise, increase `VAD_THRESHOLD`
- if very quiet vowels or speech-like noise still pass through, increase `VAD_MIN_SPEECH_RMS`
- if sounds only slightly louder than background noise still pass through, increase `VAD_MIN_SPEECH_TO_NOISE_RATIO`
- if short commands get cut off, lower `VAD_MIN_TOTAL_SPEECH_MS` and `VAD_MIN_MAX_SEGMENT_MS`
- if the start or end of speech gets clipped, increase `VAD_SPEECH_PAD_MS`

## Supported Languages

`en`, `fr`, `de`, `it`, `es`, `pt`, `el`, `nl`, `pl`, `zh`, `ja`, `ko`, `vi`, `ar`

## Current Limitations

- no partial transcripts
- no language auto-detection
- no `zeroconf`
- no native streaming results
- HTTP remains a helper/debug layer, not the primary integration path

## Tests

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests
```

HTTP smoke test script:

```bash
tests/test_api.sh
```

## Reference

- `wyoming-faster-whisper`: https://github.com/rhasspy/wyoming-faster-whisper

## License

This repository is licensed under the Apache License 2.0.
