# cohere-transcribe-server-api

`whisper.cpp`-compatible HTTP API powered by [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026).

The project is aimed at practical API compatibility for existing `whisper.cpp` clients: same main routes, multipart request shapes, and familiar response formats. It is not a full reimplementation of every `whisper.cpp` feature.

## What Is Implemented

- `POST /inference` for `whisper.cpp`-style multipart transcription requests
- `POST /v1/audio/transcriptions` for OpenAI-like clients
- `POST /load` for hot-reloading the model
- `GET /` for a small status and compatibility page
- `GET /health` for liveness and readiness checks
- Docker and Compose workflows for containerized startup
- Automated endpoint-level API tests with a mocked transcription layer

## Quick Start

### Local Python

```bash
pip install -r requirements.txt
python server.py --host 0.0.0.0 --port 8080
```

The production server uses the native `AutoProcessor` + `AutoModelForSpeechSeq2Seq` path with `transformers==5.4.0` and does not use `trust_remote_code`.

Incoming audio is normalized before inference:

- stereo or multi-channel audio is mixed down to mono
- input sample rates such as `16 kHz`, `22.05 kHz`, `44.1 kHz`, and `48 kHz` are resampled to `16 kHz`
- WAV, MP3, FLAC, OGG and other decodable formats are accepted through `soundfile` with a `librosa` fallback

With explicit options:

```bash
python server.py \
  --model CohereLabs/cohere-transcribe-03-2026 \
  --host 0.0.0.0 \
  --port 8080 \
  --language en
```

### Docker / Compose

```bash
docker build -t cohere-transcribe-server-api .
docker run --gpus all -p 8080:8080 cohere-transcribe-server-api
```

```bash
docker compose up --build
```

If you are using the gated Cohere model from Hugging Face, export your token before startup:

```bash
export HF_TOKEN=hf_your_token_here
docker compose up -d
```

The Compose file passes `HF_TOKEN` into both `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` inside the container.

The provided Compose file:

- binds the API to `127.0.0.1:8080`
- joins the external Docker network `bridge-network`
- persists Hugging Face cache under `/opt/cohere-transcribe/data`

## API

### `POST /inference`

Main `whisper.cpp`-compatible endpoint.

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `file` | file | required | WAV, MP3, FLAC, OGG and other decodable formats; audio is normalized to mono 16 kHz before inference |
| `temperature` | float | `0.0` | Passed through to transcription |
| `temperature_inc` | float | `0.2` | Accepted for compatibility, not applied |
| `response_format` | string | `json` | `json`, `text`, `verbose_json`, `srt`, `vtt` |
| `language` | string | `en` | ISO 639-1 language code |
| `prompt` | string | unset | Accepted for compatibility, not applied |
| `encode` | bool | `true` | Accepted for compatibility, not applied |
| `no_timestamps` | bool | `false` | Accepted for compatibility, not applied |
| `translate` | bool | `false` | Accepted, but translation is not supported |

Example:

```bash
curl http://127.0.0.1:8080/inference \
  -H "Content-Type: multipart/form-data" \
  -F file="@audio.wav" \
  -F temperature="0.0" \
  -F temperature_inc="0.2" \
  -F response_format="json" \
  -F language="en"
```

Default JSON response:

```json
{
  "text": "Transcribed text goes here..."
}
```

### `POST /v1/audio/transcriptions`

OpenAI-like transcription endpoint.

```bash
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file="@audio.wav" \
  -F model="CohereLabs/cohere-transcribe-03-2026" \
  -F language="en"
```

### `POST /load`

Administrative endpoint for hot-reloading the active model without restarting the server.

Typical uses:

- reload the same model after cache or credential changes
- switch to another Hugging Face model ID
- switch to a compatible local model path mounted into the container

Request:

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `model` | string | required | Hugging Face model ID or local path to a compatible checkpoint |

```bash
curl http://127.0.0.1:8080/load \
  -F model="CohereLabs/cohere-transcribe-03-2026"
```

Example response:

```json
{
  "status": "ok",
  "model": "CohereLabs/cohere-transcribe-03-2026"
}
```

If loading fails, the endpoint returns `500` and the server keeps the previously loaded model active.

### `GET /`

Returns a small HTML status page with endpoints, device/model info, compatibility notes, and a built-in `Quick Transcription` upload form.

The frontend lets you:

- upload an audio file directly from the browser
- choose the transcription language from the supported list
- send the request without using `curl`
- view the transcribed text immediately on the page

The upload form sends the audio to the same backend used by `POST /inference`.

Open it in the browser:

```text
http://127.0.0.1:8080/
```

### `GET /health`

Returns a JSON health payload for load balancers and Docker health checks.

Example response:

```json
{
  "status": "ok",
  "ready": true,
  "model": "CohereLabs/cohere-transcribe-03-2026",
  "device": "cuda:0",
  "backend": "native"
}
```

## Compatibility Matrix

| Area | Status | Notes |
|------|--------|-------|
| Route compatibility | Full | Same public routes used by current implementation |
| Multipart request shape | Full | `file`, `language`, `response_format`, `temperature` supported |
| JSON response shape | Full | Default `{"text": "..."}` shape preserved |
| `text` / `verbose_json` output | Full | Stable and tested |
| `srt` / `vtt` output | Partial | Synthetic full-audio timestamps only |
| `translate` | Not supported | Accepted but ignored |
| `prompt`, `temperature_inc`, `encode`, `no_timestamps` | Compatibility-only | Accepted but not applied |
| Auto language detection | Not supported | `language=auto` falls back to server default |
| Per-segment timestamps | Not supported | One synthetic segment covers the whole file |

## Supported Languages

| Code | Language |
|------|----------|
| en | English |
| fr | French |
| de | German |
| it | Italian |
| es | Spanish |
| pt | Portuguese |
| el | Greek |
| nl | Dutch |
| pl | Polish |
| zh | Chinese (Mandarin) |
| ja | Japanese |
| ko | Korean |
| vi | Vietnamese |
| ar | Arabic |

## CLI Options

```text
--host HOST
--port PORT
-m, --model MODEL
-l, --language LANG
-t, --threads N
-ng, --no-gpu
```

Defaults:

- GPU is auto-detected when available
- native transformers loading is the only production path
- `language=auto` is treated as the configured default language

## GPU Compatibility

The Docker image is pinned to a CUDA 12.4 compatible PyTorch build so it can run on hosts that have an NVIDIA driver with CUDA 12.6 support, such as systems with RTX 3090 cards and older fixed driver stacks.

Current container stack:

- base image: `nvidia/cuda:12.4.1-runtime-ubuntu22.04`
- PyTorch: `2.6.0+cu124`

If GPU detection works, startup logs will contain a line similar to:

```text
Using CUDA device: NVIDIA GeForce RTX 3090
```

## Verification

### Smoke Checks

```bash
python server.py --help
python -m py_compile server.py
python -m unittest discover -s tests -v
```

### Manual API Checks

```bash
./test_api.sh path/to/audio.wav
```

### Docker Runtime Checks

```bash
docker compose ps
docker compose logs -f cohere-transcribe
curl http://127.0.0.1:8080/
curl http://127.0.0.1:8080/health
```

## Operational Notes

- The Cohere model is large enough that GPU is strongly preferred for good latency.
- The Hugging Face model may require accepted license terms and cached model access in your environment.
- CPU fallback is supported, but will be slower.
- If startup fails with `401 Unauthorized` or `GatedRepoError`, you need a Hugging Face account with access to `CohereLabs/cohere-transcribe-03-2026` and a valid `HF_TOKEN`.
- The production image is pinned to `transformers==5.4.0` because this model works correctly in native mode on that stack.
- The server resamples all incoming audio to `16 kHz` mono before inference, so mixed source sample rates are expected and supported.
- If startup falls back to CPU with a CUDA driver warning, either the host driver is too old for the installed PyTorch build or the container is not seeing the GPUs correctly.

## License

Apache 2.0, matching the model packaging used by this project.
