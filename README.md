# wyoming-transcribe

Serwer ASR dla Home Assistant oparty o protokol Wyoming i model `CohereLabs/cohere-transcribe-03-2026`.

Glownym interfejsem jest serwer `wyoming`. `server.py` zostal zachowany jako pomocniczy HTTP debug server kompatybilny z podstawowymi requestami `whisper.cpp`.

## Co Jest W Repo

- `cohere_wyoming/` - wspolny runtime, backend transkrypcji i handler Wyoming
- `python -m cohere_wyoming` - glowny serwer dla Home Assistant
- `python server.py` - pomocniczy HTTP debug server
- frontend HTTP z uploadem pliku i nagrywaniem z mikrofonu
- Docker i Compose pod uruchomienie kontenerowe
- testy jednostkowe dla backendu, HTTP i handlera Wyoming

## Wymagania

- Python 3.11+
- `uv` jako podstawowy manager zaleznosci
- GPU jest preferowane, ale CPU fallback jest wspierany
- przy modelu gated z Hugging Face: `HF_TOKEN`

## Szybki Start

### 1. Instalacja przez uv

```bash
UV_CACHE_DIR=/tmp/uv-cache uv venv
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

### 2. Start serwera Wyoming

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m cohere_wyoming \
  --uri tcp://0.0.0.0:10300 \
  --language pl
```

Domyslny port dla integracji z Home Assistant to `10300`.

### 3. Opcjonalny HTTP debug server

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python server.py --host 0.0.0.0 --port 8080 --language pl
```

Tryb HTTP zostal zachowany jako narzedzie developerskie. Frontend obsluguje upload pliku, nagrywanie z mikrofonu i formaty dekodowane przez `ffmpeg`.

## Docker

Najprostsza opcja:

```bash
docker compose up --build -d
```

`compose.yml` uruchamia kontener Wyoming na porcie `10300` i zawiera gotowy preset VAD pod Home Assistant.

Mozliwy jest tez start reczny:

```bash
docker build -t wyoming-transcribe .
docker run --gpus all -p 10300:10300 \
  -e HF_TOKEN=hf_your_token_here \
  wyoming-transcribe \
  --uri tcp://0.0.0.0:10300 \
  --language pl
```

Obraz Dockera korzysta z `uv.lock`, wiec build jest zgodny z lockowanym zestawem zaleznosci.

## Home Assistant

Docelowy scenariusz:

1. uruchomic serwer Wyoming
2. dodac go w Home Assistant jako zewnetrzna usluge Wyoming ASR
3. wskazac URI w stylu `tcp://host:10300`

Aktualnie obslugiwane eventy:

- `describe`
- `transcribe`
- `audio-start`
- `audio-chunk`
- `audio-stop`

Transkrypcja jest wykonywana po odebraniu calej wypowiedzi, po `audio-stop`.

## Wykrywanie Ciszy i Szumu

Backend ma kilka warstw ochrony przed halucynacjami na ciszy:

- preferowanie lokalnego cache Hugging Face przed pobraniem z sieci
- fallback z CUDA do CPU, gdy zaladowanie modelu na GPU sie nie powiedzie
- wykrywanie efektywnej ciszy przez szybki filtr energii (`RMS/peak`)
- `silero-vad` jako dokladniejszy detektor mowy
- dodatkowy filtr `speech RMS` i `speech-to-noise ratio`, zeby odrzucac bardzo ciche dzwieki bliskie szumowi tla

Jesli `silero-vad` nie da sie zaladowac, serwer przechodzi na fallback i nadal dziala z prostszym detektorem ciszy.

Najwazniejsze zmienne srodowiskowe:

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

Przydatne opcje CLI:

- `--disable-vad`
- `--vad-threshold 0.6`

Polecany preset startowy dla Home Assistant:

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

Jak stroic w praktyce:

- jesli nadal pojawiaja sie halucynacje przy ciszy lub szumie, podnies `VAD_THRESHOLD`
- jesli przepuszcza bardzo ciche samogloski albo szum przypominajacy glos, podnies `VAD_MIN_SPEECH_RMS`
- jesli przepuszcza dzwieki tylko minimalnie glosniejsze od tla, podnies `VAD_MIN_SPEECH_TO_NOISE_RATIO`
- jesli ucina bardzo krotkie komendy, obniz `VAD_MIN_TOTAL_SPEECH_MS` i `VAD_MIN_MAX_SEGMENT_MS`
- jesli obcina poczatek lub koniec wypowiedzi, zwieksz `VAD_SPEECH_PAD_MS`

## Obslugiwane Jezyki

`en`, `fr`, `de`, `it`, `es`, `pt`, `el`, `nl`, `pl`, `zh`, `ja`, `ko`, `vi`, `ar`

## Ograniczenia Aktualnej Wersji

- brak partial transcripts
- brak autodetekcji jezyka
- brak `zeroconf`
- brak natywnego streamingu wynikow
- HTTP zostaje jako warstwa pomocnicza, nie glowna integracja

## Testy

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests
```

## Zrodlo Wzorca

- `wyoming-faster-whisper`: https://github.com/rhasspy/wyoming-faster-whisper
