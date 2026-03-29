# Cohere Transcribe API — plan domknięcia implementacji

## Summary

Projekt ma już działające MVP: serwer FastAPI, endpointy kompatybilne ścieżkami z `whisper.cpp`, podstawową dokumentację, konteneryzację i prosty skrypt testowy. Celem dalszych prac nie jest budowa od zera, tylko doprowadzenie rozwiązania do stanu „zweryfikowane i jasno opisane”, z naciskiem na praktyczną kompatybilność API, przewidywalne błędy i testowalność.

Najlepszy docelowy kierunek:
- zachować obecny model `CohereLabs/cohere-transcribe-03-2026`;
- domyślnie zostawić auto-detekcję GPU z opcją CPU fallback;
- używać natywnej ścieżki `transformers==5.4.0` bez `trust_remote_code=True`;
- utrzymać Dockerfile i `compose.yml`, bo projekt już jest przygotowany pod uruchamianie kontenerowe.

## Implementation Changes

### 1. Urealnienie kontraktu kompatybilności

- Zachować publiczne endpointy bez zmian: `/inference`, `/v1/audio/transcriptions`, `/load`, `/`.
- W dokumentacji i zachowaniu API jasno oznaczyć, że kompatybilność dotyczy przede wszystkim kształtu requestów i odpowiedzi, nie pełnego zestawu możliwości `whisper.cpp`.
- Parametry `temperature_inc`, `prompt`, `encode`, `no_timestamps` i `translate` traktować jako parametry kompatybilnościowe:
  serwer może je przyjmować, ale musi być jasno opisane, które są ignorowane lub częściowo wspierane.
- Doprecyzować, że `translate` nie jest wspierane, `auto language` nie jest wspierane, a timestampy są tylko syntetyczne na poziomie całego pliku, nie segmentów modelowych.

### 2. Stabilizacja implementacji serwera

- Uporządkować kod serwera wokół trzech odpowiedzialności:
  ładowanie modelu, dekodowanie audio, warstwa HTTP/formatowanie odpowiedzi.
- Utrzymać natywny przepływ ładowania modelu:
  `AutoProcessor` + `AutoModelForSpeechSeq2Seq` na `transformers==5.4.0` bez `trust_remote_code=True`.
- Utrzymać auto-detekcję GPU jako domyślne zachowanie.
- Zachować opcję `--no-gpu` dla środowisk CPU-only.
- Utrzymać preprocessing audio po stronie serwera:
  downmix do mono i resampling do `16 kHz` niezależnie od wejściowego sample rate.
- Upewnić się, że błędy wejścia i błędy modelu są mapowane do przewidywalnych kodów HTTP:
  pusty plik, nieobsługiwany format, brak modelu, błąd transkrypcji, błąd przeładowania modelu.

### 3. Dokumentacja zgodna z rzeczywistością

- Przepisać README tak, aby opisywał istniejący projekt jako „whisper.cpp-compatible HTTP API”, ale z wyraźną sekcją ograniczeń.
- Zmienić język planu z „dodamy nowe pliki” na „utrzymujemy i dopracowujemy istniejące pliki”.
- W sekcji uruchomienia zostawić dwa oficjalne tryby:
  lokalny Python oraz Docker/Compose.
- W sekcji kompatybilności dodać krótką tabelę:
  co jest w pełni wspierane, co częściowo wspierane, a co tylko akceptowane dla kompatybilności.

### 4. Testowalność i weryfikacja

- Zachować `test_api.sh` jako narzędzie ręczne, ale nie traktować go jako jedynego sposobu walidacji.
- Dodać automatyczne testy HTTP obejmujące:
  `GET /`, `POST /inference`, `POST /v1/audio/transcriptions`, `POST /load`.
- Dodać testy błędów:
  pusty plik, błędny plik audio, nieobsługiwany język, brak modelu.
- Dodać testy formatów odpowiedzi:
  `json`, `text`, `verbose_json`, `srt`, `vtt`.
- W testach automatycznych użyć stubu lub mocka warstwy modelu, żeby nie uzależniać CI od pobierania dużego modelu i GPU.

## Public Interfaces

### Zachowane interfejsy

- `POST /inference`
- `POST /v1/audio/transcriptions`
- `POST /load`
- `GET /`

### Zachowane parametry CLI

- `--host`
- `--port`
- `--model`
- `--language`
- `--threads`
- `--no-gpu`

### Doprecyzowania interfejsu

- `response_format=json` pozostaje domyślnym formatem zgodnym z obecnym zachowaniem.
- `verbose_json` zwraca jeden syntetyczny segment obejmujący cały plik audio.
- `srt` i `vtt` pozostają formatami uproszczonymi bez prawdziwej segmentacji modelowej.
- `language=auto` jest mapowane do domyślnego języka serwera, a nie do realnej autodetekcji.

## Test Plan

### Smoke checks

- Uruchomienie `python server.py --help`.
- Walidacja składni `python -m py_compile server.py`.
- Uruchomienie kontenera przez Dockerfile i Compose.

### API behavior

- `GET /` zwraca stronę statusową.
- `POST /inference` z poprawnym plikiem zwraca poprawny shape odpowiedzi.
- `POST /v1/audio/transcriptions` zwraca zgodny shape odpowiedzi OpenAI-like.
- `POST /load` potrafi przeładować model i zwrócić status `ok`.

### Error scenarios

- pusty upload zwraca `400`;
- niepoprawny plik audio zwraca `400`;
- brak załadowanego modelu zwraca `503`;
- błąd transkrypcji lub błędny model przy `/load` zwraca `500`.

### Compatibility scenarios

- requesty `curl` w stylu `whisper.cpp` działają bez zmian ścieżek i podstawowych parametrów;
- ignorowane parametry nie powodują awarii;
- pliki audio o różnych sample rate są poprawnie resamplowane do `16 kHz` mono;
- README i realne zachowanie endpointów są zgodne.

## Assumptions And Defaults

- Projekt nie ma celu odtworzenia pełnego feature setu `whisper.cpp`; celem jest praktyczna kompatybilność API dla istniejących klientów.
- Natywna ścieżka `transformers==5.4.0` jest domyślną i produkcyjną opcją ładowania modelu.
- GPU pozostaje domyślną preferowaną ścieżką wykonania, ale CPU fallback ma być zachowany.
- Dockerfile i `compose.yml` pozostają częścią oficjalnego sposobu uruchamiania.
- Brak repo git nie blokuje implementacji, ale przed dalszym rozwojem warto objąć projekt kontrolą wersji, żeby śledzić zmiany i status.
