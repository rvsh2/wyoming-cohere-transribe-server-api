FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-venv \
    libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY cohere_wyoming ./cohere_wyoming
COPY server.py ./server.py
COPY templates ./templates

RUN uv venv --python 3.11 /app/.venv && \
    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124 \
    VIRTUAL_ENV=/app/.venv uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:${PATH}"
ENV VIRTUAL_ENV="/app/.venv"

EXPOSE 10300

ENTRYPOINT ["python3", "-m", "cohere_wyoming"]
CMD ["--uri", "tcp://0.0.0.0:10300", "--language", "pl"]
