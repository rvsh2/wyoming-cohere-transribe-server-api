FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-venv \
    libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY requirements.txt .
RUN uv venv /app/.venv && \
    VIRTUAL_ENV=/app/.venv uv pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 && \
    VIRTUAL_ENV=/app/.venv uv pip install -r requirements.txt

ENV PATH="/app/.venv/bin:${PATH}"
ENV VIRTUAL_ENV="/app/.venv"

COPY server.py .

EXPOSE 8080

ENTRYPOINT ["python3", "server.py"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
