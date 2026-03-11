FROM node:22-bookworm-slim AS frontend-build

WORKDIR /frontend

COPY genesis/webui/frontend/package.json ./
COPY genesis/webui/frontend/package-lock.json ./
RUN npm ci

COPY genesis/webui/frontend/ ./
RUN npm run build


FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY README.md ./

COPY genesis/ ./genesis/
COPY configs/ ./configs/
COPY examples/ ./examples/
COPY tests/ ./tests/

COPY --from=frontend-build /frontend/dist ./genesis/webui/frontend/dist

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

RUN mkdir -p /app/results /app/data

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV GENESIS_DATA_DIR=/app/data
ENV GENESIS_RESULTS_DIR=/app/results
ENV GENESIS_WEBUI_PORT=8000

CMD ["python", "-m", "genesis.webui.backend_server"]
