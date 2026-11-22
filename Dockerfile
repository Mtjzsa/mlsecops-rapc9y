# One image for every Python service (MLflow server, trainer, REST API, Streamlit) - see docker-compose.yml
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so this layer is cached as long as requirements.txt does not change
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .
COPY app ./app

# Run as a non-root user; /app/logs and /mlflow are mounted as volumes
RUN useradd --create-home appuser \
    && mkdir -p /app/logs /app/data /mlflow \
    && chown -R appuser:appuser /app /mlflow
USER appuser

EXPOSE 8000 8501
CMD ["uvicorn", "bike_demand.api:app", "--host", "0.0.0.0", "--port", "8000"]
