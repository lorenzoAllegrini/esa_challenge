FROM python:3.11-slim

WORKDIR /app

# Codice
COPY esa-inference_docker/spaceai /app/spaceai
COPY esa-inference_docker/ensemble_inference.py /app/
COPY esa-inference_docker/run_inference.py /app/

# Solo i modelli
COPY experiments/esa_training/models /app/experiments/models

# Requisiti
COPY esa-inference_docker/requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "run_inference.py"]
