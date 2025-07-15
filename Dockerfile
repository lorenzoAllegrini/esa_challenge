FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir cython poetry && \
    poetry install --no-dev && \
    python spaceai/segmentators/setup.py build_ext --inplace && \
    apt-get purge -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

CMD ["python", "inference.py", "--test-parquet", "/data/test.parquet", "--artifacts-dir", "/artifacts", "--output", "/data/submission.csv"]
