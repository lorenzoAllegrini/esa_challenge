FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir poetry \
    && poetry install --no-dev

RUN python spaceai/segmentators/setup.py build_ext --inplace

CMD ["python", "inference.py", "--test-parquet", "/data/test.parquet", "--artifacts-dir", "/artifacts", "--output", "/data/submission.csv"]
