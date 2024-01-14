FROM python:3.11

WORKDIR /app

COPY pyproject.toml poetry.lock* /app/

RUN pip install --no-cache-dir poetry

RUN poetry config virtualenvs.create false
RUN poetry install

COPY . /app
ENV PYTHONPATH /app

CMD ["python", "smart_buildings/retrain/train.py"]
