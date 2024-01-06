FROM python:3.11

WORKDIR /app

COPY pyproject.toml poetry.lock* /app/

RUN pip install --no-cache-dir poetry

RUN poetry config virtualenvs.create false
RUN poetry install

COPY . /app

EXPOSE 5000

CMD ["python", "api_server/run.py"]