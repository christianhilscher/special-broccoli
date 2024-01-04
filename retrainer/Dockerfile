
FROM python:3.11
WORKDIR /app

COPY train_model.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "train_model.py"]
