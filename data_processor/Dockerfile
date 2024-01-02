FROM python:3.11-slim
WORKDIR /app
COPY data_processor.py /app
RUN pip install flask
CMD ["python", "data_processor.py"]
