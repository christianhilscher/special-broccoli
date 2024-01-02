FROM python:3.11-slim
WORKDIR /app
COPY api_server.py /app
RUN pip install flask requests
CMD ["python", "api_server.py"]
