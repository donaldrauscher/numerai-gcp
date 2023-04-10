FROM python:3.9-slim

ENV PYTHONUNBUFFERED True
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get update && \
  apt-get install -y libgomp1 && \
  rm -rf /var/lib/apt/lists/*

COPY . ./

ENTRYPOINT ["python", "main.py"]
