FROM python:3.9-slim

ENV PYTHONUNBUFFERED True

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "main.py"]
