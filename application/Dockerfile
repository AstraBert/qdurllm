FROM python:3.10.14-slim-bookworm

WORKDIR /app

ADD . /app/

RUN python3 -m pip cache purge
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 -m nltk.downloader all

EXPOSE 7860

ENTRYPOINT [ "python3", "/app/scripts/app.py" ]