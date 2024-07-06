FROM astrabert/local-search-application:latest

WORKDIR /app

RUN mkdir -p /app/scripts

ADD ./scripts/*.py /app/scripts/

EXPOSE 7860

ENTRYPOINT [ "python3", "/app/scripts/app.py" ]