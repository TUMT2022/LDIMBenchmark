# Implement the method to test in a Docker container

FROM python:3.10
ARG VERSION

WORKDIR /usr/src/app

RUN pip install ldimbenchmark==${VERSION}
