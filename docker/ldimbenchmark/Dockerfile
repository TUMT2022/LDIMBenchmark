# Implement the method to test in a Docker container

FROM python:3.10

WORKDIR /usr/src/app

COPY dist/ldimbenchmark-0.0.0-py3-none-any.whl ldimbenchmark-0.0.0-py3-none-any.whl 
RUN pip install ldimbenchmark-0.0.0-py3-none-any.whl
# RUN pip install ldimbenchmark==0.1.25
