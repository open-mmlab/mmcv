FROM python:3.7

WORKDIR /mmcv

COPY . /mmcv

RUN pip install -e .
