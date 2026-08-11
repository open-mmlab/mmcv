# Builds the mmcv wheel for Python 3.14. The CUDA toolkit comes from the base
# image, so nothing has to be installed on the host to compile the ops.
#
# Run instructions
# docker compose -f docker/builder.yml up -d --build
# docker compose -f docker/builder.yml cp builder:/io/dist/ dist/

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# python 3.14 comes from deadsnakes, ubuntu 22.04 only ships 3.10
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    build-essential && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get -y update && apt-get install -y --no-install-recommends \
    python3.14 \
    python3.14-dev \
    python3.14-venv && \
    rm -rf /var/lib/apt/lists/*

RUN python3.14 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --no-cache-dir torch==2.13.0 \
    --index-url https://download.pytorch.org/whl/cu126

# packaging is imported by setup.py, ninja and psutil speed up the ops build
RUN pip install --no-cache-dir build wheel setuptools packaging ninja psutil

# copy sources only after installing the deps to reduce building time
# after code updates
COPY mmcv/ /io/mmcv
COPY requirements/ /io/requirements
COPY .dev_scripts/ /io/.dev_scripts
COPY setup.py requirements.txt LICENSE MANIFEST.in /io/

WORKDIR /io/

ENV MMCV_WITH_OPS=1
# no GPU is visible while the image builds, so the CUDA ops have to be forced
# and told which archs to emit code for
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST=8.9

RUN python -m build --wheel --no-isolation

# checks the CPU ops only, there is no GPU here
RUN pip install --no-cache-dir dist/*.whl && \
    python .dev_scripts/check_installation.py
