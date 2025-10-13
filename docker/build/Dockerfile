ARG CUDA="11.8.0"
ARG CUDNN="8"

# FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu22.04
ARG CUDA

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install git and system dependencies for opencv-python
RUN apt-get update && apt-get install -y git \
    && apt-get update && apt-get install -y libgl1 libglib2.0-0 patchelf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.8.14 /uv /uvx /bin/

# build mmcv from source with develop mode
ARG HTTPS_PROXY=""
ENV https_proxy=${HTTPS_PROXY}
ENV FORCE_CUDA="1"
ARG CUDA_ARCH=""
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCH}

# build mmcv with uv
WORKDIR /mmcv
RUN git clone --single-branch https://github.com/vbti-development/onedl-mmcv.git /mmcv \
    && rm -r /mmcv/.git \
    && uv venv --python 3.10

# install torch
ARG PYTORCH="2.5.1"
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    CUDA_SHORT=${CUDA%.*} CU_BE=$(echo $CUDA_SHORT | tr -d .) && uv pip install --torch-backend=cu${CU_BE} torch==${PYTORCH}

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install setuptools && uv sync --group build --group optional --no-install-project --inexact

COPY entrypoint.sh /entrypoint.sh

VOLUME [ "/out" ]
ENTRYPOINT ["/entrypoint.sh" ]
