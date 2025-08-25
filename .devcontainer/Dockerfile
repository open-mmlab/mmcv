# ARG CUDA="12.4"
# ARG CUDNN="9"


# FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
# FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ARG PYTORCH="2.5.1"
ARG PYTHON_VERSION="3.10"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install git and system dependencies for opencv-python
RUN apt-get update && apt-get install -y git curl \
    && apt-get update && apt-get install -y libgl1 libglib2.0-0

# Install system dependencies for unit tests
RUN apt-get install -y ffmpeg libturbojpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y wget build-essential git software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# RUN pip install torch==${PYTORCH} torchvision --index-url https://download.pytorch.org/whl/cu121

# build mmcv from source with develop mode
# ARG HTTPS_PROXY=""
# ENV https_proxy=${HTTPS_PROXY}
# ENV FORCE_CUDA="1"
# ARG CUDA_ARCH=""
# ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCH}
# RUN git clone https://github.com/vbti-development/onedl-mmcv.git /mmcv
# WORKDIR /mmcv
# RUN git checkout 2.x && git rev-parse --short HEAD
# RUN pip install --no-cache-dir -e .[all] -v &&

RUN pip install pre-commit
