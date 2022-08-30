ARG PYTORCH="1.8.1"
ARG CUDA="10.2"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# To fix GPG key error when running apt-get update
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install git and system dependencies for opencv-python
RUN apt-get update && apt-get install -y git \
    && apt-get update && apt-get install -y libgl1 libglib2.0-0

# Install system dependencies for unit tests
RUN apt-get install -y ffmpeg libturbojpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# build mmcv from source with develop mode
ARG HTTPS_PROXY=""
ENV https_proxy=${HTTPS_PROXY}
ENV FORCE_CUDA="1"
ARG CUDA_ARCH=""
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCH}
RUN git clone https://github.com/open-mmlab/mmcv.git /mmcv
WORKDIR /mmcv
RUN git checkout 2.x && git rev-parse --short HEAD
RUN pip install --no-cache-dir -e .[all] -v && pip install pre-commit && pre-commit install
