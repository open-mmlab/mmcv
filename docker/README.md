# Docker images

There are two `Dockerfile` files to build docker images, one to build an image with the mmcv-full pre-built package and the other with the mmcv development environment.

```text
.
|-- README.md
|-- dev  # build with mmcv development environment
|   `-- Dockerfile
`-- release  # build with mmcv pre-built package
    `-- Dockerfile
```

## Build docker images

### Build with mmcv pre-built package

Build with local repository

```bash
git clone https://github.com/open-mmlab/mmcv.git && cd mmcv
docker build -t mmcv -f docker/release/Dockerfile .
```

Or build with remote repository

```bash
docker build -t mmcv https://github.com/open-mmlab/mmcv.git#master:docker/release
```

The [Dockerfile](release/Dockerfile) installs latest released version of mmcv-full by default, but you can specify mmcv versions to install expected versions.

```bash
docker image build -t mmcv -f docker/release/Dockerfile --build-arg MMCV=1.5.0 .
```

If you also want to use other versions of PyTorch and CUDA, you can also pass them when building docker images.

An example to build an image with PyTorch 1.11 and CUDA 11.3.

```bash
docker build -t mmcv -f docker/release/Dockerfile \
    --build-arg PYTORCH=1.9.0 \
    --build-arg CUDA=11.1 \
    --build-arg CUDNN=8 \
    --build-arg MMCV=1.5.0 .
```

More available versions of PyTorch and CUDA can be found at [dockerhub/pytorch](https://hub.docker.com/r/pytorch/pytorch/tags).

### Build with mmcv development environment

If you want to build an docker image with the mmcv development environment, you can use the following command

```bash
git clone https://github.com/open-mmlab/mmcv.git && cd mmcv
docker build -t mmcv -f docker/dev/Dockerfile --build-arg CUDA_ARCH=7.5 .
```

Note that `CUDA_ARCH` is the cumpute capability of your GPU and you can find it at [Compute Capability](https://developer.nvidia.com/cuda-gpus#compute).

The building process may take 10 minutes or more.

## Run images

```bash
docker run --gpus all --shm-size=8g -it mmcv
```

See [docker run](https://docs.docker.com/engine/reference/commandline/run/) for more usages.
