This dockerfile is to make it easy to build multiple versions of mmcv with different torch and cuda versions.

The process is two steps:

1. Build the docker with the correct cuda and torch versions. Base image is based on [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda/tags) image.

   ```bash
   docker build -t mmcv-builder:cu118-torch251 --build-arg CUDA=11.8.0 --build-arg PYTORCH=2.5.1 docker/build .
   ```

   Note:

   In for some cuda versions you cannot pick the CUDNN version, in that case add `--build-arg CUDNN=""` to the build command in step 1.

2. Build the wheel

   ```bash
   docker run --rm -v "$PWD/dist:/out" -e TORCH_CUDA_ARCH_LIST="7.5 8.6 8.9" mmcv-builder:cu118-torch251
   ```

   Note:

   Adjust the TORCH_CUDA_ARG_LIST to all current GPU's.

By modifying the CUDA and PYTORCH arguments different versions will be built and outputted in the `$PWD/dist` subfolder that is mounted.
