## 最常被问到的问题

我们在这里列出了一些使用者遇到的共通的问题以及他们对应的解决方法。如果您遇到了任何其他常见的问题，并且知道可以帮到大家的解决办法，
欢迎随时丰富这个列表。

- MMCV和MMDetection的兼容性问题；"ConvWS is already registered in conv layer"
    
    请按照上述说明为您的 MMDetection 版本安装正确版本的 MMCV。
    
- "No module named 'mmcv.ops'"; "No module named 'mmcv._ext'"。

    1. 使用`pip uninstall mmcv`卸载您环境中的mmcv。
    2. 按照上述说明安装mmcv-full。

- "invalid device function" 或者 "no kernel image is available for execution"。

    1. 检查您的GPU的CUDA计算能力。
    2. 运行 `python mmdet/utils/collect_env.py` 来检查是否Pytorch、torchvision和MMCV是否是针对正确的GPU架构构建的。
        您可能需要去设置 `TORCH_CUDA_ARCH_LIST` 来重新安装MMCV。
        在使用旧版的GPUS时，如：colab上的Tesla K80 (3.7)，是可能发生兼容性问题的。
    3. 检查运行环境是否和mmcv/mmdet编译时的环境相同。例如，您可能使用CUDA 10.0编译的mmcv，但是在CUDA 9.0的环境中运行它。

- "undefined symbol" 或者 "cannot open xxx.so"。
    
    1. 如果那些符号是CUDA/C++的（例如：libcudart.so 或者 GLIBCXX），检查CUDA/GCC运行时是否和编译mmcv时使用的一样。
    2. 如果那些符号是Pytorch符号（例如：符号包含caffe、aten和TH），检查Pytorch的版本是否和编译mmcv时使用的一样。
    3. 运行 `python mmdet/utils/collect_env.py` 以检查 PyTorch、torchvision 和 MMCV 构建和运行的环境是否相同。

- "RuntimeError: CUDA error: invalid configuration argument"。

    这个错误可能是由于您的GPU性能不佳造成的。尝试降低[THREADS_PER_BLOCK](https://github.com/open-mmlab/mmcv/blob/cac22f8cf5a904477e3b5461b1cc36856c2793da/mmcv/ops/csrc/common_cuda_helper.hpp#L10)
    的值并重新编译mmcv。

- "RuntimeError: nms is not compiled with GPU support"。
    
    这个错误是由于您的CUDA环境没有正确安装。
    您可以尝试重新安装您的CUDA环境，然后在重新编译mmcv前删除build/文件夹。