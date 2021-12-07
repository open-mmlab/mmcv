## 常见问题

在这里我们列出了用户经常遇到的问题以及对应的解决方法。如果您遇到了其他常见的问题，并且知道可以帮到大家的解决办法，
欢迎随时丰富这个列表。

- MMCV 和 MMDetection 的兼容性问题；"ConvWS is already registered in conv layer"

    请按照上述说明为您的 MMDetection 版本安装正确版本的 MMCV。

- "No module named 'mmcv.ops'"; "No module named 'mmcv._ext'"

    1. 使用 `pip uninstall mmcv` 卸载您环境中的 mmcv
    2. 按照上述说明安装 mmcv-full

- "invalid device function" 或者 "no kernel image is available for execution"

    1. 检查 GPU 的 CUDA 计算能力
    2. 运行  `python mmdet/utils/collect_env.py` 来检查 PyTorch、torchvision 和 MMCV 是否是针对正确的 GPU 架构构建的
        您可能需要去设置 `TORCH_CUDA_ARCH_LIST` 来重新安装 MMCV
        兼容性问题的可能会出现在使用旧版的 GPUs，如：colab 上的 Tesla K80 (3.7)
    3. 检查运行环境是否和 mmcv/mmdet 编译时的环境相同。例如，您可能使用 CUDA 10.0 编译 mmcv，但在 CUDA 9.0 的环境中运行它

- "undefined symbol" 或者 "cannot open xxx.so"。

    1. 如果符号和 CUDA/C++ 相关（例如：libcudart.so 或者 GLIBCXX），请检查 CUDA/GCC 运行时的版本是否和编译 mmcv 的一致
    2. 如果符号和 PyTorch 相关（例如：符号包含 caffe、aten 和 TH），请检查 PyTorch 运行时的版本是否和编译 mmcv 的一致
    3. 运行 `python mmdet/utils/collect_env.py` 以检查 PyTorch、torchvision 和 MMCV 构建和运行的环境是否相同

- "RuntimeError: CUDA error: invalid configuration argument"。

    这个错误可能是由于您的 GPU 性能不佳造成的。尝试降低[THREADS_PER_BLOCK](https://github.com/open-mmlab/mmcv/blob/cac22f8cf5a904477e3b5461b1cc36856c2793da/mmcv/ops/csrc/common_cuda_helper.hpp#L10)
    的值并重新编译 mmcv。

- "RuntimeError: nms is not compiled with GPU support"。

    这个错误是由于您的 CUDA 环境没有正确安装。
    您可以尝试重新安装您的 CUDA 环境，然后删除 mmcv/build 文件夹并重新编译 mmcv。
