## 常见问题

在这里我们列出了用户经常遇到的问题以及对应的解决方法。如果您遇到了其他常见的问题，并且知道可以帮到大家的解决办法，
欢迎随时丰富这个列表。

### 安装问题

- KeyError: "xxx: 'yyy is not in the zzz registry'"

  只有模块所在的文件被导入时，注册机制才会被触发，所以您需要在某处导入该文件，更多详情请查看 [KeyError: "MaskRCNN: 'RefineRoIHead is not in the models registry'"](https://github.com/open-mmlab/mmdetection/issues/5974)。

- "No module named 'mmcv.ops'"; "No module named 'mmcv.\_ext'"

  1. 使用 `pip uninstall mmcv` 卸载您环境中的 mmcv
  2. 参考 [installation instruction](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) 或者 [Build MMCV from source](https://mmcv.readthedocs.io/en/latest/get_started/build.html) 安装 mmcv-full

- "invalid device function" 或者 "no kernel image is available for execution"

  1. 检查 GPU 的 CUDA 计算能力
  2. 运行 `python mmdet/utils/collect_env.py` 来检查 PyTorch、torchvision 和 MMCV 是否是针对正确的 GPU 架构构建的，您可能需要去设置 `TORCH_CUDA_ARCH_LIST` 来重新安装 MMCV。兼容性问题可能会出现在使用旧版的 GPUs，如：colab 上的 Tesla K80 (3.7)
  3. 检查运行环境是否和 mmcv/mmdet 编译时的环境相同。例如，您可能使用 CUDA 10.0 编译 mmcv，但在 CUDA 9.0 的环境中运行它

- "undefined symbol" 或者 "cannot open xxx.so"

  1. 如果符号和 CUDA/C++ 相关（例如：libcudart.so 或者 GLIBCXX），请检查 CUDA/GCC 运行时的版本是否和编译 mmcv 的一致
  2. 如果符号和 PyTorch 相关（例如：符号包含 caffe、aten 和 TH），请检查 PyTorch 运行时的版本是否和编译 mmcv 的一致
  3. 运行 `python mmdet/utils/collect_env.py` 以检查 PyTorch、torchvision 和 MMCV 构建和运行的环境是否相同

- "RuntimeError: CUDA error: invalid configuration argument"

  这个错误可能是由于您的 GPU 性能不佳造成的。尝试降低 [THREADS_PER_BLOCK](https://github.com/open-mmlab/mmcv/blob/cac22f8cf5a904477e3b5461b1cc36856c2793da/mmcv/ops/csrc/common_cuda_helper.hpp#L10)
  的值并重新编译 mmcv。

- "RuntimeError: nms is not compiled with GPU support"

  这个错误是由于您的 CUDA 环境没有正确安装。
  您可以尝试重新安装您的 CUDA 环境，然后删除 mmcv/build 文件夹并重新编译 mmcv。

- "Segmentation fault"

  1. 检查 GCC 的版本，通常是因为 PyTorch 版本与 GCC 版本不匹配 （例如 GCC \< 4.9 )，我们推荐用户使用 GCC 5.4，我们也不推荐使用 GCC 5.5， 因为有反馈 GCC 5.5 会导致 "segmentation fault" 并且切换到 GCC 5.4 就可以解决问题
  2. 检查是否正确安装 CUDA 版本的 PyTorc。输入以下命令并检查是否返回 True
     ```shell
     python -c 'import torch; print(torch.cuda.is_available())'
     ```
  3. 如果 `torch` 安装成功，那么检查 MMCV 是否安装成功。输入以下命令，如果没有报错说明 mmcv-full 安装成。
     ```shell
     python -c 'import mmcv; import mmcv.ops'
     ```
  4. 如果 MMCV 与 PyTorch 都安装成功了，则可以使用 `ipdb` 设置断点或者使用 `print` 函数，分析是哪一部分的代码导致了 `segmentation fault`

- "libtorch_cuda_cu.so: cannot open shared object file"

  `mmcv-full` 依赖 `libtorch_cuda_cu.so` 文件，但程序运行时没能找到该文件。我们可以检查该文件是否存在 `~/miniconda3/envs/{environment-name}/lib/python3.7/site-packages/torch/lib` 也可以尝试重装 PyTorch。

- "fatal error C1189: #error:  -- unsupported Microsoft Visual Studio version!"

  如果您在 Windows 上编译 mmcv-full 并且 CUDA 的版本是 9.2，您很可能会遇到这个问题 `"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\include\crt/host_config.h(133): fatal error C1189: #error:  -- unsupported Microsoft Visual Studio version! Only the versions 2012, 2013, 2015 and 2017 are supported!"`，您可以尝试使用低版本的 Microsoft Visual Studio，例如 vs2017。

- "error: member "torch::jit::detail::ModulePolicy::all_slots" may not be initialized"

  如果您在 Windows 上编译 mmcv-full 并且 PyTorch 的版本是 1.5.0，您很可能会遇到这个问题 `- torch/csrc/jit/api/module.h(474): error: member "torch::jit::detail::ModulePolicy::all_slots" may not be initialized`。解决这个问题的方法是将 `torch/csrc/jit/api/module.h` 文件中所有 `static constexpr bool all_slots = false;` 替换为 `static bool all_slots = false;`。更多细节可以查看 [member "torch::jit::detail::AttributePolicy::all_slots" may not be initialized](https://github.com/pytorch/pytorch/issues/39394)。

- "error: a member with an in-class initializer must be const"

  如果您在 Windows 上编译 mmcv-full 并且 PyTorch 的版本是 1.6.0，您很可能会遇到这个问题 `"- torch/include\torch/csrc/jit/api/module.h(483): error: a member with an in-class initializer must be const"`. 解决这个问题的方法是将 `torch/include\torch/csrc/jit/api/module.h` 文件中的所有 `CONSTEXPR_EXCEPT_WIN_CUDA ` 替换为 `const`。更多细节可以查看 [Ninja: build stopped: subcommand failed](https://github.com/open-mmlab/mmcv/issues/575)。

- "error: member "torch::jit::ProfileOptionalOp::Kind" may not be initialized"

  如果您在 Windows 上编译 mmcv-full 并且 PyTorch 的版本是 1.7.0，您很可能会遇到这个问题 `torch/include\torch/csrc/jit/ir/ir.h(1347): error: member "torch::jit::ProfileOptionalOp::Kind" may not be initialized`. 解决这个问题的方法是修改 PyTorch 中的几个文件：

  - 删除 `torch/include\torch/csrc/jit/ir/ir.h` 文件中的 `static constexpr Symbol Kind = ::c10::prim::profile;` 和 `tatic constexpr Symbol Kind = ::c10::prim::profile_optional;`
  - 将 `torch\include\pybind11\cast.h` 文件中的 `explicit operator type&() { return *(this->value); }` 替换为 `explicit operator type&() { return *((type*)this->value); }`
  - 将 `torch/include\torch/csrc/jit/api/module.h` 文件中的 所有 `CONSTEXPR_EXCEPT_WIN_CUDA` 替换为 `const`

  更多细节可以查看 [Ensure default extra_compile_args](https://github.com/pytorch/pytorch/pull/45956)。

- MMCV 和 MMDetection 的兼容性问题；"ConvWS is already registered in conv layer"

  请参考 [installation instruction](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation) 为您的 MMDetection 版本安装正确版本的 MMCV。

### 使用问题

- "RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one"

  1. 这个错误是因为有些参数没有参与 loss 的计算，可能是代码中存在多个分支，导致有些分支没有参与 loss 的计算。更多细节见 [Expected to have finished reduction in the prior iteration before starting a new one](https://github.com/pytorch/pytorch/issues/55582)。
  2. 你可以设置 DDP 中的 `find_unused_parameters` 为 `True`，或者手动查找哪些参数没有用到。

- "RuntimeError: Trying to backward through the graph a second time"

  不能同时设置 `GradientCumulativeOptimizerHook` 和 `OptimizerHook`，这会导致 `loss.backward()` 被调用两次，于是程序抛出 `RuntimeError`。我们只需设置其中的一个。更多细节见 [Trying to backward through the graph a second time](https://github.com/open-mmlab/mmcv/issues/1379)。
