## Frequently Asked Questions

We list some common troubles faced by many users and their corresponding solutions here.
Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them.

### Installation

- KeyError: "xxx: 'yyy is not in the zzz registry'"

  The registry mechanism will be triggered only when the file of the module is imported.
  So you need to import that file somewhere. More details can be found at [KeyError: "MaskRCNN: 'RefineRoIHead is not in the models registry'"](https://github.com/open-mmlab/mmdetection/issues/5974).

- "No module named 'mmcv.ops'"; "No module named 'mmcv.\_ext'"

  1. Uninstall existing mmcv in the environment using `pip uninstall mmcv`
  2. Install mmcv-full following the [installation instruction](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) or [Build MMCV from source](https://mmcv.readthedocs.io/en/latest/get_started/build.html)

- "invalid device function" or "no kernel image is available for execution"

  1. Check the CUDA compute capability of you GPU
  2. Run `python mmdet/utils/collect_env.py` to check whether PyTorch, torchvision, and MMCV are built for the correct GPU architecture. You may need to set `TORCH_CUDA_ARCH_LIST` to reinstall MMCV. The compatibility issue could happen when  using old GPUS, e.g., Tesla K80 (3.7) on colab.
  3. Check whether the running environment is the same as that when mmcv/mmdet is compiled. For example, you may compile mmcv using CUDA 10.0 bug run it on CUDA9.0 environments

- "undefined symbol" or "cannot open xxx.so"

  1. If those symbols are CUDA/C++ symbols (e.g., libcudart.so or GLIBCXX), check
     whether the CUDA/GCC runtimes are the same as those used for compiling mmcv
  2. If those symbols are Pytorch symbols (e.g., symbols containing caffe, aten, and TH), check whether the Pytorch version is the same as that used for compiling mmcv
  3. Run `python mmdet/utils/collect_env.py` to check whether PyTorch, torchvision, and MMCV are built by and running on the same environment

- "RuntimeError: CUDA error: invalid configuration argument"

  This error may be caused by the poor performance of GPU. Try to decrease the value of [THREADS_PER_BLOCK](https://github.com/open-mmlab/mmcv/blob/cac22f8cf5a904477e3b5461b1cc36856c2793da/mmcv/ops/csrc/common_cuda_helper.hpp#L10)
  and recompile mmcv.

- "RuntimeError: nms is not compiled with GPU support"

  This error is because your CUDA environment is not installed correctly.
  You may try to re-install your CUDA environment and then delete the build/ folder before re-compile mmcv.

- "Segmentation fault"

  1. Check your GCC version and use GCC >= 5.4. This usually caused by the incompatibility between PyTorch and the environment (e.g., GCC \< 4.9 for PyTorch). We also recommend the users to avoid using GCC 5.5 because many feedbacks report that GCC 5.5 will cause "segmentation fault" and simply changing it to GCC 5.4 could solve the problem
  2. Check whether PyTorch is correctly installed and could use CUDA op, e.g. type the following command in your terminal and see whether they could correctly output results
     ```shell
     python -c 'import torch; print(torch.cuda.is_available())'
     ```
  3. If PyTorch is correctly installed, check whether MMCV is correctly installed. If MMCV is correctly installed, then there will be no issue of the command
     ```shell
     python -c 'import mmcv; import mmcv.ops'
     ```
  4. If MMCV and PyTorch are correctly installed, you can use `ipdb` to set breakpoints or directly add `print` to debug and see which part leads the `segmentation fault`

- "libtorch_cuda_cu.so: cannot open shared object file"

  `mmcv-full` depends on the share object but it can not be found. We can check whether the object exists in `~/miniconda3/envs/{environment-name}/lib/python3.7/site-packages/torch/lib` or try to re-install the PyTorch.

- "fatal error C1189: #error:  -- unsupported Microsoft Visual Studio version!"

  If you are building mmcv-full on Windows and the version of CUDA is 9.2, you will probably encounter the error `"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\include\crt/host_config.h(133): fatal error C1189: #error:  -- unsupported Microsoft Visual Studio version! Only the versions 2012, 2013, 2015 and 2017 are supported!"`, in which case you can use a lower version of Microsoft Visual Studio like vs2017.

- "error: member "torch::jit::detail::ModulePolicy::all_slots" may not be initialized"

  If your version of PyTorch is 1.5.0 and you are building mmcv-full on Windows, you will probably encounter the error `- torch/csrc/jit/api/module.h(474): error: member "torch::jit::detail::ModulePolicy::all_slots" may not be initialized`. The way to solve the error is to replace all the `static constexpr bool all_slots = false;` with `static bool all_slots = false;` at this file `https://github.com/pytorch/pytorch/blob/v1.5.0/torch/csrc/jit/api/module.h`. More details can be found at [member "torch::jit::detail::AttributePolicy::all_slots" may not be initialized](https://github.com/pytorch/pytorch/issues/39394).

- "error: a member with an in-class initializer must be const"

  If your version of PyTorch is 1.6.0 and you are building mmcv-full on Windows, you will probably encounter the error `"- torch/include\torch/csrc/jit/api/module.h(483): error: a member with an in-class initializer must be const"`. The way to solve the error is to replace all the `CONSTEXPR_EXCEPT_WIN_CUDA ` with `const` at `torch/include\torch/csrc/jit/api/module.h`. More details can be found at [Ninja: build stopped: subcommand failed](https://github.com/open-mmlab/mmcv/issues/575).

- "error: member "torch::jit::ProfileOptionalOp::Kind" may not be initialized"

  If your version of PyTorch is 1.7.0 and you are building mmcv-full on Windows, you will probably encounter the error `torch/include\torch/csrc/jit/ir/ir.h(1347): error: member "torch::jit::ProfileOptionalOp::Kind" may not be initialized`. The way to solve the error needs to modify several local files of PyTorch:

  - delete `static constexpr Symbol Kind = ::c10::prim::profile;` and `tatic constexpr Symbol Kind = ::c10::prim::profile_optional;` at `torch/include\torch/csrc/jit/ir/ir.h`
  - replace `explicit operator type&() { return *(this->value); }` with `explicit operator type&() { return *((type*)this->value); }` at `torch\include\pybind11\cast.h`
  - replace all the `CONSTEXPR_EXCEPT_WIN_CUDA` with `const` at `torch/include\torch/csrc/jit/api/module.h`

  More details can be found at [Ensure default extra_compile_args](https://github.com/pytorch/pytorch/pull/45956).

- Compatibility issue between MMCV and MMDetection; "ConvWS is already registered in conv layer"

  Please install the correct version of MMCV for the version of your MMDetection following the [installation instruction](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation).

### Usage

- "RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one"

  1. This error indicates that your module has parameters that were not used in producing loss. This phenomenon may be caused by running different branches in your code in DDP mode. More datails at [Expected to have finished reduction in the prior iteration before starting a new one](https://github.com/pytorch/pytorch/issues/55582).
  2. You can set ` find_unused_parameters = True` in the config to solve the above problems or find those unused parameters manually

- "RuntimeError: Trying to backward through the graph a second time"

  `GradientCumulativeOptimizerHook` and `OptimizerHook` are both set which causes the `loss.backward()` to be called twice so `RuntimeError` was raised. We can only use one of these. More datails at [Trying to backward through the graph a second time](https://github.com/open-mmlab/mmcv/issues/1379).
