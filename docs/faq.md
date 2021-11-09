## Frequently Asked Questions

We list some common troubles faced by many users and their corresponding solutions here.
Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them.

### Installation

- Compatibility issue between MMCV and MMDetection; "ConvWS is already registered in conv layer"

    Please install the correct version of MMCV for the version of your MMDetection following the instruction above.

- "No module named 'mmcv.ops'"; "No module named 'mmcv._ext'"

    1. Uninstall existing mmcv in the environment using `pip uninstall mmcv`.

    2. Install mmcv-full following the instruction above.

- "invalid device function" or "no kernel image is available for execution"

    1. Check the CUDA compute capability of you GPU.

    2. Run `python mmdet/utils/collect_env.py` to check whether PyTorch, torchvision,
       and MMCV are built for the correct GPU architecture.
       You may need to set `TORCH_CUDA_ARCH_LIST` to reinstall MMCV.
       The compatibility issue could happen when  using old GPUS, e.g., Tesla K80 (3.7) on colab.

    3. Check whether the running environment is the same as that when mmcv/mmdet is compiled.
       For example, you may compile mmcv using CUDA 10.0 bug run it on CUDA9.0   environments.

- "undefined symbol" or "cannot open xxx.so"

    1. If those symbols are CUDA/C++ symbols (e.g., libcudart.so or GLIBCXX), check
       whether the CUDA/GCC runtimes are the same as those used for compiling mmcv.

    2. If those symbols are Pytorch symbols (e.g., symbols containing caffe, aten, and TH), check whether
       the Pytorch version is the same as that used for compiling mmcv.

    3. Run `python mmdet/utils/collect_env.py` to check whether PyTorch, torchvision,
       and MMCV are built by and running on the same environment.

- "RuntimeError: CUDA error: invalid configuration argument"

    This error may be due to your poor GPU. Try to decrease the value of [THREADS_PER_BLOCK](https://github.com/open-mmlab/mmcv/blob/cac22f8cf5a904477e3b5461b1cc36856c2793da/mmcv/ops/csrc/common_cuda_helper.hpp#L10)
    and recompile mmcv.

- "RuntimeError: nms is not compiled with GPU support"

    This error is because your CUDA environment is not installed correctly.
    You may try to re-install your CUDA environment and then delete the build/ folder before re-compile mmcv.

- "Segmentation fault"

    1. Check you GCC version and use GCC 5.4. This usually caused by the incompatibility between PyTorch and the environment (e.g., GCC < 4.9 for PyTorch). We also recommend the users to avoid using GCC 5.5 because many feedbacks report that GCC 5.5 will cause "segmentation fault" and simply changing it to GCC 5.4 could solve the problem.

    2. Check whether PyTorch is correctly installed and could use CUDA op, e.g. type the following command in your terminal.

        ```shell
        python -c 'import torch; print(torch.cuda.is_available())'
        ```

        And see whether they could correctly output results.

    3. If Pytorch is correctly installed, check whether MMCV is correctly installed.

        ```shell
        python -c 'import mmcv; import mmcv.ops'
        ```

        If MMCV is correctly installed, then there will be no issue of the above two commands.

    4. If MMCV and Pytorch is correctly installed, you man use `ipdb`, `pdb` to set breakpoints or directly add 'print' to debug and see which part leads the segmentation fault.

- "libtorch_cuda_cu.so: cannot open shared object file"

    `mmcv-full` depends on the share object but it can not be found. We can check whether the object exists in `~/miniconda3/envs/{environment-name}/lib/python3.7/site-packages/torch/lib` or directly re-install the PyTorch.

### Usage

- "RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one"

    1. This error indicates that your module has parameters that were not used in producing loss. This phenomenon may be caused by running different branches in your code in DDP mode. More datails at https://github.com/pytorch/pytorch/issues/55582.

    2. You can set ` find_unused_parameters = True` in the config to solve the above problems or find those unused parameters manually.

- "RuntimeError: Trying to backward through the graph a second time"

   `GradientCumulativeOptimizerHook` and `OptimizerHook` are both set which causing `loss.backward()` to be called twice so `RuntimeError` was raised. We can only use one of these. More datails at https://github.com/open-mmlab/mmcv/issues/1379.
