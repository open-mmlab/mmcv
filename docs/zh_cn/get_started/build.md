## 从源码编译 MMCV

### 编译 mmcv-full

在编译 mmcv-full 之前，请确保 PyTorch 已经成功安装在环境中，可以参考 [PyTorch 官方安装文档](https://github.com/pytorch/pytorch#installation)。可使用以下命令验证

```bash
python -c 'import torch;print(torch.__version__)'
```

```{note}
1. 如需编译 ONNX Runtime 自定义算子，请参考[如何编译ONNX Runtime自定义算子？](https://mmcv.readthedocs.io/zh_CN/latest/deployment/onnxruntime_op.html#id1)
2. 如需编译 TensorRT 自定义，请参考[如何编译MMCV中的TensorRT插件](https://mmcv.readthedocs.io/zh_CN/latest/deployment/tensorrt_plugin.html#id3)
```

#### 在 Linux 上编译 mmcv-full

| TODO: 视频教程

- 克隆代码仓库

```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
```

:::{note}
如果克隆代码仓库的速度过慢，可以使用以下命令克隆

```bash
git clone https://gitee.com/open-mmlab/mmcv.git
```

需要注意注意的是：gitee 上的 mmcv 不一定和 github 上的保持一致，因为每天只同步一次。
:::

- 安装 `ninja` 和 `psutil` 以加快编译速度

```bash
pip install -r requirements/optional.txt
```

- 检查 nvcc 是否正确设置（如果没有 GPU，可以跳过）

```
nvcc --version
```

上述命令如果输出以下信息，表示 nvcc 的设置没有问题，否则需要设置 CUDA_HOME

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Mon_Nov_30_19:08:53_PST_2020
Cuda compilation tools, release 11.2, V11.2.67
Build cuda_11.2.r11.2/compiler.29373293_0
```

:::{note}
如果想要支持 ROCm，可以参考 [AMD ROCm](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html) 安装 ROCm。
:::

- 检查 gcc 的版本是否符合要求（大于等于5.4）

```bash
gcc --version
```

- 开始编译（预估耗时 10 分钟）

```bash
MMCV_WITH_OPS=1 pip install -e . -v
```

:::{note}
如果你打算使用 `opencv-python-headless` 而不是 `opencv-python`，例如在一个很小的容器环境或者没有图形用户界面的服务器中，你可以先安装 `opencv-python-headless`，这样在安装 mmcv 依赖的过程中会跳过 `opencv-python`。

另外，如果编译过程安装依赖库的时间过长，可以指定 pypi 源

```bash
MMCV_WITH_OPS=1 pip install -e . -v -i https://pypi.tuna.tsinghua.edu.cn/simple
```

:::

- 验证安装

```bash
python .dev_scripts/check_installation.py
```

如果上述命令没有报错，说明安装成功。如有报错，请参考[问题解决页面](https://mmcv.readthedocs.io/zh_CN/latest/faq.html)查看是否已经有解决方案。
如果没有找到解决方案，欢迎提 [issue](https://github.com/open-mmlab/mmcv/issues)。

#### 在 macOS 上编译 mmcv-full

| TODO: 视频教程以及 MPS 的编译步骤

- 克隆代码仓库

```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
```

:::{note}
如果克隆代码仓库的速度过慢，可以使用以下命令克隆

```bash
git clone https://gitee.com/open-mmlab/mmcv.git
```

需要注意注意的是：gitee 上的 mmcv 不一定和 github 上的保持一致，因为每天只同步一次。
:::

- 安装 `ninja` 和 `psutil` 以加快编译速度

```bash
pip install -r requirements/optional.txt
```

- 开始编译

```bash
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' MMCV_WITH_OPS=1 pip install -e .
```

:::{note}
如果你打算使用 `opencv-python-headless` 而不是 `opencv-python`，例如在一个很小的容器环境或者没有图形用户界面的服务器中，你可以先安装 `opencv-python-headless`，这样在安装 mmcv 依赖的过程中会跳过 `opencv-python`。

另外，如果编译过程安装依赖库的时间过长，可以指定 pypi 源

```bash
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' MMCV_WITH_OPS=1 pip install -e . -v -i https://pypi.tuna.tsinghua.edu.cn/simple
```

:::

- 验证安装

```bash
python .dev_scripts/check_installation.py
```

如果上述命令没有报错，说明安装成功。如有报错，请参考[问题解决页面](https://mmcv.readthedocs.io/zh_CN/latest/faq.html#id2)查看是否已经有解决方案。
如果没有找到解决方案，欢迎提 [issue](https://github.com/open-mmlab/mmcv/issues)。

#### 在 Windows 上编译 mmcv-full

| TODO: 视频教程

在 Windows 上编译 mmcv-full 比 Linux 复杂，本节将一步步介绍如何在 Windows 上编译 mmcv-full。

##### 依赖项

请首先安装以下的依赖项：

- [Git](https://git-scm.com/download/win)：安装期间，请选择 **add git to Path**
- [Visual Studio Community 2019](https://visualstudio.microsoft.com)：用于编译 C++ 和 CUDA 代码
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)：包管理工具
- [CUDA 10.2](https://developer.nvidia.com/cuda-10.2-download-archive)：如果只需要 CPU 版本可以不安装 CUDA，安装CUDA时，可根据需要进行自定义安装。如果已经安装新版本的显卡驱动，建议取消驱动程序的安装

```{note}
您需要知道如何在 Windows 上设置变量环境，尤其是 "PATH" 的设置，以下安装过程都会用到。
```

##### 设置 Python 环境

- 从 Windows 菜单启动 Anaconda 命令行

如 Miniconda 安装程序建议，不要使用原始的 `cmd.exe` 或是 `powershell.exe`。命令行有两个版本，一个基于 PowerShell，一个基于传统的 `cmd.exe`。请注意以下说明都是使用的基于 PowerShell

- 创建一个新的 Conda 环境

```shell
conda create --name mmcv python=3.7  # 经测试，3.6, 3.7, 3.8 也能通过
conda activate mmcv  # 确保做任何操作前先激活环境
```

- 安装 PyTorch 时，可以根据需要安装支持 CUDA 或不支持 CUDA 的版本

```shell
# CUDA version
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
# CPU version
conda install pytorch torchvision cpuonly -c pytorch
```

- 克隆代码仓库

```shell
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
```

:::{note}
如果克隆代码仓库的速度过慢，可以使用以下命令克隆

```bash
git clone https://gitee.com/open-mmlab/mmcv.git
```

需要注意注意的是：gitee 上的 mmcv 不一定和 github 上的保持一致，因为每天只同步一次。
:::

- 安装 `ninja` 和 `psutil` 以加快编译速度

```bash
pip install -r requirements/optional.txt
```

- 安装 mmcv 依赖

```
pip install -r requirements/runtime.txt
```

:::{note}

如果安装依赖库的时间过长，可以指定 pypi 源

```bash
pip install -r requirements/runtime.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

:::

##### 设置 MSVC 编译器

设置环境变量。添加 `C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\Hostx86\x64` 到 `PATH`，则 `cl.exe` 可以在命令行中运行，如下所示。

```none
(base) PS C:\Users\xxx> cl
Microsoft (R) C/C++ Optimizing  Compiler Version 19.27.29111 for x64
Copyright (C) Microsoft Corporation.   All rights reserved.

usage: cl [ option... ] filename... [ / link linkoption... ]
```

为了兼容性，我们使用 x86-hosted 以及 x64-targeted 版本，即路径中的 `Hostx86\x64` 。

因为 PyTorch 将解析 `cl.exe` 的输出以检查其版本，只有 utf-8 将会被识别，你可能需要将系统语言更改为英语。控制面板 -> 地区-> 管理-> 非 Unicode 来进行语言转换。

##### 编译与安装 mmcv-full

mmcv-full 有两个版本：

- 只包含 CPU 算子的版本

  编译 CPU 算子，但只有 x86 将会被编译，并且编译版本只能在 CPU only 情况下运行

- 既包含 CPU 算子，又包含 CUDA 算子的版本

  同时编译 CPU 和 CUDA 算子，`ops` 模块的 x86 与 CUDA 的代码都可以被编译。同时编译的版本可以在 CUDA 上调用 GPU

###### CPU 版本

- 设置环境变量

```shell
$env:MMCV_WITH_OPS = 1
$env:MAX_JOBS = 8  # 根据你可用CPU以及内存量进行设置
```

- 编译安装

```shell
conda activate mmcv  # 激活环境
cd mmcv  # 改变路径
python setup.py build_ext  # 如果成功, cl 将被启动用于编译算子
python setup.py develop  # 安装
pip list  # 检查是否安装成功
```

###### GPU 版本

- 设置环境变量

```shell
$env:MMCV_WITH_OPS = 1
$env:MAX_JOBS = 8  # 根据你可用CPU以及内存量进行设置
```

- 检查 `CUDA_PATH` 或者 `CUDA_HOME` 环境变量已经存在在 `envs` 之中

```none
(base) PS C:\Users\WRH> ls env:

Name                           Value
----                           -----
CUDA_PATH                      C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2
CUDA_PATH_V10_1                C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1
CUDA_PATH_V10_2                C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2
```

如果没有，你可以按照下面的步骤设置

```shell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2"
# 或者
$env:CUDA_HOME = $env:CUDA_PATH_V10_2  # CUDA_PATH_V10_2 已经在环境变量中
```

- 设置 CUDA 的目标架构

```shell
$env:TORCH_CUDA_ARCH_LIST="6.1" # 支持 GTX 1080
# 或者用所有支持的版本，但可能会变得很慢
$env:TORCH_CUDA_ARCH_LIST="3.5 3.7 5.0 5.2 6.0 6.1 7.0 7.5"
```

```{note}
我们可以点击[这里](https://developer.nvidia.com/cuda-gpus)查看 GPU 的计算能力
```

- 编译安装

```shell
$env:MMCV_WITH_OPS = 1
$env:MAX_JOBS = 8 # 根据你可用CPU以及内存量进行设置
conda activate mmcv # 激活环境
cd mmcv  # 改变路径
python setup.py build_ext  # 如果成功, cl 将被启动用于编译算子
python setup.py develop # 安装
pip list # 检查是否安装成功
```

```{note}
如果你的 PyTorch 版本是 1.6.0，你可能会遇到一些这个 [issue](https://github.com/pytorch/pytorch/issues/42467) 提到的错误，则可以参考这个 [pull request](https://github.com/pytorch/pytorch/pull/43380/files) 修改 本地环境的 PyTorch 源代码
```

##### 验证安装

```bash
python .dev_scripts/check_installation.py
```

如果上述命令没有报错，说明安装成功。如有报错，请参考[问题解决页面](https://mmcv.readthedocs.io/zh_CN/latest/faq.html)查看是否已经有解决方案。
如果没有找到解决方案，欢迎提 [issue](https://github.com/open-mmlab/mmcv/issues)。

### 编译 mmcv

如果你需要使用和 PyTorch 相关的模块，请确保 PyTorch 已经成功安装在环境中，可以参考 [PyTorch 官方安装文档](https://github.com/pytorch/pytorch#installation)。

- 克隆代码仓库

```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
```

:::{note}
如果克隆代码仓库的速度过慢，可以使用以下命令克隆

```bash
git clone https://gitee.com/open-mmlab/mmcv.git
```

需要注意注意的是：gitee 上的 mmcv 不一定和 github 上的保持一致，因为每天只同步一次。
:::

- 开始编译

```
pip install -e . -v
```

:::{note}
如果你打算使用 `opencv-python-headless` 而不是 `opencv-python`，例如在一个很小的容器环境或者没有图形用户界面的服务器中，你可以先安装 `opencv-python-headless`，这样在安装 mmcv 依赖的过程中会跳过 `opencv-python`。

另外，如果编译过程安装依赖库的时间过长，可以指定 pypi 源

```bash
MMCV_WITH_OPS=1 pip install -e . -v -i https://pypi.tuna.tsinghua.edu.cn/simple
```

:::

- 验证安装

```
python -c 'import mmcv;print(mmcv.__version__)'
```
