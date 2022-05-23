## 从源码编译 MMCV

### 在 Linux 或者 macOS 上编译 MMCV

克隆算法库

```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
```

建议安装 `ninja` 以加快编译速度

```bash
pip install -r requirements/optional.txt
```

你可以安装 lite 版本

```bash
pip install -e .
```

也可以安装 full 版本

```bash
MMCV_WITH_OPS=1 pip install -e .
```

如果是在 macOS 上编译，则需要在安装命令前添加一些环境变量

```bash
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++'
```

例如

```bash
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' MMCV_WITH_OPS=1 pip install -e .
```

```{note}
如果你打算使用 `opencv-python-headless` 而不是 `opencv-python`，例如在一个很小的容器环境或者没有图形用户界面的服务器中，你可以先安装 `opencv-python-headless`，这样在安装 mmcv 依赖的过程中会跳过 `opencv-python`
```

### 在 Windows 上编译 MMCV

在 Windows 上编译 MMCV 比 Linux 复杂，本节将一步步介绍如何在 Windows 上编译 MMCV。

#### 依赖项

请首先安装以下的依赖项：

- [Git](https://git-scm.com/download/win)：安装期间，请选择 **add git to Path**
- [Visual Studio Community 2019](https://visualstudio.microsoft.com)：用于编译 C++ 和 CUDA 代码
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)：包管理工具
- [CUDA 10.2](https://developer.nvidia.com/cuda-10.2-download-archive)：如果只需要 CPU 版本可以不安装 CUDA，安装CUDA时，可根据需要进行自定义安装。如果已经安装新版本的显卡驱动，建议取消驱动程序的安装

```{note}
您需要知道如何在 Windows 上设置变量环境，尤其是 "PATH" 的设置，以下安装过程都会用到。
```

#### 设置 Python 环境

1. 从 Windows 菜单启动 Anaconda 命令行

```{note}
如 Miniconda 安装程序建议，不要使用原始的 `cmd.exe` 或是 `powershell.exe`。命令行有两个版本，一个基于 PowerShell，一个基于传统的 `cmd.exe`。请注意以下说明都是使用的基于 PowerShell
```

2. 创建一个新的 Conda 环境

   ```shell
   conda create --name mmcv python=3.7  # 经测试，3.6, 3.7, 3.8 也能通过
   conda activate mmcv  # 确保做任何操作前先激活环境
   ```

3. 安装 PyTorch 时，可以根据需要安装支持 CUDA 或不支持 CUDA 的版本

   ```shell
   # CUDA version
   conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
   # CPU version
   conda install pytorch torchvision cpuonly -c pytorch
   ```

4. 准备 MMCV 源代码

   ```shell
   git clone https://github.com/open-mmlab/mmcv.git
   cd mmcv
   ```

5. 安装所需 Python 依赖包

   ```shell
   pip3 install -r requirements/runtime.txt
   ```

6. 建议安装 `ninja` 以加快编译速度

   ```bash
   pip install -r requirements/optional.txt
   ```

#### 编译与安装 MMCV

MMCV 有三种安装的模式：

1. Lite 版本（不包含算子）

   这种方式下，没有算子被编译，这种模式的 mmcv 是原生的 python 包

2. Full 版本（只包含 CPU 算子）

   编译 CPU 算子，但只有 x86 将会被编译，并且编译版本只能在 CPU only 情况下运行

3. Full 版本（既包含 CPU 算子，又包含 CUDA 算子）

   同时编译 CPU 和 CUDA 算子，`ops` 模块的 x86 与 CUDA 的代码都可以被编译。同时编译的版本可以在 CUDA 上调用 GPU

##### 通用步骤

1. 设置 MSVC 编译器

   设置环境变量。添加 `C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\Hostx86\x64` 到 `PATH`，则 `cl.exe` 可以在命令行中运行，如下所示。

   ```none
   (base) PS C:\Users\xxx> cl
   Microsoft (R) C/C++ Optimizing  Compiler Version 19.27.29111 for x64
   Copyright (C) Microsoft Corporation.   All rights reserved.

   usage: cl [ option... ] filename... [ / link linkoption... ]
   ```

   为了兼容性，我们使用 x86-hosted 以及 x64-targeted 版本，即路径中的 `Hostx86\x64` 。

   因为 PyTorch 将解析 `cl.exe` 的输出以检查其版本，只有 utf-8 将会被识别，你可能需要将系统语言更改为英语。控制面板 -> 地区-> 管理-> 非 Unicode 来进行语言转换。

##### 安装方式一：Lite version（不包含算子）

在完成上述的公共步骤后，从菜单打开 Anaconda 命令框，输入以下命令

```shell
# 激活环境
conda activate mmcv
# 切换到 mmcv 根目录
cd mmcv
# 安装
python setup.py develop
# 检查是否安装成功
pip list
```

##### 安装方式二：Full version（只编译 CPU 算子）

1. 完成上述的公共步骤

2. 设置环境变量

   ```shell
   $env:MMCV_WITH_OPS = 1
   $env:MAX_JOBS = 8  # 根据你可用CPU以及内存量进行设置
   ```

3. 编译安装

   ```shell
   conda activate mmcv  # 激活环境
   cd mmcv  # 改变路径
   python setup.py build_ext  # 如果成功, cl 将被启动用于编译算子
   python setup.py develop  # 安装
   pip list  # 检查是否安装成功
   ```

##### 安装方式三：Full version（既编译 CPU 算子又编译 CUDA 算子）

1. 完成上述的公共步骤

2. 设置环境变量

   ```shell
   $env:MMCV_WITH_OPS = 1
   $env:MAX_JOBS = 8  # 根据你可用CPU以及内存量进行设置
   ```

3. 检查 `CUDA_PATH` 或者 `CUDA_HOME` 环境变量已经存在在 `envs` 之中

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

4. 设置 CUDA 的目标架构

   ```shell
   $env:TORCH_CUDA_ARCH_LIST="6.1" # 支持 GTX 1080
   # 或者用所有支持的版本，但可能会变得很慢
   $env:TORCH_CUDA_ARCH_LIST="3.5 3.7 5.0 5.2 6.0 6.1 7.0 7.5"
   ```

```{note}
我们可以在 [here](https://developer.nvidia.com/cuda-gpus) 查看 GPU 的计算能力
```

5. 编译安装

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

如果编译安装 mmcv 的过程中遇到了问题，你也许可以在 [Frequently Asked Question](../faq.html) 找到解决方法
