## 从源码编译 MMCV

### 编译 mmcv-full

在编译 mmcv-full 之前，请确保 PyTorch 已经成功安装在环境中，可以参考 [PyTorch 官方安装文档](https://pytorch.org/get-started/locally/#start-locally)。可使用以下命令验证

```bash
python -c 'import torch;print(torch.__version__)'
```

```{note}
- 如需编译 ONNX Runtime 自定义算子，请参考[如何编译ONNX Runtime自定义算子](https://mmcv.readthedocs.io/zh_CN/latest/deployment/onnxruntime_op.html#id1)
- 如需编译 TensorRT 自定义，请参考[如何编译MMCV中的TensorRT插件](https://mmcv.readthedocs.io/zh_CN/latest/deployment/tensorrt_plugin.html#id3)
```

:::{note}

- 如果克隆代码仓库的速度过慢，可以使用以下命令克隆（注意：gitee 的 mmcv 不一定和 github 的保持一致，因为每天只同步一次）

```bash
git clone https://gitee.com/open-mmlab/mmcv.git
```

- 如果打算使用 `opencv-python-headless` 而不是 `opencv-python`，例如在一个很小的容器环境或者没有图形用户界面的服务器中，你可以先安装 `opencv-python-headless`，这样在安装 mmcv 依赖的过程中会跳过 `opencv-python`。

- 如果编译过程安装依赖库的时间过长，可以[设置 pypi 源](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

:::

#### 在 Linux 上编译 mmcv-full

| TODO: 视频教程

1. 克隆代码仓库

   ```bash
   git clone https://github.com/open-mmlab/mmcv.git
   cd mmcv
   ```

2. 安装 `ninja` 和 `psutil` 以加快编译速度

   ```bash
   pip install -r requirements/optional.txt
   ```

3. 检查 nvcc 的版本（要求大于等于 9.2，如果没有 GPU，可以跳过）

   ```bash
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

4. 检查 gcc 的版本（要求大于等于**5.4**）

   ```bash
   gcc --version
   ```

5. 开始编译（预估耗时 10 分钟）

   ```bash
   MMCV_WITH_OPS=1 pip install -e . -v
   ```

6. 验证安装

   ```bash
   python .dev_scripts/check_installation.py
   ```

   如果上述命令没有报错，说明安装成功。如有报错，请查看[问题解决页面](https://mmcv.readthedocs.io/zh_CN/latest/faq.html)是否已经有解决方案。

   如果没有找到解决方案，欢迎提 [issue](https://github.com/open-mmlab/mmcv/issues)。

#### 在 macOS 上编译 mmcv-full

| TODO: 视频教程

```{note}
如果你使用的 mac 是 M1 芯片，请安装 PyTorch 的 nightly 版本，否则会遇到 [issues#2218](https://github.com/open-mmlab/mmcv/issues/2218) 中的问题。
```

1. 克隆代码仓库

   ```bash
   git clone https://github.com/open-mmlab/mmcv.git
   cd mmcv
   ```

2. 安装 `ninja` 和 `psutil` 以加快编译速度

   ```bash
   pip install -r requirements/optional.txt
   ```

3. 开始编译

   ```bash
   MMCV_WITH_OPS=1 pip install -e .
   ```

4. 验证安装

   ```bash
   python .dev_scripts/check_installation.py
   ```

   如果上述命令没有报错，说明安装成功。如有报错，请查看[问题解决页面](../faq.md)是否已经有解决方案。

   如果没有找到解决方案，欢迎提 [issue](https://github.com/open-mmlab/mmcv/issues)。

#### 在 Windows 上编译 mmcv-full

| TODO: 视频教程

在 Windows 上编译 mmcv-full 比 Linux 复杂，本节将一步步介绍如何在 Windows 上编译 mmcv-full。

##### 依赖项

请先安装以下的依赖项：

- [Git](https://git-scm.com/download/win)：安装期间，请选择 **add git to Path**
- [Visual Studio Community 2019](https://visualstudio.microsoft.com)：用于编译 C++ 和 CUDA 代码
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)：包管理工具
- [CUDA 10.2](https://developer.nvidia.com/cuda-10.2-download-archive)：如果只需要 CPU 版本可以不安装 CUDA，安装 CUDA 时，可根据需要进行自定义安装。如果已经安装新版本的显卡驱动，建议取消驱动程序的安装

```{note}
如果不清楚如何安装以上依赖，请参考[Windows 环境从零安装 mmcv-full](https://zhuanlan.zhihu.com/p/434491590)。
另外，你需要知道如何在 Windows 上设置变量环境，尤其是 "PATH" 的设置，以下安装过程都会用到。
```

##### 通用步骤

1. 从 Windows 菜单启动 Anaconda 命令行

   如 Miniconda 安装程序建议，不要使用原始的 `cmd.exe` 或是 `powershell.exe`。命令行有两个版本，一个基于 PowerShell，一个基于传统的 `cmd.exe`。请注意以下说明都是使用的基于 PowerShell

2. 创建一个新的 Conda 环境

   ```powershell
   (base) PS C:\Users\xxx> conda create --name mmcv python=3.7
   (base) PS C:\Users\xxx> conda activate mmcv  # 确保做任何操作前先激活环境
   ```

3. 安装 PyTorch 时，可以根据需要安装支持 CUDA 或不支持 CUDA 的版本

   ```powershell
   # CUDA version
   (mmcv) PS C:\Users\xxx> conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
   # CPU version
   (mmcv) PS C:\Users\xxx> conda install install pytorch torchvision cpuonly -c pytorch
   ```

4. 克隆代码仓库

   ```powershell
   (mmcv) PS C:\Users\xxx> git clone https://github.com/open-mmlab/mmcv.git
   (mmcv) PS C:\Users\xxx> cd mmcv
   ```

5. 安装 `ninja` 和 `psutil` 以加快编译速度

   ```powershell
   (mmcv) PS C:\Users\xxx\mmcv> pip install -r requirements/optional.txt
   ```

6. 安装 mmcv 依赖

   ```powershell
   (mmcv) PS C:\Users\xxx\mmcv> pip install -r requirements/runtime.txt
   ```

7. 设置 MSVC 编译器

   设置环境变量。添加 `C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\Hostx86\x64` 到 `PATH`，则 `cl.exe` 可以在命令行中运行，如下所示。

   ```powershell
   (mmcv) PS C:\Users\xxx\mmcv> cl
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

1. 设置环境变量

   ```powershell
   (mmcv) PS C:\Users\xxx\mmcv> $env:MMCV_WITH_OPS = 1
   ```

2. 编译安装

   ```powershell
   (mmcv) PS C:\Users\xxx\mmcv> python setup.py build_ext  # 如果成功, cl 将被启动用于编译算子
   (mmcv) PS C:\Users\xxx\mmcv> python setup.py develop  # 安装
   ```

###### GPU 版本

1. 设置环境变量

   ```powershell
   (mmcv) PS C:\Users\xxx\mmcv> $env:MMCV_WITH_OPS = 1
   ```

2. 检查 `CUDA_PATH` 或者 `CUDA_HOME` 环境变量已经存在在 `envs` 之中

   ```powershell
   (mmcv) PS C:\Users\xxx\mmcv> ls env:

   Name                           Value
   ----                           -----
   CUDA_PATH                      C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2
   CUDA_PATH_V10_1                C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1
   CUDA_PATH_V10_2                C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2
   ```

   如果没有，你可以按照下面的步骤设置

   ```powershell
   (mmcv) PS C:\Users\xxx\mmcv> $env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2"
   # 或者
   (mmcv) PS C:\Users\xxx\mmcv> $env:CUDA_HOME = $env:CUDA_PATH_V10_2  # CUDA_PATH_V10_2 已经在环境变量中
   ```

3. 设置 CUDA 的目标架构

   ```powershell
   # 这里需要改成你的显卡对应的目标架构
   (mmcv) PS C:\Users\xxx\mmcv> $env:TORCH_CUDA_ARCH_LIST="7.5"
   ```

   :::{note}
   可以点击 [cuda-gpus](https://developer.nvidia.com/cuda-gpus) 查看 GPU 的计算能力，也可以通过 CUDA 目录下的 deviceQuery.exe 工具查看

   ```powershell
   (mmcv) PS C:\Users\xxx\mmcv> &"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\extras\demo_suite\deviceQuery.exe"
   Device 0: "NVIDIA GeForce GTX 1660 SUPER"
   CUDA Driver Version / Runtime Version          11.7 / 11.1
   CUDA Capability Major/Minor version number:    7.5
   ```

   上面的 7.5 表示目标架构。注意：需把上面命令的 v10.2 换成你的 CUDA 版本。
   :::

4. 编译安装

   ```powershell
   (mmcv) PS C:\Users\xxx\mmcv> python setup.py build_ext  # 如果成功, cl 将被启动用于编译算子
   (mmcv) PS C:\Users\xxx\mmcv> python setup.py develop # 安装
   ```

   ```{note}
   如果你的 PyTorch 版本是 1.6.0，你可能会遇到一些 [issue](https://github.com/pytorch/pytorch/issues/42467) 提到的错误，你可以参考这个 [pull request](https://github.com/pytorch/pytorch/pull/43380/files) 修改本地环境的 PyTorch 源代码
   ```

##### 验证安装

```powershell
(mmcv) PS C:\Users\xxx\mmcv> python .dev_scripts/check_installation.py
```

如果上述命令没有报错，说明安装成功。如有报错，请查看[问题解决页面](../faq.md)是否已经有解决方案。
如果没有找到解决方案，欢迎提 [issue](https://github.com/open-mmlab/mmcv/issues)。

### 编译 mmcv

如果你需要使用和 PyTorch 相关的模块，请确保 PyTorch 已经成功安装在环境中，可以参考 [PyTorch 官方安装文档](https://pytorch.org/get-started/locally/#start-locally)。

1. 克隆代码仓库

   ```bash
   git clone https://github.com/open-mmlab/mmcv.git
   cd mmcv
   ```

2. 开始编译

   ```bash
   pip install -e . -v
   ```

3. 验证安装

   ```bash
   python -c 'import mmcv;print(mmcv.__version__)'
   ```

### 在 IPU 机器编译 mmcv

首先你需要有可用的 IPU 云机器，可以查看[这里](https://www.graphcore.ai/ipus-in-the-cloud)。

#### 选项1: 使用 Docker

1. 拉取镜像

   ```bash
   docker pull graphcore/pytorch
   ```

2. 编译 mmcv

#### 选项2: 使用 SDK

1. 编译 mmcv

2. 参考 [IPU PyTorch document](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/installation.html) 安装 sdk。

### 在昇腾 NPU 机器编译 mmcv-full

在编译 mmcv-full 前，需要安装 torch_npu，完整安装教程详见 [PyTorch 安装指南](https://gitee.com/ascend/pytorch/blob/master/docs/zh/PyTorch%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97/PyTorch%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97.md#pytorch%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97)

#### 选项 1: 使用 pip 安装 Ascend 编译版本的 mmcv-full

Ascend 编译版本的 mmcv-full 在 mmcv >= 1.7.0 时已经支持直接 pip 安装

```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/ascend/torch1.8.0/index.html
```

#### 选项 2: 使用 NPU 设备源码编译安装 mmcv-full

- 拉取 [MMCV 源码](https://github.com/open-mmlab/mmcv/tree/master)

```bash
git pull https://github.com/open-mmlab/mmcv/tree/master
```

- 编译

```bash
MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
```

- 安装

```bash
MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
```

#### 验证

```python
import torch
import torch_npu
from mmcv.ops import softmax_focal_loss

# Init tensor to the NPU
x = torch.randn(3, 10).npu()
y = torch.tensor([1, 5, 3]).npu()
w = torch.ones(10).float().npu()

output = softmax_focal_loss(x, y, 2.0, 0.25, w, 'none')
print(output)
```

### 在寒武纪 MLU 机器编译 mmcv-full

#### 安装 torch_mlu

##### 选项1: 基于寒武纪 docker image 安装

首先请下载并且拉取寒武纪 docker (请向 service@cambricon.com 发邮件以获得最新的寒武纪 pytorch 发布 docker)。

```
docker pull ${docker image}
```

进入 docker, [编译 MMCV MLU](#编译mmcv-mlu) 并[进行验证](#验证是否成功安装)。

##### 选项2：基于 cambricon pytorch 源码编译安装

请向 service@cambricon.com 发送邮件或联系 Cambricon 工程师以获取合适版本的 CATCH 软件包，在您获得合适版本的 CATCH 软件包后，请参照 ${CATCH-path}/CONTRIBUTING.md 中的步骤安装 CATCH。

#### 编译 MMCV

克隆代码仓库

```bash
git clone https://github.com/open-mmlab/mmcv.git
```

算子库 mlu-ops 在编译 MMCV 时自动下载到默认路径(mmcv/mlu-ops)，你也可以在编译前设置环境变量 MMCV_MLU_OPS_PATH 指向已经存在的 mlu-ops 算子库路径。

```bash
export MMCV_MLU_OPS_PATH=/xxx/xxx/mlu-ops
```

开始编译

```bash
cd mmcv
export MMCV_WITH_OPS=1
export FORCE_MLU=1
python setup.py install
```

#### 验证是否成功安装

完成上述安装步骤之后，您可以尝试运行下面的 Python 代码以测试您是否成功在 MLU 设备上安装了 mmcv-full

```python
import torch
import torch_mlu
from mmcv.ops import sigmoid_focal_loss
x = torch.randn(3, 10).mlu()
x.requires_grad = True
y = torch.tensor([1, 5, 3]).mlu()
w = torch.ones(10).float().mlu()
output = sigmoid_focal_loss(x, y, 2.0, 0.25, w, 'none')
```
