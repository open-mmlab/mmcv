## Build MMCV from source

### Build mmcv-full

Before installing mmcv-full, make sure that PyTorch has been successfully installed following the [PyTorch official installation guide](https://pytorch.org/get-started/locally/#start-locally). This can be verified using the following command

```bash
python -c 'import torch;print(torch.__version__)'
```

If version information is output, then PyTorch is installed.

```{note}
- To compile ONNX Runtime custom operators, please refer to [How to build custom operators for ONNX Runtime](https://mmcv.readthedocs.io/en/latest/deployment/onnxruntime_op.html#how-to-build-custom-operators-for-onnx-runtime)
- To compile TensorRT customization, please refer to [How to build TensorRT plugins in MMCV](https://mmcv.readthedocs.io/en/latest/deployment/tensorrt_plugin.html#how-to-build-tensorrt-plugins-in-mmcv)
```

```{note}
If you would like to use `opencv-python-headless` instead of `opencv-python`,
e.g., in a minimum container environment or servers without GUI,
you can first install it before installing MMCV to skip the installation of `opencv-python`.
```

#### Build on Linux

1. Clone the repo

   ```bash
   git clone https://github.com/open-mmlab/mmcv.git
   cd mmcv
   ```

2. Install `ninja` and `psutil` to speed up the compilation

   ```bash
   pip install -r requirements/optional.txt
   ```

3. Check the nvcc version (requires 9.2+. Skip if no GPU available.)

   ```bash
   nvcc --version
   ```

   If the above command outputs the following message, it means that the nvcc setting is OK, otherwise you need to set CUDA_HOME.

   ```
   nvcc: NVIDIA (R) Cuda compiler driver
   Copyright (c) 2005-2020 NVIDIA Corporation
   Built on Mon_Nov_30_19:08:53_PST_2020
   Cuda compilation tools, release 11.2, V11.2.67
   Build cuda_11.2.r11.2/compiler.29373293_0
   ```

   :::{note}
   If you want to support ROCm, you can refer to [AMD ROCm](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html) to install ROCm.
   :::

4. Check the gcc version (requires 5.4+)

   ```bash
   gcc --version
   ```

5. Start building (takes 10+ min)

   ```bash
   MMCV_WITH_OPS=1 pip install -e . -v
   ```

6. Validate the installation

   ```bash
   python .dev_scripts/check_installation.py
   ```

   If no error is reported by the above command, the installation is successful. If there is an error reported, please check [Frequently Asked Questions](../faq.md) to see if there is already a solution.

   If no solution is found, please feel free to open an [issue](https://github.com/open-mmlab/mmcv/issues).

#### Build on macOS

```{note}
If you are using a mac with an M1 chip, install the nightly version of PyTorch, otherwise you will encounter the problem in [issues#2218](https://github.com/open-mmlab/mmcv/issues/2218).
```

1. Clone the repo

   ```bash
   git clone https://github.com/open-mmlab/mmcv.git
   cd mmcv
   ```

2. Install `ninja` and `psutil` to speed up the compilation

   ```bash
   pip install -r requirements/optional.txt
   ```

3. Start building

   ```bash
   MMCV_WITH_OPS=1 pip install -e .
   ```

4. Validate the installation

   ```bash
   python .dev_scripts/check_installation.py
   ```

   If no error is reported by the above command, the installation is successful. If there is an error reported, please check [Frequently Asked Questions](../faq.md) to see if there is already a solution.

   If no solution is found, please feel free to open an [issue](https://github.com/open-mmlab/mmcv/issues).

#### Build on Windows

Building MMCV on Windows is a bit more complicated than that on Linux.
The following instructions show how to get this accomplished.

##### Prerequisite

The following software is required for building MMCV on windows.
Install them first.

- [Git](https://git-scm.com/download/win)
  - During installation, tick **add git to Path**.
- [Visual Studio Community 2019](https://visualstudio.microsoft.com)
  - A compiler for C++ and CUDA codes.
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
  - Official distributions of Python should work too.
- [CUDA 10.2](https://developer.nvidia.com/cuda-10.2-download-archive)
  - Not required for building CPU version.
  - Customize the installation if necessary. As a recommendation, skip the driver installation if a newer version is already installed.

```{note}
You should know how to set up environment variables, especially `Path`, on Windows. The following instruction relies heavily on this skill.
```

##### Common steps

1. Launch Anaconda prompt from Windows Start menu

   Do not use raw `cmd.exe` s instruction is based on PowerShell syntax.

2. Create a new conda environment

   ```powershell
   (base) PS C:\Users\xxx> conda create --name mmcv python=3.7
   (base) PS C:\Users\xxx> conda activate mmcv  # make sure to activate environment before any operation
   ```

3. Install PyTorch. Choose a version based on your need.

   ```powershell
   # CUDA version
   (mmcv) PS C:\Users\xxx> conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
   # CPU version
   (mmcv) PS C:\Users\xxx> conda install install pytorch torchvision cpuonly -c pytorch
   ```

4. Clone the repo

   ```powershell
   (mmcv) PS C:\Users\xxx> git clone https://github.com/open-mmlab/mmcv.git
   (mmcv) PS C:\Users\xxx\mmcv> cd mmcv
   ```

5. Install `ninja` and `psutil` to speed up the compilation

   ```powershell
   (mmcv) PS C:\Users\xxx\mmcv> pip install -r requirements/optional.txt
   ```

6. Install required Python packages

   ```shell
   (mmcv) PS C:\Users\xxx\mmcv> pip install -r requirements/runtime.txt
   ```

7. Set up MSVC compiler

   Set Environment variable, add `C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\Hostx86\x64` to `PATH`, so that `cl.exe` will be available in prompt, as shown below.

   ```powershell
   (mmcv) PS C:\Users\xxx\mmcv> cl
   Microsoft (R) C/C++ Optimizing  Compiler Version 19.27.29111 for x64
   Copyright (C) Microsoft Corporation.   All rights reserved.

   usage: cl [ option... ] filename... [ / link linkoption... ]
   ```

   For compatibility, we use the x86-hosted and x64-targeted compiler. note `Hostx86\x64` in the path.

   You may want to change the system language to English because pytorch will parse text output from `cl.exe` to check its version. However only utf-8 is recognized. Navigate to Control Panel -> Region -> Administrative -> Language for Non-Unicode programs and change it to English.

##### Build and install MMCV

mmcv-full can be built in two ways:

1. Full version (CPU ops)

   Module `ops` will be compiled as a pytorch extension, but only x86 code will be compiled. The compiled ops can be executed on CPU only.

2. Full version (CUDA ops)

   Both x86 and CUDA codes of `ops` module will be compiled. The compiled version can be run on both CPU and CUDA-enabled GPU (if implemented).

###### CPU version

1. Set up environment variables

   ```powershell
   (mmcv) PS C:\Users\xxx\mmcv> $env:MMCV_WITH_OPS = 1
   ```

2. Build and install

   ```powershell
   (mmcv) PS C:\Users\xxx\mmcv> python setup.py build_ext
   (mmcv) PS C:\Users\xxx\mmcv> python setup.py develop
   ```

###### GPU version

2. Make sure `CUDA_PATH` or `CUDA_HOME` is already set in `envs` via `ls env:`, desired output is shown as below:

   ```powershell
   (mmcv) PS C:\Users\xxx\mmcv> ls env:

   Name                           Value
   ----                           -----
   CUDA_PATH                      C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2
   CUDA_PATH_V10_1                C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1
   CUDA_PATH_V10_2                C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2
   ```

   This should already be done by CUDA installer. If not, or you have multiple version of CUDA toolkit installed, set it with

   ```powershell
   (mmcv) PS C:\Users\xxx\mmcv> $env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2"
   # OR
   (mmcv) PS C:\Users\xxx\mmcv> $env:CUDA_HOME = $env:CUDA_PATH_V10_2 # if CUDA_PATH_V10_2 is in envs:
   ```

3. Set CUDA target arch

   ```shell
   # Here you need to change to the target architecture corresponding to your GPU
   (mmcv) PS C:\Users\xxx\mmcv> $env:TORCH_CUDA_ARCH_LIST="7.5"
   ```

   :::{note}
   Check your the compute capability of your GPU from [here](https://developer.nvidia.com/cuda-gpus).

   ```powershell
   (mmcv) PS C:\Users\xxx\mmcv> &"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\extras\demo_suite\deviceQuery.exe"
   Device 0: "NVIDIA GeForce GTX 1660 SUPER"
   CUDA Driver Version / Runtime Version          11.7 / 11.1
   CUDA Capability Major/Minor version number:    7.5
   ```

   The 7.5 above indicates the target architecture. Note: You need to replace v10.2 with your CUDA version in the above command.
   :::

4. Build and install

   ```powershell
   # build
   python setup.py build_ext # if success, cl will be launched to compile ops
   # install
   python setup.py develop
   ```

   ```{note}
   If you are compiling against PyTorch 1.6.0, you might meet some errors from PyTorch as described in [this issue](https://github.com/pytorch/pytorch/issues/42467). Follow [this pull request](https://github.com/pytorch/pytorch/pull/43380/files) to modify the source code in your local PyTorch installation.
   ```

##### Validate installation

```powershell
(mmcv) PS C:\Users\xxx\mmcv> python .dev_scripts/check_installation.py
```

If no error is reported by the above command, the installation is successful. If there is an error reported, please check [Frequently Asked Questions](../faq.md) to see if there is already a solution.
If no solution is found, please feel free to open an [issue](https://github.com/open-mmlab/mmcv/issues).

### Build mmcv

If you need to use PyTorch-related modules, make sure PyTorch has been successfully installed in your environment by referring to the [PyTorch official installation guide](https://github.com/pytorch/pytorch#installation).

1. Clone the repo

   ```bash
   git clone https://github.com/open-mmlab/mmcv.git
   cd mmcv
   ```

2. Start building

   ```bash
   pip install -e . -v
   ```

3. Validate installation

   ```bash
   python -c 'import mmcv;print(mmcv.__version__)'
   ```

### Build mmcv-full on IPU machine

Firstly, you need to apply for an IPU cloud machine, see [here](https://www.graphcore.ai/ipus-in-the-cloud).

#### Option 1: Docker

1. Pull docker

   ```bash
   docker pull graphcore/pytorch
   ```

2. Build MMCV under same python environment

#### Option 2: Install from SDK

1. Build MMCV

2. Use pip to install sdk according to [IPU PyTorch document](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/installation.html). Also, you need to apply for machine and sdk to Graphcore.

### Build mmcv-full on Ascend NPU machine

Before building mmcv-full, `torch_npu` should be installed. See the complete installation tutorial [PyTorch Installation Guide](https://gitee.com/ascend/pytorch/blob/master/docs/en/PyTorch%20Installation%20Guide/PyTorch%20Installation%20Guide.md)

#### Option 1: Install mmcv-full with pip

The Ascend compiled version of mmcv-full is already supported when the version of mmcv >= 1.7.0, we can pip install directly

```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/ascend/torch1.8.0/index.html
```

#### Option 2: Build mmcv-full NPU (Ascend) from Source

- Pull the source code

```bash
git pull https://github.com/open-mmlab/mmcv/tree/master
```

- Build

```bash
MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
```

- Install

```bash
MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
```

#### Test Case

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

### Build mmcv-full on Cambricon MLU Devices

#### Install torch_mlu

##### Option1: Install mmcv-full based on Cambricon docker image

Firstly, install and pull Cambricon docker image (please email service@cambricon.com for the latest release docker):

```bash
docker pull ${docker image}
```

Run and attach to the docker, [Install mmcv-full on MLU device](#install-mmcv\-full-on-cambricon-mlu-device) and [make sure you've installed mmcv-full on MLU device successfully](#test-code)

##### Option2: Install mmcv-full from compiling Cambricon PyTorch source code

Please email service@cambricon.com or contact with Cambricon engineers for a suitable version of CATCH package. After you get the suitable version of CATCH package, please follow the steps in ${CATCH-path}/CONTRIBUTING.md to install Cambricon PyTorch.

#### Install mmcv-full on Cambricon MLU device

Clone the repo

```bash
git clone https://github.com/open-mmlab/mmcv.git
```

The mlu-ops library will be downloaded to the default directory (mmcv/mlu-ops) while building MMCV. You can also set `MMCV_MLU_OPS_PATH` to an existing mlu-ops library before building as follows:

```bash
export MMCV_MLU_OPS_PATH=/xxx/xxx/mlu-ops
```

Install mmcv-full

```bash
cd mmcv
export MMCV_WITH_OPS=1
export FORCE_MLU=1
python setup.py install
```

#### Test Code

After finishing previous steps, you can run the following python code to make sure that you've installed mmcv-full on MLU device successfully

```python
import torch
import torch_mlu
from mmcv.ops import sigmoid_focal_loss
x = torch.randn(3, 10).mlu()
x.requires_grad = True
y = torch.tensor([1, 5, 3]).mlu()
w = torch.ones(10).float().mlu()
output = sigmoid_focal_loss(x, y, 2.0, 0.25, w, 'none')
print(output)
```
