# Install MMCV on Windows

Setting up MMCV on Windows is a bit more complicated than that on Linux.
This instruction shows how to get this accomplished on Windows.

## Prerequisite

The following software is required for setting up MMCV on windows.
Install them first.

- [Git](https://git-scm.com/download/win)
  - During installation, tick **add git to PATH**.
- [Visual Studio Community 2019](https://visualstudio.microsoft.com)
  - A compiler for C++ and CUDA codes.
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
  - Official distributions of Python should work too.
- [CUDA 10.2](https://developer.nvidia.com/cuda-10.2-download-archive)
  - Not required if only CPU version is needed.
  - Customize installation if needed. As a recommendation, cancel driver installation if a later version is already installed.

**You should know how to set up environment variable, especially `PATH`, on Windows. The following instruction rely heavily on this skill.**

## Setup Python Environment

- Launch Anaconda prompt from Windows Start menu
  - Do not use raw `cmd.exe` or `powershell.exe` as suggested by Miniconda installer
  - There are two Anaconda prompt available, one is based on PowerShell, the other is based on the traditional `cmd.exe`. This instruction is based on the PowerShell one.
- Create a new conda environment
  ```powershell
  conda create --name mmcv python=3.7  # 3.6, 3.7, 3.8 should work too as tested
  conda activate mmcv   # make sure to activate environment before any operation
  ```
- Install Pytorch, with or without CUDA support based on your need
  ```powershell
  # CUDA version
  conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
  # CPU version
  conda install pytorch torchvision cpuonly -c pytorch
  ```
- Prepare source code
  ```powershell
  git clone https://github.com/open-mmlab/mmcv.git
  git checkout v1.2.0  # based on target version
  cd mmcv
  ```
- Install required Python packages
  ```powershell
  pip3 install -r requirements.txt
  ```
  - Or following other ways (e.g. `conda install`) to install all requirements.

## Build and install MMCV

MMCV can be built three ways:

- Basic setup without OPs
  In this way, only `flow_warp` module will be compiled as a Cython extension.
- Full setup with OPs on CPU
  In addition to `flow_warp` module, module `ops` will be compiled as a torch extension, but only x86_64 code will be compiled. The compiled version can be run on CPU only.
- Full setup with OPs on CUDA
  Both x86 and CUDA code of `ops` module will be compiled. The compiled version can be run on a CUDA enabled GPU.

### Common steps

- Set up MSVC compiler

  - Set Environment variable, add `C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\Hostx86\x64` to `PATH`, so that `cl.exe` is available in prompt, as shown below.

  ```plain
  (base) PS C:\Users\WRH> cl
  Microsoft (R) C/C++ Optimizing Compiler Version 19.27.29111 for x64
  Copyright (C) Microsoft Corporation.  All rights reserved.

  usage: cl [ option... ] filename... [ /link linkoption... ]
  ```

  For compatibility, we use the _x86-hosted and x64-targeted_ version, note `Hostx86\x64` in the path.

  - You may want to change the system language to English because pytorch will parse the output of `cl.exe` to check its version. However only utf-8 is recognized. Do it from Control Panel -> Region -> Administration -> Non-Unicode .

### Basic setup without OPS

Launch Anaconda shell from Start menu and issue the following commands:

```powershell
# activate environment
conda activate mmcv
# change directory
cd mmcv
# build
$env:MMCV_WITH_OPS = 0
python setup.py build_ext # if success, cl will be launched to compile flow_warp.
# install
python setup.py develop
# check
pip list
```

### Full setup with OPs (CPU)

- Set up environment variables
  ```powershell
  $env:MMCV_WITH_OPS = 1
  $env:MAX_JOBS = 8  # based on your available number of CPU cores and amount of memory
  ```
- Following build steps as above
  ```powershell
  # activate environment
  conda activate mmcv
  # change directory
  cd mmcv
  # build
  python setup.py build_ext
  # install
  python setup.py develop
  # check
  pip list
  ```

### Full setup with OPs (CUDA)

- Set up environment variables
  ```powershell
  $env:MMCV_WITH_OPS = 1
  $env:MAX_JOBS = 8  # based on available number of CPU cores and amount of memory
  ```
- Check `CUDA_PATH` or `CUDA_HOME` is already set in `envs` (should be done by CUDA installer), otherwise set it with

  ```powershell
  $env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\"
  # OR
  $env:CUDA_HOME = $env:CUDA_PATH_V10_2 # if CUDA_PATH_V10_2 is already set during CUDA installation
  ```

- Set CUDA target arch
  ```powershell
  # Suppose using GTX 1080
  $env:TORCH_CUDA_ARCH_LIST="6.1"
  # OR build for all suppoted arch, will be slow
  $env:TORCH_CUDA_ARCH_LIST="3.5 3.7 5.0 5.2 6.0 6.1 7.0 7.5"
  ```
- Launch compiling the same way as CPU
  ```powershell
  $env:MMCV_WITH_OPS = 1
  $env:MAX_JOBS = 8  # based on available number of CPU cores and amount of memory
  # activate environment
  conda activate mmcv
  # change directory
  cd mmcv
  # build
  python setup.py build_ext
  # install
  python setup.py develop
  # check
  pip list
  ```
