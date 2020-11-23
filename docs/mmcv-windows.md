# Build MMCV | MMCV-full on Windows

Building MMCV on Windows is a bit more complicated than that on Linux.
The following instructions show how to get this accomplished.

## Prerequisite

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
  - Customize installation if necessary. As a recommendation, cancel driver installation if a later version is already installed.

**You should know how to set up environment variables, especially `Path`, on Windows. The following instruction relies heavily on this skill.**

## Setup Python Environment

1.  Launch Anaconda prompt from Windows Start menu

    - Do not use raw `cmd.exe` s instruction is based on PowerShell syntax.

1.  Create a new conda environment

    ```powershell
    conda create --name mmcv python=3.7  # 3.6, 3.7, 3.8 should work too as tested
    conda activate mmcv  # make sure to activate environment before any operation
    ```

1.  Install Pytorch. Choose a version based on your need.

    ```powershell
    # latest CUDA version
    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
    # latest CPU version
    conda install pytorch torchvision cpuonly -c pytorch
    ```

    - We didn't make CUDA codes successfully built for Pytorch version less than 1.6.0. For version 1.6.0, minor modification of Pytorch's source code is required.

1.  Prepare MMCV source code

    ```powershell
    git clone https://github.com/open-mmlab/mmcv.git
    git checkout v1.2.0  # based on your target version
    cd mmcv
    ```

1.  Install required Python packages

    ```powershell
    pip3 install -r requirements.txt
    ```

    - Or `conda install` all requirements.

## Build and install MMCV

MMCV can be built in three ways:

1. Lite version without ops

   In this way, only `flow_warp` module will be compiled as a Cython extension.

1. Full version with ops, but only built for CPU

   In addition to `flow_warp` module, module `ops` will be compiled as a pytorch extension, but only x86 code will be compiled. The compiled ops can be executed on CPU only.

1. Full version with ops, built for both CPU and GPU

   Both x86 and CUDA codes of `ops` module will be compiled. The compiled version can be run on both CPU and CUDA-enabled GPU (if implemented).

### Common steps

1. Set up MSVC compiler

   - Set Environment variable, add `C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\Hostx86\x64` to `PATH`, so that `cl.exe` will be available in prompt, as shown below.

   ```plain
   (base) PS C:\Users\WRH> cl
   Microsoft (R) C/C++ Optimizing Compiler Version 19.27.29111 for x64
   Copyright (C) Microsoft Corporation.  All rights reserved.

   usage: cl [ option... ] filename... [ /link linkoption... ]
   ```

   - For compatibility, we use the x86-hosted and x64-targeted compiler. note `Hostx86\x64` in the path.
   - You may want to change the system language to English because pytorch will parse text output from `cl.exe` to check its version. However only utf-8 is recognized. Navigate to Control Panel -> Region -> Administrative -> Language for Non-Unicode programs and change it to English.

### Build MMCV lite version

After finishing above common steps, launch Anaconda shell from Start menu and issue the following commands:

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

### Build MMCV full version with CPU

1.  Finish above common steps
1.  Set up environment variables

    ```powershell
    $env:MMCV_WITH_OPS = 1
    $env:MAX_JOBS = 8  # based on your available number of CPU cores and amount of memory
    ```

1.  Following build steps of the lite version

    ```powershell
    # activate environment
    conda activate mmcv
    # change directory
    cd mmcv
    # build
    python setup.py build_ext # if success, cl will be launched to compile flow_warp first and then ops
    # install
    python setup.py develop
    # check
    pip list
    ```

### Build MMCV full version with CUDA

1. Finish above common steps
1. Make sure `CUDA_PATH` or `CUDA_HOME` is already set in `envs` via `ls env:`, desired output is shown as below:

   ```plain
   (base) PS C:\Users\WRH> ls env:

   Name                           Value
   ----                           -----
   <... omit some lines ...>
   <... omit some lines ...>
   ```

   This should already be done by CUDA installer. If not, or you have multiple version of CUDA tookit installed, set it with

   ```powershell
   $env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\"
   # OR
   $env:CUDA_HOME = $env:CUDA_PATH_V10_2 # if CUDA_PATH_V10_2 is in envs:
   ```

1. Set CUDA target arch

   ```powershell
   # Suppose you are using GTX 1080, which is of capability 6.1
   $env:TORCH_CUDA_ARCH_LIST="6.1"
   # OR build all suppoted arch, will be slow
   $env:TORCH_CUDA_ARCH_LIST="3.5 3.7 5.0 5.2 6.0 6.1 7.0 7.5"
   ```

   - Check your the compute capability of your GPU from [here](https://developer.nvidia.com/cuda-gpus).

1. Launch compiling the same way as CPU

   ```powershell
   $env:MMCV_WITH_OPS = 1
   $env:MAX_JOBS = 8  # based on available number of CPU cores and amount of memory
   # activate environment
   conda activate mmcv
   # change directory
   cd mmcv
   # build
   python setup.py build_ext # if success, cl will be launched to compile flow_warp first and then ops
   # install
   python setup.py develop
   # check
   pip list
   ```
