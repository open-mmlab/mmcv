## Build MMCV from source

### Build on Linux or macOS

After cloning the repo with

```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
```

You can either

- install the lite version

  ```bash
  pip install -e .
  ```

- install the full version

  ```bash
  MMCV_WITH_OPS=1 pip install -e .
  ```

If you are on macOS, add the following environment variables before the installing command.

```bash
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++'
```

e.g.,

```bash
CC=clang CXX=clang++ CFLAGS='-stdlib=libc++' MMCV_WITH_OPS=1 pip install -e .
```

Note: If you would like to use `opencv-python-headless` instead of `opencv-python`,
e.g., in a minimum container environment or servers without GUI,
you can first install it before installing MMCV to skip the installation of `opencv-python`.

### Build on Windows

Building MMCV on Windows is a bit more complicated than that on Linux.
The following instructions show how to get this accomplished.

#### Prerequisite

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

**You should know how to set up environment variables, especially `Path`, on Windows. The following instruction relies heavily on this skill.**

#### Setup Python Environment

1. Launch Anaconda prompt from Windows Start menu

    Do not use raw `cmd.exe` s instruction is based on PowerShell syntax.

1. Create a new conda environment

    ```shell
    conda create --name mmcv python=3.7  # 3.6, 3.7, 3.8 should work too as tested
    conda activate mmcv  # make sure to activate environment before any operation
    ```

1. Install PyTorch. Choose a version based on your need.

    ```shell
    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
    ```

    We only tested PyTorch version >= 1.6.0.

1. Prepare MMCV source code

    ```shell
    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv
    ```

1. Install required Python packages

    ```shell
    pip3 install -r requirements.txt
    ```

#### Build and install MMCV

MMCV can be built in three ways:

1. Lite version (without ops)

   In this way, no custom ops are compiled and mmcv is a pure python package.

1. Full version (CPU ops)

   Module `ops` will be compiled as a pytorch extension, but only x86 code will be compiled. The compiled ops can be executed on CPU only.

1. Full version (CUDA ops)

   Both x86 and CUDA codes of `ops` module will be compiled. The compiled version can be run on both CPU and CUDA-enabled GPU (if implemented).

##### Common steps

1. Set up MSVC compiler

    Set Environment variable, add `C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\Hostx86\x64` to `PATH`, so that `cl.exe` will be available in prompt, as shown below.

    ```none
    (base) PS C:\Users\xxx> cl
    Microsoft (R) C/C++ Optimizing  Compiler Version 19.27.29111 for x64
    Copyright (C) Microsoft Corporation.   All rights reserved.

    usage: cl [ option... ] filename... [ / link linkoption... ]
    ```

    For compatibility, we use the x86-hosted and x64-targeted compiler. note `Hostx86\x64` in the path.

    You may want to change the system language to English because pytorch will parse text output from `cl.exe` to check its version. However only utf-8 is recognized. Navigate to Control Panel -> Region -> Administrative -> Language for Non-Unicode programs and change it to English.

##### Option 1: Build MMCV (lite version)

After finishing above common steps, launch Anaconda shell from Start menu and issue the following commands:

```shell
# activate environment
conda activate mmcv
# change directory
cd mmcv
# install
python setup.py develop
# check
pip list
```

##### Option 2: Build MMCV (full version with CPU)

1. Finish above common steps
1. Set up environment variables

    ```shell
    $env:MMCV_WITH_OPS = 1
    $env:MAX_JOBS = 8  # based on your available number of CPU cores and amount of memory
    ```

1. Following build steps of the lite version

    ```shell
    # activate environment
    conda activate mmcv
    # change directory
    cd mmcv
    # build
    python setup.py build_ext # if success, cl will be launched to compile ops
    # install
    python setup.py develop
    # check
    pip list
    ```

##### Option 3: Build MMCV (full version with CUDA)

1. Finish above common steps
1. Make sure `CUDA_PATH` or `CUDA_HOME` is already set in `envs` via `ls env:`, desired output is shown as below:

   ```none
   (base) PS C:\Users\WRH> ls env:

   Name                           Value
   ----                           -----
   <... omit some lines ...>
   CUDA_PATH                      C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2
   CUDA_PATH_V10_1                C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1
   CUDA_PATH_V10_2                C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2
   <... omit some lines ...>
   ```

   This should already be done by CUDA installer. If not, or you have multiple version of CUDA tookit installed, set it with

   ```shell
   $env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2"
   # OR
   $env:CUDA_HOME = $env:CUDA_PATH_V10_2 # if CUDA_PATH_V10_2 is in envs:
   ```

1. Set CUDA target arch

   ```shell
   # Suppose you are using GTX 1080, which is of capability 6.1
   $env:TORCH_CUDA_ARCH_LIST="6.1"
   # OR build all suppoted arch, will be slow
   $env:TORCH_CUDA_ARCH_LIST="3.5 3.7 5.0 5.2 6.0 6.1 7.0 7.5"
   ```

   Note: Check your the compute capability of your GPU from [here](https://developer.nvidia.com/cuda-gpus).

1. Launch compiling the same way as CPU

   ```shell
   $env:MMCV_WITH_OPS = 1
   $env:MAX_JOBS = 8  # based on available number of CPU cores and amount of memory
   # activate environment
   conda activate mmcv
   # change directory
   cd mmcv
   # build
   python setup.py build_ext # if success, cl will be launched to compile ops
   # install
   python setup.py develop
   # check
   pip list
   ```

   **Note**: If you are compiling against PyTorch 1.6.0, you might meet some errors from PyTorch as described in [this issue](https://github.com/pytorch/pytorch/issues/42467).
   Follow [this pull request](https://github.com/pytorch/pytorch/pull/43380/files) to modify the source code in your local PyTorch installation.

If you meet issues when running or compiling mmcv, we list some common issues in [TROUBLESHOOTING](./trouble_shooting.html).
