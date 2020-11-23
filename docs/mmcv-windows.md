# 0. Prepare

By default, use win64 distribution for all softwares.

- [Install Git](https://git-scm.com/download/win)
- [Install Visual Studio Community 2019](https://visualstudio.microsoft.com)
- [Install Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Install CUDA 10.2](https://developer.nvidia.com/cuda-10.2-download-archive)
  - Should be after VS for VS integration
  - Customize Driver installation if needed.
- Do some proxy setting if necessary
  - [conda](https://conda.io/projects/conda/en/latest/user-guide/configuration/use-condarc.html#configure-conda-for-use-behind-a-proxy-server-proxy-servers)

# 1. Setup Python Environment

- Launch Anaconda shell from start menu
  - Do not use raw `cmd.exe` as suggested by miniconda installer.
- `conda create --name innolab python=3.7`
- `conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`
- `conda install pytorch torchvision cpuonly -c pytorch`
- `conda activate mmcv`

# 1. Set up MMCV

## Basic setup without OPS

- Set up compiler
  - Set Environment variable, so that PATH contains `cl.exe`.
  ```powershell
  $env:PATH = "%path%;%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\Hostx64\x64"
  $env:MAX_JOBS = 8
  ls env:  #
  ```
  - You may want to change system language to English because pytorch will parse the output of `cl.exe` to check its version, however only utf-8 is recognized, do it from Control Panel -> Region -> Administration -> Non-Unicode .
- Launch Anaconda shell from start menu
  - `echo %path%` to check
- `conda activate mmcv`
- `git clone https://github.com/open-mmlab/mmcv.git`
- `git checkout ...` ( based on target version )
- `cd` to `mmcv` directory
- `pip3 install -r requirements.txt`
  - Or other ways (e.g. `conda install`) to install all requirements.
- `python setup.py build_ext`
  - Slow for the first tile, maybe setting up ninja ?
- `python setup.py develop`
- `pip list` to check mmcv is installed

## Full setup with OPS (CPU)

- PowerShell
  ```powershell
  $env:MMCV_WITH_OPS = 1
  $env:MAX_JOBS = 10
  ```
- `python setup.py build_ext`
  - Slow for the first tile, maybe setting up ninja ?

## Full setup with OPS (CUDA)

- Check `CUDA_PATH` or `CUDA_HOME` is already set in envs (should be done by CUDA installer), otherwise set with
  ```powershell
  $env:USE_NINJA=0
  $env:TORCH_CUDA_ARCH_LIST="3.5+PTX 3.7 5.0 5.2+PTX 6.0 6.1+PTX 7.0+PTX 7.5+PTX"
  $env:TORCH_CUDA_ARCH_LIST="3.5 3.7 5.0 5.2 6.0 6.1 7.0 7.5"
  $env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\"
  ```
