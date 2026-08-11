# AZ build notes

Fork of mmcv 2.2.0, patched to build on Python 3.13+. Target: **Linux x86_64, Python 3.14**.

The wheel is tied to the torch version and CUDA variant it was compiled against — build one per
combination and note it in the version/filename if you publish more than one.

## Build

torch must be installed first (`setup.py` imports it), so build without isolation.

```bash
# CPU
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
# or GPU — CUDA 12.6 minimum, there are no cp314 torch wheels for cu121/cu124
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126

pip install build wheel setuptools packaging ninja psutil
MMCV_WITH_OPS=1 python -m build --wheel --no-isolation
```

A CUDA build needs the CUDA toolkit (`nvcc`) but **not** a GPU — set `FORCE_CUDA=1`, plus
`TORCH_CUDA_ARCH_LIST` for the arch you deploy on (e.g. `8.9`), otherwise torch compiles for every
arch it supports:

```bash
MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.9" python -m build --wheel --no-isolation
```

Output lands in `dist/` as `mmcv-2.2.0-cp314-cp314-linux_x86_64.whl`.

Check it:

```bash
python .dev_scripts/check_installation.py
```

## CI

`.github/workflows/package_mmcv_cp314_x86_64.yml` does all of the above for torch 2.13.0 + cu126 and
publishes the result. Run it from the Actions tab; it needs the `AZPYPI_PASSWORD` secret.

## Publish (manual)

```bash
twine upload --repository-url https://pip.azmed.co:8870 \
  -u "dev-team@azmed.co" -p "$AZPYPI_PWD" dist/*
```
