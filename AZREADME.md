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

For a CUDA build the machine needs the CUDA toolkit (`nvcc`); add `FORCE_CUDA=1` if no GPU is
visible at build time. Output lands in `dist/` as `mmcv-2.2.0-cp314-cp314-linux_x86_64.whl`.

Check it:

```bash
python .dev_scripts/check_installation.py
```

## Publish

```bash
twine upload --repository-url https://pip.azmed.co:8870 \
  -u "dev-team@azmed.co" -p "$AZPYPI_PWD" dist/*
```
