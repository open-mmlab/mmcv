# AZ build notes

Fork of mmcv 2.2.0, patched to build on Python 3.13+. Target: **Linux x86_64, Python 3.14**,
torch 2.13.0, CUDA 12.6.

## Build

The wheel is compiled inside a container, so the host only needs Docker:

```bash
docker compose -f docker/builder.yml up -d --build
docker compose -f docker/builder.yml cp builder:/io/dist/ dist/
docker compose -f docker/builder.yml down --remove-orphans
```

`docker/build.Dockerfile` holds the torch version, the CUDA base image and
`TORCH_CUDA_ARCH_LIST` (`8.9` — it must cover the GPUs we deploy on). It also runs
`.dev_scripts/check_installation.py` on the built wheel, CPU ops only, as there is no GPU during an
image build.

## CI

`.github/workflows/package_mmcv_cp314_x86_64.yml` runs the above and publishes the result. Trigger
it from the Actions tab; it needs the `AZPYPI_PASSWORD` secret.

## Publish

```bash
twine upload --repository-url https://pip.azmed.co:8870 \
  -u "dev-team@azmed.co" -p "$AZPYPI_PWD" dist/*
```
