#!/bin/sh
set -e

uv build
uvx auditwheel repair dist/*.whl -w dist/wheelhouse \
    -z 9 \
    --exclude libc10.so \
    --exclude libc10_cuda.so \
    --exclude libtorch.so \
    --exclude libtorch_cpu.so \
    --exclude libtorch_cuda.so \
    --exclude libtorch_python.so \
    --exclude libshm.so

cp dist/wheelhouse/*.whl /out
cp dist/*.tar.gz /out
