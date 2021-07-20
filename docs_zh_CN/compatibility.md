## MMCV 兼容性说明

### MMCV v1.3.10

为了灵活地支持更多的后端和硬件，例如 `NVIDIA GPUs` 、`AMD GPUs`，我们重构了 `mmcv/ops/csrc` 目录。注意，这次重构不会影响 API 的使用。更多相关信息，请参考 [PR1206](https://github.com/open-mmlab/mmcv/pull/1206)。
