# Impl/libtorch

## 简介

Impl/libtorch 模块使用 [PyTorch C++ API](https://pytorch.org/cppdocs/) 实现了一致性测试套件中的算子声明，大部分 API 来自于 ATen 提供的算子库，部分算子参考了 Pytorch 内部和 [vision](https://github.com/pytorch/vision) 中的实现。

## 开始
使用之前请确保 CUDA 和 PyTorch 已经成功安装在环境中，当前适配版本 `pytorch == 1.10`

### i. 编译
因为使用到了 PyTorch C++ API，因此这里使用当前环境中的 libtorch 库调用 ATen
```bash
cmake .. -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` -DIMPL_CUDA=OFF -DIMPL_OPT=TORCH

make -j4
```
或在测试套件根目录运行脚本 `sh scripts/build_impl.sh torch` 编译

### ii. 运行与测试
```python3
python main.py --mode gen_data --fname all

python main.py --mode run_test --fname all
```

## 功能介绍

### i. buildATen
> at::Tensor buildATen(*diopiTensorHandle_t tensor*);

用于 `diopiTensor -> at::Tensor` 的构造，函数调用了 `at::Storage(...)` 使用外部**已申请的内存空间**构造 `at::Tensor`

### ii. updateATen2Tensor
> void updateATen2Tensor(*diopiContextHandle_t ctx, const at::Tensor& atOut, diopiTensorHandle_t out*);

一个 `at::func` 会返回一个(组) `at::Tensor` 作为算子的输出结果，`diopiTensor` 没有办法转化为 `at::Tensor` 直接参与计算得到结果，需要将输出的 at::Tensor 内存拷贝回 diopiTensor 中,
因此每个 `at::func` 调用后都需要调用 `updateATen2Tensor` 将结果拷回。

当 `at::func` 得到一组 `std::tuple<at::Tensor, at::Tensor, ...>` 或 `std::vector<at::Tensor>`，函数提供了相应的重载可以调用。

### iii. buildDiopiTensor
> void buildDiopiTensor(*diopiContextHandle_t ctx, at::Tensor& input, diopiTensorHandle_t\* out*)

用于 `at::Tensor -> diopiTensor` 的构造，部分算子（比如 [`nonzero`](https://pytorch.org/docs/stable/generated/torch.nonzero.html)）在算子运行时才可以获得到结果张量的形状大小。

这里提供了根据 `ATen` 构造 `diopiTensor` 的方法，函数传入一个指向 `nullptr` 的 `diopiTensorHandle_t`，函数内部按照输出 `at::Tensor` 的形状和内存构造 `diopiTensor`。

