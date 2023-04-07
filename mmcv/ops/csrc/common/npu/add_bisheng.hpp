// 导入毕昇C++依赖头文件
#include <acl/acl.h>
#include <sycl/sycl.hpp>
// 导入pytorch头文件
#include <torch/extension.h>
// 导入昇腾pytorch头文件
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>

// 调用算子的辅助函数
void bscpp_add_launch(const at::Tensor &self, const at::Tensor &other, at::Tensor &result);
// pytorch扩展算子接口
at::Tensor bscpp_add(const at::Tensor &self, const at::Tensor &other);
