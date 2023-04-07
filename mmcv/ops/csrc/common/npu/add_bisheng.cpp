// 导入毕昇C++依赖头文件
#include <acl/acl.h>
#include <sycl/sycl.hpp>
// 导入pytorch头文件
#include <torch/extension.h>
// 导入昇腾pytorch头文件
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>

using namespace at;
using namespace sycl;
using namespace sycl::access;

// 算子kernel实现文件
template <typename T, int group_count, typename kernel_name>
void bscpp_add_kernel(queue &sycl_queue, const T *in1, const T *in2, T *out, size_t length) {
  sycl_queue.submit([&](handler &cgh) {
      cgh.parallel_for_work_group<kernel_name>(sycl::range<1> { group_count }, [=](group<1> grp) {
          const auto gid = grp.get_id(0);
          const auto idx_base = gid * length;
          [[loop::parallel]] for (size_t i = 0; i < length; ++i) {
              out[i + idx_base] = in1[i + idx_base] + in2[i + idx_base];
          }
      });
  });
  sycl_queue.wait();
}

// 调用算子的辅助函数
void bscpp_add_launch(const at::Tensor &self, const at::Tensor &other, at::Tensor &result) {
  const float *self_ptr = static_cast<float *>(self.storage().data_ptr().get());
  const float *other_ptr = static_cast<float *>(other.storage().data_ptr().get());
  float *result_ptr = static_cast<float *>(result.storage().data_ptr().get());

  // 根据PyTorch当前stream和context生成毕昇C++任务管理队列queue
  aclrtContext acl_context;
  aclrtGetCurrentContext(&acl_context);
  sycl::context sycl_context = sycl::make_context<sycl::backend::cce>(acl_context);
  int device_id;
  aclrtGetDevice(&device_id);
  auto npu_stream = c10_npu::getCurrentNPUStream(device_id);
  auto acl_stream = npu_stream.stream();
  queue sycl_queue = sycl::make_queue<sycl::backend::cce>(acl_stream, sycl_context);

  constexpr size_t group_count = 1;
  size_t len = result.numel();
  bscpp_add_kernel<float, group_count, class bscpp_add_launch_float>(sycl_queue, self_ptr, other_ptr, result_ptr, len);
}

// pytorch扩展算子接口
at::Tensor bscpp_add(const at::Tensor &self, const at::Tensor &other) {
  static auto warn_once = [](){ 
      std::cout << "Warning: kernel [bscpp_add] is written by BiShengCPP." << std::endl;
      return true;
  }();

  at::Tensor result = at::empty(self.sizes(), self.options());
  bscpp_add_launch(self, other, result);

  return result;
}


// 将c++接口绑定至python侧
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("bscpp_add", &bscpp_add, "add two float tensors by BiShengCPP.");
// }