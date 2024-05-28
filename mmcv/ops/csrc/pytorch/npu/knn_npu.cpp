#include "pytorch_npu_helper.hpp"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

using namespace NPU_NAME_SPACE;
using namespace std;

void knn_forward_npu(int b, int n, int m, int nsample,
                     const Tensor xyz, const Tensor new_xyz, Tensor idx, Tensor dist2) {
  at::Tensor xyz_npu = xyz.contiguous();
  at::Tensor center_xyz_npu = new_xyz.contiguous();
  bool is_from_knn = true;
  EXEC_NPU_CMD(aclnnKnn, xyz_npu, center_xyz_npu, nsample, is_from_knn, idx, dist2);
}

void knn_forward_impl(int b, int n, int m, int nsample,
                      const Tensor xyz, const Tensor new_xyz, Tensor idx, Tensor dist2);

REGISTER_NPU_IMPL(knn_forward_impl, knn_forward_npu);
