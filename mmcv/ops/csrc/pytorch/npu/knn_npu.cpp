#include "pytorch_npu_helper.hpp"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

using namespace NPU_NAME_SPACE;
using namespace std;

void knn_forward_npu(int b, int n, int m, int nsample, const Tensor xyz,
                     const Tensor new_xyz, Tensor idx, Tensor dist2) {
  // transpose known from [B, N, 3] to [B, 3, N]
  at::Tensor source = xyz.transpose(2, 1).contiguous();
  at::Tensor target = new_xyz.contiguous();

  at::Tensor dist =
      at::zeros({target.sizes()[0], target.sizes()[1], source.sizes()[2]},
                target.options());
  bool is_from_knn = true;
  EXEC_NPU_CMD_SYNC(aclnnKnn, source, target, is_from_knn, dist);

  idx = idx.to(at::kLong);
  int64_t dim = 2;
  bool largest = false;
  bool sorted = true;
  EXEC_NPU_CMD_SYNC(aclnnTopk, dist, nsample, dim, largest, sorted, dist2, idx);
  idx = idx.to(at::kInt);
}

void knn_forward_impl(int b, int n, int m, int nsample, const Tensor xyz,
                      const Tensor new_xyz, Tensor idx, Tensor dist2);

REGISTER_NPU_IMPL(knn_forward_impl, knn_forward_npu);
