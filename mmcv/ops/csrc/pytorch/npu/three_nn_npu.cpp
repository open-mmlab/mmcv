#include "pytorch_npu_helper.hpp"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

using namespace NPU_NAME_SPACE;
using namespace std;

void three_nn_forward_npu(int b, int n, int m, const Tensor unknown,
                          const Tensor known, Tensor dist2, Tensor idx) {
  // transpose known [B, N, 3] -> [B, 3, N]
  at::Tensor source = known.transpose(2, 1).contiguous();
  at::Tensor target = unknown.contiguous();
  auto originDtype = source.scalar_type();
  if (originDtype == at::kHalf) {
    source = source.to(at::kFloat);
    target = target.to(at::kFloat);
    dist2 = dist2.to(at::kFloat);
  }

  at::Tensor dist =
      at::zeros({target.sizes()[0], target.sizes()[1], source.sizes()[2]},
                target.options().dtype(at::kFloat));
  bool is_from_knn = false;
  EXEC_NPU_CMD_SYNC(aclnnKnn, source, target, is_from_knn, dist);

  idx = idx.to(at::kLong);
  int64_t dim = 2;
  bool largest = false;
  bool sorted = true;
  uint32_t nsample = 3;
  EXEC_NPU_CMD_SYNC(aclnnTopk, dist, nsample, dim, largest, sorted, dist2, idx);
  idx = idx.to(at::kInt);

  if (originDtype == at::kHalf) {
    dist2 = dist2.to(at::kHalf);
  }
}

void three_nn_forward_impl(int b, int n, int m, const Tensor unknown,
                           const Tensor known, Tensor dist2, Tensor idx);

REGISTER_NPU_IMPL(three_nn_forward_impl, three_nn_forward_npu);
