#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void chamfer_distance_forward_npu(Tensor XYZ1, Tensor XYZ2, Tensor dist1,
                                  Tensor dist2, Tensor idx1, Tensor idx2) {
  bool is_half = XYZ1.scalar_type() == at::kHalf;
  at::Tensor xyz1 = at::ones_like(XYZ1);
  at::Tensor xyz2 = at::ones_like(XYZ2);
  at::Tensor distf1 = at::ones_like(dist1);
  at::Tensor distf2 = at::ones_like(dist2);
  xyz1 = XYZ1.transpose(1, 2).transpose(0, 1);
  xyz2 = XYZ2.transpose(1, 2).transpose(0, 1);
  if (is_half) {
    xyz1 = xyz1.to(at::kFloat);
    xyz2 = xyz2.to(at::kFloat);
    distf1 = dist1.to(at::kFloat);
    distf2 = dist2.to(at::kFloat);
  }
  OpCommand cmd;
  cmd.Name("ChamferDistance")
      .Input(xyz1)
      .Input(xyz2)
      .Output(distf1)
      .Output(distf2)
      .Output(idx1)
      .Output(idx2)
      .Run();
  if (is_half) {
    distf1 = distf1.to(at::kHalf);
    distf2 = distf2.to(at::kHalf);
  }
  dist1.copy_(distf1);
  dist2.copy_(distf2);
}

void chamfer_distance_backward_npu(Tensor xyz1, Tensor xyz2, Tensor idx1,
                                   Tensor idx2, Tensor grad_dist1,
                                   Tensor grad_dist2, Tensor grad_xyz1,
                                   Tensor grad_xyz2) {
  EXEC_NPU_CMD(aclnnChamferDistanceBackward, xyz1, xyz2, idx1, idx2, grad_dist1,
               grad_dist2, grad_xyz1, grad_xyz2);
}

void chamfer_distance_forward_impl(Tensor XYZ1, Tensor XYZ2, Tensor dist1,
                                   Tensor dist2, Tensor idx1, Tensor idx2);
REGISTER_NPU_IMPL(chamfer_distance_forward_impl, chamfer_distance_forward_npu);

void chamfer_distance_backward_impl(Tensor xyz1, Tensor xyz2, Tensor idx1,
                                    Tensor idx2, Tensor grad_dist1,
                                    Tensor grad_dist2, Tensor grad_xyz1,
                                    Tensor grad_xyz2);
REGISTER_NPU_IMPL(chamfer_distance_backward_impl,
                  chamfer_distance_backward_npu);
