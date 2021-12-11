#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void PointsInBoxesPartForwardCUDAKernelLauncher(int batch_size, int boxes_num,
                                                int pts_num, const Tensor boxes,
                                                const Tensor pts,
                                                Tensor box_idx_of_points);

void points_in_boxes_part_forward_cuda(int batch_size, int boxes_num,
                                       int pts_num, const Tensor boxes,
                                       const Tensor pts,
                                       Tensor box_idx_of_points) {
  PointsInBoxesPartForwardCUDAKernelLauncher(batch_size, boxes_num, pts_num,
                                             boxes, pts, box_idx_of_points);
};

void PointsInBoxesAllForwardCUDAKernelLauncher(int batch_size, int boxes_num,
                                               int pts_num, const Tensor boxes,
                                               const Tensor pts,
                                               Tensor box_idx_of_points);

void points_in_boxes_all_forward_cuda(int batch_size, int boxes_num,
                                      int pts_num, const Tensor boxes,
                                      const Tensor pts,
                                      Tensor box_idx_of_points) {
  PointsInBoxesAllForwardCUDAKernelLauncher(batch_size, boxes_num, pts_num,
                                            boxes, pts, box_idx_of_points);
};
#endif

void points_in_boxes_part_forward(Tensor boxes_tensor, Tensor pts_tensor,
                                  Tensor box_idx_of_points_tensor) {
  // params boxes: (B, N, 7) [x, y, z, x_size, y_size, z_size, rz] in LiDAR
  // coordinate, z is the bottom center, each box params pts: (B, npoints, 3)
  // [x, y, z] in LiDAR coordinate params boxes_idx_of_points: (B, npoints),
  // default -1

  if (pts_tensor.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(boxes_tensor);
    CHECK_CUDA_INPUT(pts_tensor);
    CHECK_CUDA_INPUT(box_idx_of_points_tensor);

    int batch_size = boxes_tensor.size(0);
    int boxes_num = boxes_tensor.size(1);
    int pts_num = pts_tensor.size(1);

    const float *boxes = boxes_tensor.data_ptr<float>();
    const float *pts = pts_tensor.data_ptr<float>();
    int *box_idx_of_points = box_idx_of_points_tensor.data_ptr<int>();

    points_in_boxes_part_forward_cuda(batch_size, boxes_num, pts_num,
                                      boxes_tensor, pts_tensor,
                                      box_idx_of_points_tensor);
#else
    AT_ERROR("points_in_boxes_part is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("points_in_boxes_part is not implemented on CPU");
  }
}

void points_in_boxes_all_forward(Tensor boxes_tensor, Tensor pts_tensor,
                                 Tensor box_idx_of_points_tensor) {
  // params boxes: (B, N, 7) [x, y, z, x_size, y_size, z_size, rz] in LiDAR
  // coordinate, z is the bottom center. params pts: (B, npoints, 3) [x, y, z]
  // in LiDAR coordinate params boxes_idx_of_points: (B, npoints), default -1

  if (pts_tensor.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(boxes_tensor);
    CHECK_CUDA_INPUT(pts_tensor);
    CHECK_CUDA_INPUT(box_idx_of_points_tensor);

    int batch_size = boxes_tensor.size(0);
    int boxes_num = boxes_tensor.size(1);
    int pts_num = pts_tensor.size(1);

    const float *boxes = boxes_tensor.data_ptr<float>();
    const float *pts = pts_tensor.data_ptr<float>();
    int *box_idx_of_points = box_idx_of_points_tensor.data_ptr<int>();

    points_in_boxes_all_forward_cuda(batch_size, boxes_num, pts_num,
                                     boxes_tensor, pts_tensor,
                                     box_idx_of_points_tensor);
#else
    AT_ERROR("points_in_boxes_all is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("points_in_boxes_all is not implemented on CPU");
  }
}
