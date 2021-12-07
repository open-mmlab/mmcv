// Modified from
// https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp

/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;

void iou3d_boxes_overlap_bev_forward_impl(const int num_a, const Tensor boxes_a,
                                          const int num_b, const Tensor boxes_b,
                                          Tensor ans_overlap) {
  DISPATCH_DEVICE_IMPL(iou3d_boxes_overlap_bev_forward_impl, num_a, boxes_a,
                       num_b, boxes_b, ans_overlap);
}

void iou3d_boxes_iou_bev_forward_impl(const int num_a, const Tensor boxes_a,
                                      const int num_b, const Tensor boxes_b,
                                      Tensor ans_iou) {
  DISPATCH_DEVICE_IMPL(iou3d_boxes_iou_bev_forward_impl, num_a, boxes_a, num_b,
                       boxes_b, ans_iou);
}

void iou3d_nms_forward_impl(const Tensor boxes, unsigned long long *mask,
                            int boxes_num, float nms_overlap_thresh) {
  DISPATCH_DEVICE_IMPL(iou3d_nms_forward_impl, boxes, mask, boxes_num,
                       nms_overlap_thresh);
}

void iou3d_nms_normal_forward_impl(const Tensor boxes, unsigned long long *mask,
                                   int boxes_num, float nms_overlap_thresh) {
  DISPATCH_DEVICE_IMPL(iou3d_nms_normal_forward_impl, boxes, mask, boxes_num,
                       nms_overlap_thresh);
}

void iou3d_boxes_overlap_bev_forward(Tensor boxes_a, Tensor boxes_b,
                                     Tensor ans_overlap) {
  // params boxes_a: (N, 5) [x1, y1, x2, y2, ry]
  // params boxes_b: (M, 5)
  // params ans_overlap: (N, M)

  int num_a = boxes_a.size(0);
  int num_b = boxes_b.size(0);

  iou3d_boxes_overlap_bev_forward_impl(num_a, boxes_a, num_b, boxes_b,
                                       ans_overlap);
}

void iou3d_boxes_iou_bev_forward(Tensor boxes_a, Tensor boxes_b,
                                 Tensor ans_iou) {
  // params boxes_a: (N, 5) [x1, y1, x2, y2, ry]
  // params boxes_b: (M, 5)
  // params ans_overlap: (N, M)
  int num_a = boxes_a.size(0);
  int num_b = boxes_b.size(0);

  iou3d_boxes_iou_bev_forward_impl(num_a, boxes_a, num_b, boxes_b, ans_iou);
}

void iou3d_nms_forward(Tensor boxes, Tensor keep, Tensor keep_num,
                       float nms_overlap_thresh) {
  // params boxes: (N, 5) [x1, y1, x2, y2, ry]
  // params keep: (N)
  CHECK_CONTIGUOUS(boxes);
  CHECK_CONTIGUOUS(keep);

  int boxes_num = boxes.size(0);
  int64_t *keep_data = keep.data_ptr<int64_t>();
  int64_t *keep_num_data = keep_num.data_ptr<int64_t>();

  const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

  Tensor mask =
      at::empty({boxes_num, col_blocks}, boxes.options().dtype(at::kLong));
  unsigned long long *mask_data =
      (unsigned long long *)mask.data_ptr<int64_t>();
  iou3d_nms_forward_impl(boxes, mask_data, boxes_num, nms_overlap_thresh);

  at::Tensor mask_cpu = mask.to(at::kCPU);
  unsigned long long *mask_host =
      (unsigned long long *)mask_cpu.data_ptr<int64_t>();

  std::vector<unsigned long long> remv_cpu(col_blocks);
  memset(&remv_cpu[0], 0, sizeof(unsigned long long) * col_blocks);

  int num_to_keep = 0;

  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / THREADS_PER_BLOCK_NMS;
    int inblock = i % THREADS_PER_BLOCK_NMS;

    if (!(remv_cpu[nblock] & (1ULL << inblock))) {
      keep_data[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv_cpu[j] |= p[j];
      }
    }
    *keep_num_data = num_to_keep;
  }
}

void iou3d_nms_normal_forward(Tensor boxes, Tensor keep, Tensor keep_num,
                              float nms_overlap_thresh) {
  // params boxes: (N, 5) [x1, y1, x2, y2, ry]
  // params keep: (N)

  CHECK_CONTIGUOUS(boxes);
  CHECK_CONTIGUOUS(keep);

  int boxes_num = boxes.size(0);
  int64_t *keep_data = keep.data_ptr<int64_t>();
  int64_t *keep_num_data = keep_num.data_ptr<int64_t>();

  const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

  Tensor mask =
      at::empty({boxes_num, col_blocks}, boxes.options().dtype(at::kLong));
  unsigned long long *mask_data =
      (unsigned long long *)mask.data_ptr<int64_t>();
  iou3d_nms_normal_forward_impl(boxes, mask_data, boxes_num,
                                nms_overlap_thresh);

  at::Tensor mask_cpu = mask.to(at::kCPU);
  unsigned long long *mask_host =
      (unsigned long long *)mask_cpu.data_ptr<int64_t>();

  std::vector<unsigned long long> remv_cpu(col_blocks);
  memset(&remv_cpu[0], 0, sizeof(unsigned long long) * col_blocks);
  int num_to_keep = 0;

  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / THREADS_PER_BLOCK_NMS;
    int inblock = i % THREADS_PER_BLOCK_NMS;

    if (!(remv_cpu[nblock] & (1ULL << inblock))) {
      keep_data[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv_cpu[j] |= p[j];
      }
    }
  }

  *keep_num_data = num_to_keep;
}
