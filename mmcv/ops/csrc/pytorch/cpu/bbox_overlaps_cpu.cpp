// Copyright(c) OpenMMLab.All rights reserved.
#include "box_iou_rotated_utils.hpp"
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

template <typename T>
void bbox_overlaps_cpu_kernel(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                              const int mode_flag, const bool aligned, const int offset)
{
    int output_size = ious.numel();
    auto num_boxes1 = boxes1.size(0);
    auto num_boxes2 = boxes2.size(0);

    if (aligned)
    {
        for (int i = 0; i < output_size; i++)
        {
            ious[i] = single_box_iou<T>(boxes1[i].data_ptr<T>(),
                                        boxes2[i].data_ptr<T>(), offset, mode_flag);
        }
    }
    else
    {
        for (int i = 0; i < num_boxes1; i++)
        {
            for (int j = 0; j < num_boxes2; j++)
            {
                ious[i][j] = single_box_iou<T>(
                    boxes1[i].data_ptr<T>(), boxes2[j].data_ptr<T>(), offset, mode_flag);
            }
        }
    }
}

void bbox_overlaps_cpu(const Tensor boxes1, const Tensor boxes2, Tensor ious, const int mode,
                       const bool aligned, const int offset)
{
    bbox_overlaps_cpu_kernel<float>(boxes1, boxes2, ious, mode, aligned, offset);
}

void bbox_overlaps_impl(const Tensor boxes1, const Tensor boxes2, Tensor ious, const int mode,
                        const bool aligned, const int offset);

REGISTER_DEVICE_IMPL(bbox_overlaps_impl, CPU, bbox_overlaps_cpu);
