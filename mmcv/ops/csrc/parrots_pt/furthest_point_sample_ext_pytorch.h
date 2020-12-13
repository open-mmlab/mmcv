#ifndef _FURTHEST_POINT_SAMPLE_EXT_PYTORCH
#define _FURTHEST_POINT_SAMPLE_EXT_PYTORCH
#include <torch/extension.h>

int furthest_point_sampling_wrapper(int b, int n, int m,
                                    at::Tensor points_tensor,
                                    at::Tensor temp_tensor,
                                    at::Tensor idx_tensor);


int furthest_point_sampling_with_dist_wrapper(int b, int n, int m,
                                              at::Tensor points_tensor,
                                              at::Tensor temp_tensor,
                                              at::Tensor idx_tensor);

#endif //_FURTHEST_POINT_SAMPLE_EXT_PYTORCH
