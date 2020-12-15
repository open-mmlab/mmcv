#ifndef _FURTHEST_POINT_SAMPLE_EXT_PYTORCH
#define _FURTHEST_POINT_SAMPLE_EXT_PYTORCH
#include <torch/extension.h>
using namespace at;

int furthest_point_sampling(int b, int n, int m, const Tensor points,
                            Tensor temp, Tensor idx);

int furthest_point_sampling_with_dist(int b, int n, int m, const Tensor points,
                                      Tensor temp, Tensor idx);
#endif  //_FURTHEST_POINT_SAMPLE_EXT_PYTORCH
