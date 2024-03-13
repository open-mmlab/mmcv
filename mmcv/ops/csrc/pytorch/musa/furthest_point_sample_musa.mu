// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/sampling_gpu.cu

#include <stdio.h>
#include <stdlib.h>

#include "furthest_point_sample_musa_kernel.muh"
#include "pytorch_musa_helper.hpp"

inline int opt_n_threads(int work_size) {
  const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

  return max(min(1 << pow_2, 1024), 1);
}

void FurthestPointSamplingForwardMUSAKernelLauncher(int b, int n, int m,
                                                    const float* dataset,
                                                    float* temp, int* idxs) {
  // dataset: (B, N, 3)
  // tmp: (B, N)
  // output:
  //      idx: (B, M)

  musaStream_t stream = c10::musa::getCurrentMUSAStream();

  unsigned int n_threads = opt_n_threads(n);

  switch (n_threads) {
    case 1024:
      furthest_point_sampling_forward_musa_kernel<1024>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 512:
      furthest_point_sampling_forward_musa_kernel<512>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 256:
      furthest_point_sampling_forward_musa_kernel<256>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 128:
      furthest_point_sampling_forward_musa_kernel<128>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 64:
      furthest_point_sampling_forward_musa_kernel<64>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 32:
      furthest_point_sampling_forward_musa_kernel<32>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 16:
      furthest_point_sampling_forward_musa_kernel<16>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 8:
      furthest_point_sampling_forward_musa_kernel<8>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 4:
      furthest_point_sampling_forward_musa_kernel<4>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 2:
      furthest_point_sampling_forward_musa_kernel<2>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 1:
      furthest_point_sampling_forward_musa_kernel<1>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    default:
      furthest_point_sampling_forward_musa_kernel<512>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
  }

  AT_MUSA_CHECK(musaGetLastError());
}

void FurthestPointSamplingWithDistForwardMUSAKernelLauncher(
    int b, int n, int m, const float* dataset, float* temp, int* idxs) {
  // dataset: (B, N, N)
  // temp: (B, N)
  // output:
  //      idx: (B, M)

  musaStream_t stream = c10::musa::getCurrentMUSAStream();

  unsigned int n_threads = opt_n_threads(n);

  switch (n_threads) {
    case 1024:
      furthest_point_sampling_with_dist_forward_musa_kernel<1024>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 512:
      furthest_point_sampling_with_dist_forward_musa_kernel<512>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 256:
      furthest_point_sampling_with_dist_forward_musa_kernel<256>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 128:
      furthest_point_sampling_with_dist_forward_musa_kernel<128>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 64:
      furthest_point_sampling_with_dist_forward_musa_kernel<64>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 32:
      furthest_point_sampling_with_dist_forward_musa_kernel<32>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 16:
      furthest_point_sampling_with_dist_forward_musa_kernel<16>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 8:
      furthest_point_sampling_with_dist_forward_musa_kernel<8>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 4:
      furthest_point_sampling_with_dist_forward_musa_kernel<4>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 2:
      furthest_point_sampling_with_dist_forward_musa_kernel<2>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 1:
      furthest_point_sampling_with_dist_forward_musa_kernel<1>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    default:
      furthest_point_sampling_with_dist_forward_musa_kernel<512>
          <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
  }

  AT_MUSA_CHECK(musaGetLastError());
}
