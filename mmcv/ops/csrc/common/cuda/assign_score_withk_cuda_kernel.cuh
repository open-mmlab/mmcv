// Copyright (c) OpenMMLab. All rights reserved
#ifndef ASSIGN_SCORE_WITHK_CUDA_KERNEL_CUH
#define ASSIGN_SCORE_WITHK_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

// input: points(B,N0,M,O), centers(B,N0,M,O), scores(B,N1,K,M), knn_idx(B,N1,K)
// output: fout(B,O,N)
// algo: fout(b,i,k,j) = s(b,i,k,m)*p(b,c(i),k,m,j) =  s(b,i,k,m)*p(b,i(k),m,j)
//       i(k) = idx(b,i,k)
//      sum: fout(b,i,j) = fout(b,i,j) + s(b,i,k,m)*p(b,i,k,m,j)
//      avg: fout(b,i,j) = sum(fout(b,i,k,j)) / k
//      max: fout(b,i,j) = max(fout(b,i,k,j), sum(s(b,i,k,m)*p(b,i,k,m,j)))

template <typename T>
__global__ void assign_score_withk_forward_cuda_kernel(
    const int B, const int N0, const int N1, const int M, const int K,
    const int O, const int aggregate, const T* points, const T* centers,
    const T* scores, const int64_t* knn_idx, T* output) {
  // ----- parallel loop for B, N1, K and O ---------
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= B * N1 * K * O) return;
  // ------- loop for M ----------
  for (int m = 0; m < M; m++) {
    int b = (int)(i / (O * N1 * K));
    int o = (int)(i % (O * N1 * K) / (N1 * K));
    int n = (int)(i % (N1 * K) / K);
    int k = (int)(i % K);
    int cn = (int)knn_idx[b * K * N1 + n * K +
                          0];  // The first neighbor is the center point
    int kn = (int)knn_idx[b * K * N1 + n * K + k];
    if (kn >= N0 ||
        kn < 0) {  // if index overflows, it is out of the neighborhood range
      continue;
    }
    assert(b < B);
    assert(kn < N0);
    assert(cn < N0);
    assert(o < O);
    assert(n < N1);
    atomicAdd(output + b * N1 * O * K + o * N1 * K + n * K + k,
              points[b * N0 * M * O + kn * M * O + m * O + o] *
                      scores[b * N1 * K * M + n * K * M + k * M + m] -
                  centers[b * N0 * M * O + cn * M * O + m * O + o] *
                      scores[b * N1 * K * M + n * K * M + k * M + m]);
  }
}

template <typename T>
__global__ void assign_score_withk_points_backward_cuda_kernel(
    const int B, const int N0, const int N, const int M, const int K,
    const int O, const int aggregate, const T* grad_out, const T* scores,
    const int64_t* knn_idx, T* grad_points, T* grad_centers) {
  // ----- parallel loop for B, M, O ---------
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= B * M * O) return;
  int b = (int)(i / (M * O));
  int m = (int)(i % (M * O) / O);
  int o = (int)(i % O);

  // ----- loop for N,K ---------
  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k++) {
      int kn = knn_idx[b * N * K + n * K + k];
      int cn = knn_idx[b * N * K + n * K + 0];
      if (kn >= N0 ||
          kn < 0) {  // if index overflows, it is out of the neighborhood range
        continue;
      }
      atomicAdd(grad_points + b * N0 * M * O + kn * M * O + m * O + o,
                scores[b * N * K * M + n * K * M + k * M + m] *
                    grad_out[b * O * N * K + o * N * K + n * K + k]);
      atomicAdd(grad_centers + b * N0 * M * O + cn * M * O + m * O + o,
                -scores[b * N * K * M + n * K * M + k * M + m] *
                    grad_out[b * O * N * K + o * N * K + n * K + k]);
    }
  }
}

template <typename T>
__global__ void assign_score_withk_scores_backward_cuda_kernel(
    const int B, const int N0, const int N, const int M, const int K,
    const int O, const int aggregate, const T* grad_out, const T* points,
    const T* centers, const int64_t* knn_idx, T* grad_scores) {
  // ----- parallel loop for B, N, K, M ---------
  long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= B * N * K * M) return;
  int b = (int)(i / (N * M * K));
  int n = (int)(i % (N * M * K) / M / K);
  int k = (int)(i % (M * K) / M);
  int m = (int)(i % M);
  int cn = knn_idx[b * N * K + n * K + 0];
  int kn = knn_idx[b * N * K + n * K + k];
  if (kn >= N0 ||
      kn < 0) {  // if index overflows, it is out of the neighborhood range
    return;
  }

  // -------------- loop for O ------------------------
  for (int o = 0; o < O; o++) {
    atomicAdd(grad_scores + b * N * K * M + n * K * M + k * M + m,
              (points[b * N0 * M * O + kn * M * O + m * O + o] -
               centers[b * N0 * M * O + cn * M * O + m * O + o]) *
                  grad_out[b * O * N * K + o * N * K + n * K + k]);
  }
}

#endif  // ASSIGN_SCORE_WITHK_CUDA_KERNEL_CUH
