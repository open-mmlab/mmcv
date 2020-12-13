#include "furthest_point_sample_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

#define TOTAL_THREADS 1024

inline int opt_n_threads(int work_size) {
  const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

  return max(min(1 << pow_2, TOTAL_THREADS), 1);
}

void FurthestPointSamplingCUDAKernelLauncher(int b, int n, int m,
                                             const Tensor dataset_tensor,
                                             Tensor temp_tensor,
                                             Tensor idxs_tensor) {
  // dataset: (B, N, 3)
  // tmp: (B, N)
  // output:
  //      idx: (B, M)

  cudaError_t err;
  unsigned int n_threads = opt_n_threads(n);
  float *dataset = dataset_tensor.data_ptr<float>();
  float *temp = temp_tensor.data_ptr<float>();
  int *idxs = idxs_tensor.data_ptr<int>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  switch (n_threads) {
    case 1024:
      furthest_point_sampling_cuda_kernel<1024><<<b, n_threads, 0, stream>>>(
          b, n, m, dataset, temp, idxs);
      break;
    case 512:
      furthest_point_sampling_cuda_kernel<512><<<b, n_threads, 0, stream>>>(
          b, n, m, dataset, temp, idxs);
      break;
    case 256:
      furthest_point_sampling_cuda_kernel<256><<<b, n_threads, 0, stream>>>(
          b, n, m, dataset, temp, idxs);
      break;
    case 128:
      furthest_point_sampling_cuda_kernel<128><<<b, n_threads, 0, stream>>>(
          b, n, m, dataset, temp, idxs);
      break;
    case 64:
      furthest_point_sampling_cuda_kernel<64><<<b, n_threads, 0, stream>>>(
          b, n, m, dataset, temp, idxs);
      break;
    case 32:
      furthest_point_sampling_cuda_kernel<32><<<b, n_threads, 0, stream>>>(
          b, n, m, dataset, temp, idxs);
      break;
    case 16:
      furthest_point_sampling_cuda_kernel<16><<<b, n_threads, 0, stream>>>(
          b, n, m, dataset, temp, idxs);
      break;
    case 8:
      furthest_point_sampling_cuda_kernel<8><<<b, n_threads, 0, stream>>>(
          b, n, m, dataset, temp, idxs);
      break;
    case 4:
      furthest_point_sampling_cuda_kernel<4><<<b, n_threads, 0, stream>>>(
          b, n, m, dataset, temp, idxs);
      break;
    case 2:
      furthest_point_sampling_cuda_kernel<2><<<b, n_threads, 0, stream>>>(
          b, n, m, dataset, temp, idxs);
      break;
    case 1:
      furthest_point_sampling_cuda_kernel<1><<<b, n_threads, 0, stream>>>(
          b, n, m, dataset, temp, idxs);
      break;
    default:
      furthest_point_sampling_cuda_kernel<512><<<b, n_threads, 0, stream>>>(
          b, n, m, dataset, temp, idxs);
  }

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

void FurthestPointSamplingWithDistCUDAKernelLauncher(
    int b, int n, int m, const Tensor dataset_tensor, Tensor temp_tensor,
    Tensor idxs_tensor) {
  // dataset: (B, N, N)
  // temp: (B, N)
  // output:
  //      idx: (B, M)

  cudaError_t err;
  unsigned int n_threads = opt_n_threads(n);

  float *dataset = dataset_tensor.data_ptr<float>();
  float *temp = temp_tensor.data_ptr<float>();
  int *idxs = idxs_tensor.data_ptr<int>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  switch (n_threads) {
    case 1024:
      furthest_point_sampling_with_dist_cuda_kernel<
          1024><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 512:
      furthest_point_sampling_with_dist_cuda_kernel<
          512><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 256:
      furthest_point_sampling_with_dist_cuda_kernel<
          256><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 128:
      furthest_point_sampling_with_dist_cuda_kernel<
          128><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 64:
      furthest_point_sampling_with_dist_cuda_kernel<
          64><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 32:
      furthest_point_sampling_with_dist_cuda_kernel<
          32><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 16:
      furthest_point_sampling_with_dist_cuda_kernel<
          16><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 8:
      furthest_point_sampling_with_dist_cuda_kernel<
          8><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 4:
      furthest_point_sampling_with_dist_cuda_kernel<
          4><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 2:
      furthest_point_sampling_with_dist_cuda_kernel<
          2><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    case 1:
      furthest_point_sampling_with_dist_cuda_kernel<
          1><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
      break;
    default:
      furthest_point_sampling_with_dist_cuda_kernel<
          512><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
  }

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
