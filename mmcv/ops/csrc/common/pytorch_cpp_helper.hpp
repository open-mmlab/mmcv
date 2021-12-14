#ifndef PYTORCH_CPP_HELPER
#define PYTORCH_CPP_HELPER
#include <torch/extension.h>

#include <vector>

using namespace at;

// #define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
#define THREADS_PER_BLOCK 512

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) \
  TORCH_CHECK(!x.device().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_INPUT(x) \
  CHECK_CUDA(x);            \
  CHECK_CONTIGUOUS(x)
#define CHECK_CPU_INPUT(x) \
  CHECK_CPU(x);            \
  CHECK_CONTIGUOUS(x)

inline int GET_BLOCKS(const int N, const int num_threads = THREADS_PER_BLOCK) {
  int optimal_block_num = (N + num_threads - 1) / num_threads;
  int max_block_num = 4096;
  return std::min(optimal_block_num, max_block_num);
}

#endif  // PYTORCH_CPP_HELPER
