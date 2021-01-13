#ifndef TRT_CUDA_HELPER_HPP
#define TRT_CUDA_HELPER_HPP

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(0);                                                 \
    }                                                          \
  }

#endif  // TRT_CUDA_HELPER_HPP
