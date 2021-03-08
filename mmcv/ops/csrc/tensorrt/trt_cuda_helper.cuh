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

/**
 * Returns a view of the original tensor with its dimensions permuted.
 *
 * @param[out] dst pointer to the destination tensor
 * @param[in] src pointer to the source tensor
 * @param[in] src_size shape of the src tensor
 * @param[in] permute The desired ordering of dimensions
 * @param[in] src_dim dim of src tensor
 * @param[in] stream cuda stream handle
 */
template <class scalar_t>
void memcpyPermute(scalar_t *dst, const scalar_t *src, int *src_size,
                   int *permute, int src_dim, cudaStream_t stream = 0);

#endif  // TRT_CUDA_HELPER_HPP
