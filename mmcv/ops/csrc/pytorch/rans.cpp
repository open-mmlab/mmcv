// Copyright(c) OpenMMLab.All rights reserved.
#include <pybind11/pybind11.h>

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

namespace py = pybind11;
std::string rans_encode_with_indexes_impl(
    const Tensor symbols, const Tensor indexes, const Tensor cdfs,
    const Tensor cdfs_sizes, const Tensor offsets, int num_threads) {
  return DISPATCH_DEVICE_IMPL(rans_encode_with_indexes_impl, symbols, indexes,
                              cdfs, cdfs_sizes, offsets, num_threads);
}

py::bytes rans_encode_with_indexes(const Tensor symbols, const Tensor indexes,
                                   const Tensor cdfs, const Tensor cdfs_sizes,
                                   const Tensor offsets, int num_threads) {
  return rans_encode_with_indexes_impl(symbols, indexes, cdfs, cdfs_sizes,
                                       offsets, num_threads);
}

Tensor rans_decode_with_indexes_impl(const std::string& encoded,
                                     const Tensor indexes, const Tensor cdfs,
                                     const Tensor cdfs_sizes,
                                     const Tensor offsets) {
  return DISPATCH_DEVICE_IMPL(rans_decode_with_indexes_impl, encoded, indexes,
                              cdfs, cdfs_sizes, offsets);
}

Tensor rans_decode_with_indexes(const std::string& encoded,
                                const Tensor indexes, const Tensor cdfs,
                                const Tensor cdfs_sizes, const Tensor offsets) {
  return rans_decode_with_indexes_impl(encoded, indexes, cdfs, cdfs_sizes,
                                       offsets);
}

Tensor pmf_to_quantized_cdf_impl(const Tensor pmfs, const Tensor pmf_lengths,
                                 const Tensor tail_masses) {
  return DISPATCH_DEVICE_IMPL(pmf_to_quantized_cdf_impl, pmfs, pmf_lengths,
                              tail_masses);
}

Tensor pmf_to_quantized_cdf(const Tensor pmfs, const Tensor pmf_lengths,
                            const Tensor tail_masses) {
  return pmf_to_quantized_cdf_impl(pmfs, pmf_lengths, tail_masses);
}
