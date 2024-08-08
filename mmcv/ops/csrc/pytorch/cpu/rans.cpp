// Copyright(c) OpenMMLab.All rights reserved.

// Modeified from
// https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/cpp_exts/rans/rans_interface.hpp
#include "utils/rans/rans.hpp"

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#include "utils/rans/rans64.h"

// ------------------- RANS ENCODING -------------------
std::string RansEncodeWithIndexesCPUKernelLaucher(
    const Tensor symbols, const Tensor indexes, const Tensor cdfs,
    const Tensor cdfs_sizes, const Tensor offsets, int num_threads) {
  // check input
  check_rans_encode_input(symbols, indexes, cdfs, cdfs_sizes, offsets,
                          num_threads);

  // convert tensor to vector
  const auto symbols_vector = TENSOR_TO_VECTOR_1D(symbols, int32_t);
  const auto indexes_vector = TENSOR_TO_VECTOR_1D(indexes, int32_t);
  std::vector<std::vector<int32_t>> cdfs_vector(cdfs.size(0));
  for (int i = 0; i < cdfs.size(0); i++) {
    cdfs_vector[i] = TENSOR_TO_VECTOR_1D(cdfs[i], int32_t);
  }
  const auto cdfs_sizes_vector = TENSOR_TO_VECTOR_1D(cdfs_sizes, int32_t);
  const auto offsets_vector = TENSOR_TO_VECTOR_1D(offsets, int32_t);

  // prepare outputs and nbytes
  std::vector<std::vector<uint32_t>> outputs(
      num_threads, std::vector<uint32_t>(symbols.size(0) / num_threads, 0xCC));
  std::vector<uint32_t> nbytes(num_threads, 0);

  // multi-threading
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; i++) {
    int begin_idx = GET_BEGIN_IDX(i, symbols.size(0), num_threads);
    int end_idx = GET_END_IDX(i, symbols.size(0), num_threads);
    threads.push_back(std::thread(
        rans_encode_with_indexes_cpu_kernel, std::cref(symbols_vector),
        std::cref(indexes_vector), std::cref(cdfs_vector),
        std::cref(cdfs_sizes_vector), std::cref(offsets_vector), begin_idx,
        end_idx, std::ref(outputs[i]), std::ref(nbytes[i])));
  }

  // start threads
  for (auto &t : threads) {
    t.join();
  }

  // set header
  RansHeader header;
  header.encode_with_cpu = true;
  header.num_threads = num_threads;
  header.nbytes_mode =
      get_nbytes_mode(*std::max_element(nbytes.begin(), nbytes.end()));

  // output encoded
  std::string encoded = "";
  for (int i = 0; i < num_threads; i++) {
    uint32_t *ptr =
        outputs[i].data() + outputs[i].size() - nbytes[i] / sizeof(uint32_t);
    encoded += std::string(reinterpret_cast<char *>(ptr), nbytes[i]);
  }
  // output nbytes
  encoded += compress_nbytes(nbytes.data(), header.nbytes_mode, num_threads);
  // output header
  encoded += compress_rans_header(header);

  return encoded;
}

// ------------------- RANS DECODING -------------------
Tensor RansDecodeWithIndexesCPUKernelLaucher(const std::string &encoded,
                                             const Tensor indexes,
                                             const Tensor cdfs,
                                             const Tensor cdfs_sizes,
                                             const Tensor offsets) {
  // check input
  check_rans_decode_input(encoded, indexes, cdfs, cdfs_sizes, offsets);

  // allocate output
  Tensor output = torch::zeros(indexes.sizes(), torch::kInt32);

  // convert the input to vector
  const auto indexes_vector = TENSOR_TO_VECTOR_1D(indexes, int32_t);
  std::vector<std::vector<int32_t>> cdfs_vector(cdfs.size(0));
  for (int i = 0; i < cdfs.size(0); i++) {
    cdfs_vector[i] = TENSOR_TO_VECTOR_1D(cdfs[i], int32_t);
  }
  const auto cdfs_sizes_vector = TENSOR_TO_VECTOR_1D(cdfs_sizes, int32_t);
  const auto offsets_vector = TENSOR_TO_VECTOR_1D(offsets, int32_t);

  // decode header
  RansHeader header = decompress_rans_header(
      encoded.substr(encoded.length() - COMPRESS_HEADER_BYTES));
  check_num_threads(header.num_threads);
  int num_threads = header.num_threads;

  // decode nbytes
  uint32_t *nbytes = new uint32_t[num_threads];
  int nbytes_bytes = num_threads * get_nbytes_size(header.nbytes_mode);
  decompress_nbytes(
      encoded.substr(encoded.length() - COMPRESS_HEADER_BYTES - nbytes_bytes),
      nbytes, num_threads, header.nbytes_mode);

  // multi-threading
  std::vector<std::thread> threads;
  uint32_t decoded_start_idx = 0;
#pragma unroll
  for (int i = 0; i < num_threads; ++i) {
    int begin_idx = GET_BEGIN_IDX(i, indexes.size(0), num_threads);
    int end_idx = GET_END_IDX(i, indexes.size(0), num_threads);
    threads.push_back(
        std::thread(rans_decode_with_indexes_cpu_kernel,
                    (uint32_t *)(encoded.data()) + decoded_start_idx,
                    std::cref(indexes_vector), std::cref(cdfs_vector),
                    std::cref(cdfs_sizes_vector), std::cref(offsets_vector),
                    begin_idx, end_idx, output));
    decoded_start_idx += nbytes[i] / sizeof(uint32_t);
  }

  for (auto &t : threads) {
    t.join();
  }

  // free memory
  delete[] nbytes;

  return output;
}

// ------------------- PMF TO QUANTIZED CDF -------------------
Tensor PMFtoQuantizedCDFCPUKernelLauncher(const Tensor pmfs,
                                          const Tensor pmf_lengths,
                                          const Tensor tail_masses) {
  check_pmf_to_quantized_cdf_input(pmfs, pmf_lengths, tail_masses);
  Tensor max_length_tensor = torch::max(pmf_lengths);
  int cdf_max_length = max_length_tensor.item<int>() + 2;
  Tensor quantized_cdfs = zeros({pmfs.size(0), cdf_max_length},
                                TensorOptions().dtype(ScalarType::Int));
  pmf_to_quantized_cdf_cpu_kernel(pmfs, pmf_lengths, tail_masses,
                                  quantized_cdfs);
  return quantized_cdfs;
}

// ------------------- BIND RANS ENCODE -------------------
std::string rans_encode_with_indexes_cpu(
    const Tensor symbols, const Tensor indexes, const Tensor cdfs,
    const Tensor cdfs_sizes, const Tensor offsets, int num_threads) {
  return RansEncodeWithIndexesCPUKernelLaucher(
      symbols, indexes, cdfs, cdfs_sizes, offsets, num_threads);
}
std::string rans_encode_with_indexes_impl(
    const Tensor symbols, const Tensor indexes, const Tensor cdfs,
    const Tensor cdfs_sizes, const Tensor offsets, int num_threads);

REGISTER_DEVICE_IMPL(rans_encode_with_indexes_impl, CPU,
                     rans_encode_with_indexes_cpu);

// ------------------- BIND RANS DECODE -------------------
Tensor rans_decode_with_indexes_cpu(const std::string &encoded,
                                    const Tensor indexes, const Tensor cdfs,
                                    const Tensor cdfs_sizes,
                                    const Tensor offsets) {
  return RansDecodeWithIndexesCPUKernelLaucher(encoded, indexes, cdfs,
                                               cdfs_sizes, offsets);
}
Tensor rans_decode_with_indexes_impl(const std::string &encoded,
                                     const Tensor indexes, const Tensor cdfs,
                                     const Tensor cdfs_sizes,
                                     const Tensor offsets);

REGISTER_DEVICE_IMPL(rans_decode_with_indexes_impl, CPU,
                     rans_decode_with_indexes_cpu);

// ------------------- BIND PMF TO QUANTIZED CDF -------------------
Tensor pmf_to_quantized_cdf_cpu(const Tensor pmfs, const Tensor pmf_lengths,
                                const Tensor tail_masses) {
  return PMFtoQuantizedCDFCPUKernelLauncher(pmfs, pmf_lengths, tail_masses);
}

Tensor pmf_to_quantized_cdf_impl(const Tensor pmfs, const Tensor pmf_lengths,
                                 const Tensor tail_masses);
REGISTER_DEVICE_IMPL(pmf_to_quantized_cdf_impl, CPU, pmf_to_quantized_cdf_cpu);
