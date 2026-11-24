// Copyright (c) OpenMMLab. All rights reserved

#include "pytorch_cuda_helper.hpp"
#include "rans_cuda_kernel.cuh"

// ------------------- RANS ENCODING -------------------
// vector would be optimized by compiler more than 10x faster than Tensor
void rans_encode_cpu_kernel(const std::vector<RansSymbol> &symbols,
                            const std::vector<RansCUDAKernelResult> &results,
                            const int begin_idx, const int end_idx,
                            std::vector<uint32_t> &output, uint32_t &nbyte) {
  Rans64State rans;
  Rans64EncInit(&rans);

  uint32_t *ptr = output.data() + output.size();
  assert(ptr != nullptr);

  for (int i = end_idx - 1; i >= begin_idx; i--) {
    if (results[i].bypass_mode) {
      // unlikely...
      /* Determine the number of bypasses (in bypass_precision size) needed to
       * encode the raw value. */
      std::vector<RansSymbol> _syms;
      int32_t n_bypass = 0;
      uint32_t raw_val = results[i].raw_value;
      while ((raw_val >> (n_bypass * bypass_precision)) != 0) {
        ++n_bypass;
      }

      /* Encode number of bypasses */
      int32_t val = n_bypass;
      while (val >= max_bypass_val) {
        _syms.push_back({max_bypass_val, max_bypass_val + 1, true});
        val -= max_bypass_val;
      }
      _syms.push_back(
          {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});

      /* Encode raw value */
      for (int32_t j = 0; j < n_bypass; ++j) {
        const int32_t val =
            (raw_val >> (j * bypass_precision)) & max_bypass_val;
        _syms.push_back(
            {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});
      }

      while (!_syms.empty()) {
        const RansSymbol sym = _syms.back();
        Rans64EncPutBits(&rans, &ptr, sym.start, bypass_precision);
        _syms.pop_back();
      }
    }
    Rans64EncPut(&rans, &ptr, symbols[i].start, symbols[i].range, precision);
  }

  Rans64EncFlush(&rans, &ptr);

  nbyte = std::distance(ptr, output.data() + output.size()) * sizeof(uint32_t);
}

std::string RansEncodeWithIndexesCUDAKernelLauncher(
    const Tensor symbols, const Tensor indexes, const Tensor cdfs,
    const Tensor cdfs_sizes, const Tensor offsets, int num_threads) {
  // param symbols: (N, )
  // param indexes: (N, )
  // param cdfs: (M, max_cdf_bin_size)
  // param cdfs_sizes: (M, )
  // param offsets: (M, )

  // check input
  check_rans_encode_input(symbols, indexes, cdfs, cdfs_sizes, offsets,
                          num_threads);

  // set device
  at::cuda::CUDAGuard device_guard(symbols.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // count number of symbols
  const int num_symbols = symbols.size(0);
  const int num_cdfs = cdfs.size(0);
  const int max_cdf_bin_size = cdfs.size(1);

  // set kernel params
  int cuda_threads = min(num_symbols, THREADS_PER_BLOCK);
  dim3 blocks((num_symbols + cuda_threads - 1) / cuda_threads);
  dim3 threads(cuda_threads);

  // malloc device memory for cache results
  auto rans_symbols_tensor =
      CREATE_TENSOR(num_symbols * sizeof(RansSymbol), torch::ScalarType::Byte,
                    symbols.device());  // store encoded symbols;
  RansSymbol *rans_symbols = (RansSymbol *)rans_symbols_tensor.data_ptr();
  auto results_tensor = CREATE_TENSOR(
      num_symbols * sizeof(RansCUDAKernelResult), torch::ScalarType::Byte,
      symbols.device());  // store whether value == max_value;
  RansCUDAKernelResult *results =
      (RansCUDAKernelResult *)results_tensor.data_ptr();
  // get encoded symbols
  rans_encode_with_indexes_cuda_kernel<<<blocks, threads, 0, stream>>>(
      symbols.data_ptr<int>(), indexes.data_ptr<int>(), cdfs.data_ptr<int>(),
      cdfs_sizes.data_ptr<int>(), offsets.data_ptr<int>(), num_symbols,
      num_cdfs, max_cdf_bin_size, rans_symbols, results);
  // sync
  cudaStreamSynchronize(stream);

  // move results to cpu vector
  auto rans_symbols_tensor_cpu = rans_symbols_tensor.to(torch::kCPU);
  auto results_tensor_cpu = results_tensor.to(torch::kCPU);
  auto rstc_ptr =
      reinterpret_cast<RansSymbol *>(rans_symbols_tensor_cpu.data_ptr());
  auto rt_ptr =
      reinterpret_cast<RansCUDAKernelResult *>(results_tensor_cpu.data_ptr());
  auto rans_symbols_vector =
      std::vector<RansSymbol>(rstc_ptr, rstc_ptr + num_symbols);
  auto results_vector =
      std::vector<RansCUDAKernelResult>(rt_ptr, rt_ptr + num_symbols);

  // prepare outputs and nbytes
  std::vector<std::vector<uint32_t>> outputs(
      num_threads, std::vector<uint32_t>(symbols.size(0) / num_threads, 0xCC));
  std::vector<uint32_t> nbytes(num_threads, 0);

  // multi-threading
  std::vector<std::thread> cpu_threads;
#pragma unroll
  for (int i = 0; i < num_threads; i++) {
    int begin_idx = GET_BEGIN_IDX(i, symbols.size(0), num_threads);
    int end_idx = GET_END_IDX(i, symbols.size(0), num_threads);
    cpu_threads.push_back(
        std::thread(rans_encode_cpu_kernel, std::cref(rans_symbols_vector),
                    std::cref(results_vector), begin_idx, end_idx,
                    std::ref(outputs[i]), std::ref(nbytes[i])));
  }

  // start threads
  for (auto &t : cpu_threads) {
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
#pragma unroll
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

Tensor RansDecodeWithIndexesCUDAKernelLauncher(const std::string &encoded,
                                               const Tensor indexes,
                                               const Tensor cdfs,
                                               const Tensor cdfs_sizes,
                                               const Tensor offsets) {
  // param encoded: encoded symbols bitstream
  // param indexes: (N, )
  // param cdfs: (M, max_cdf_bin_size)
  // param cdfs_sizes: (M, )
  // param offsets: (M, )
  check_rans_decode_input(encoded, indexes, cdfs, cdfs_sizes, offsets);

  // set device
  at::cuda::CUDAGuard device_guard(indexes.device());

  // allocate output
  Tensor output = torch::zeros(indexes.sizes(), torch::kInt32);

  // convert the input to vector
  auto indexes_cpu = indexes.to(torch::kCPU);
  auto cdfs_cpu = cdfs.to(torch::kCPU);
  auto cdfs_sizes_cpu = cdfs_sizes.to(torch::kCPU);
  auto offsets_cpu = offsets.to(torch::kCPU);
  const auto indexes_vector = TENSOR_TO_VECTOR_1D(indexes_cpu, int32_t);
  std::vector<std::vector<int32_t>> cdfs_vector(cdfs.size(0));
  for (int i = 0; i < cdfs.size(0); i++) {
    cdfs_vector[i] = TENSOR_TO_VECTOR_1D(cdfs_cpu[i], int32_t);
  }
  const auto cdfs_sizes_vector = TENSOR_TO_VECTOR_1D(cdfs_sizes_cpu, int32_t);
  const auto offsets_vector = TENSOR_TO_VECTOR_1D(offsets_cpu, int32_t);

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

  return output.to(indexes.device());
}

Tensor PMFtoQuantizedCDFCUDAKernelLauncher(const Tensor pmfs,
                                           const Tensor pmf_lengths,
                                           const Tensor tail_masses) {
  check_pmf_to_quantized_cdf_input(pmfs, pmf_lengths, tail_masses);

  // set device
  at::cuda::CUDAGuard device_guard(pmfs.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // gpu not supported
  if (pmfs.size(1) > THREADS_PER_BLOCK) {
    Tensor max_length_tensor = torch::max(pmf_lengths).cpu();
    int cdf_max_length = max_length_tensor.item<int>() + 2;
    Tensor pmfs_cpu = pmfs.to(torch::kCPU);
    Tensor pmf_lengths_cpu = pmf_lengths.to(torch::kCPU);
    Tensor tail_masses_cpu = tail_masses.to(torch::kCPU);
    Tensor quantized_cdfs_cpu = zeros({pmfs.size(0), cdf_max_length},
                                      TensorOptions().dtype(ScalarType::Int));
    pmf_to_quantized_cdf_cpu_kernel(pmfs_cpu, pmf_lengths_cpu, tail_masses_cpu,
                                    quantized_cdfs_cpu);
    return quantized_cdfs_cpu.to(pmfs.device());
  }

  // allocate output
  Tensor max_length_tensor = torch::max(pmf_lengths).cpu();
  int cdf_max_length = max_length_tensor.item<int>() + 2;
  Tensor quantized_cdfs =
      zeros({pmfs.size(0), cdf_max_length},
            TensorOptions().dtype(ScalarType::Int).device(pmfs.device()));

  // launch kernel
  dim3 blocks = dim3(pmfs.size(0));
  dim3 threads = dim3(cdf_max_length);
  pmf_to_quantized_cdf_cuda_kernel<<<blocks, threads, 0, stream>>>(
      pmfs.data_ptr<float>(), pmf_lengths.data_ptr<int>(),
      tail_masses.data_ptr<float>(), quantized_cdfs.data_ptr<int>(),
      pmfs.size(0), pmfs.size(1), cdf_max_length);

  return quantized_cdfs;
}
