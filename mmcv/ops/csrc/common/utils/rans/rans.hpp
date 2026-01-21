// Copyright(c) OpenMMLab.All rights reserved.
#ifndef RANS_HPP
#define RANS_HPP

#include <cstdint>
#include <thread>

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#include "utils/rans/rans64.h"

// ------------------- RANS CONSTANTS -------------------
/* probability range, this could be a parameter... */
constexpr int precision = 16;

constexpr uint16_t bypass_precision = 4; /* number of bits in bypass mode */
constexpr uint16_t max_bypass_val = (1 << bypass_precision) - 1;

struct RansSymbol {
  uint16_t start;
  uint16_t range;
  bool bypass;  // bypass flag to write raw bits to the stream
};

// ------------------- UTILS -------------------
#define TENSOR_TO_VECTOR_1D(tensor, type)    \
  std::vector<type>(tensor.data_ptr<type>(), \
                    tensor.data_ptr<type>() + tensor.numel())

// ------------------- RANS ENCODING -------------------
// vector would be optimized by compiler more than 10x faster than Tensor
inline void rans_encode_with_indexes_cpu_kernel(
    const std::vector<int32_t> &symbols, const std::vector<int32_t> &indexes,
    const std::vector<std::vector<int32_t>> &cdfs,
    const std::vector<int32_t> &cdfs_sizes, const std::vector<int32_t> &offsets,
    const uint32_t begin_idx, const uint32_t end_idx,
    std::vector<uint32_t> &output, uint32_t &nbyte) {
  std::vector<RansSymbol> _syms;
  assert(cdfs.size() == cdfs_sizes.size());

  // backward loop on symbols from the end;
  for (size_t i = begin_idx; i < end_idx; ++i) {
    const int32_t cdf_idx = indexes[i];
    assert(cdf_idx >= 0);
    assert(cdf_idx < cdfs.size());

    const auto &cdf = cdfs[cdf_idx];

    const int32_t max_value = cdfs_sizes[cdf_idx] - 2;
    assert(max_value >= 0);
    assert((max_value + 1) < cdf.size());

    int32_t value = symbols[i] - offsets[cdf_idx];

    uint32_t raw_val = 0;
    if (value < 0) {
      raw_val = -2 * value - 1;
      value = max_value;
    } else if (value >= max_value) {
      raw_val = 2 * (value - max_value);
      value = max_value;
    }

    assert(value >= 0);
    assert(value < cdfs_sizes[cdf_idx] - 1);

    _syms.push_back({static_cast<uint16_t>(cdf[value]),
                     static_cast<uint16_t>(cdf[value + 1] - cdf[value]),
                     false});

    /* Bypass coding mode (value == max_value -> sentinel flag) */
    if (value == max_value) {
      /* Determine the number of bypasses (in bypass_precision size) needed to
       * encode the raw value. */
      int32_t n_bypass = 0;
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
    }
  }

  Rans64State rans;
  Rans64EncInit(&rans);

  uint32_t *ptr = output.data() + output.size();
  assert(ptr != nullptr);

  while (!_syms.empty()) {
    const RansSymbol sym = _syms.back();

    if (!sym.bypass) {
      Rans64EncPut(&rans, &ptr, sym.start, sym.range, precision);
    } else {
      // unlikely...
      Rans64EncPutBits(&rans, &ptr, sym.start, bypass_precision);
    }
    _syms.pop_back();
  }

  Rans64EncFlush(&rans, &ptr);

  nbyte = std::distance(ptr, output.data() + output.size()) * sizeof(uint32_t);
}

// ------------------- RANS DECODING -------------------
inline void rans_decode_with_indexes_cpu_kernel(
    uint32_t *ptr, const std::vector<int32_t> &indexes,
    const std::vector<std::vector<int32_t>> &cdfs,
    const std::vector<int32_t> &cdfs_sizes, const std::vector<int32_t> &offsets,
    const int begin_idx, const int end_idx, Tensor output) {
  // use accessor to accelerate the access
  auto output_accessor = output.accessor<int32_t, 1>();
  Rans64State rans;
  assert(ptr != nullptr);
  Rans64DecInit(&rans, &ptr);

  for (int i = begin_idx; i < end_idx; ++i) {
    const int32_t cdf_idx = indexes[i];
    assert(cdf_idx >= 0);
    assert(cdf_idx < cdfs.size());

    const auto &cdf = cdfs[cdf_idx];

    const int32_t max_value = cdfs_sizes[cdf_idx] - 2;
    assert(max_value >= 0);
    assert((max_value + 1) < cdf.size());

    const int32_t offset = offsets[cdf_idx];

    const uint32_t cum_freq = Rans64DecGet(&rans, precision);

    const auto cdf_end = cdf.begin() + cdfs_sizes[cdf_idx];
    const auto it = std::find_if(cdf.begin(), cdf_end,
                                 [cum_freq](int v) { return v > cum_freq; });
    assert(it != cdf_end + 1);
    const uint32_t s = std::distance(cdf.begin(), it) - 1;

    Rans64DecAdvance(&rans, &ptr, cdf[s], cdf[s + 1] - cdf[s], precision);

    int32_t value = static_cast<int32_t>(s);

    if (value == max_value) {
      /* Bypass decoding mode */
      int32_t val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
      int32_t n_bypass = val;

      while (val == max_bypass_val) {
        val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
        n_bypass += val;
      }

      int32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
        assert(val <= max_bypass_val);
        raw_val |= val << (j * bypass_precision);
      }
      value = raw_val >> 1;
      if (raw_val & 1) {
        value = -value - 1;
      } else {
        value += max_value;
      }
    }
    output_accessor[i] = value + offset;
  }
}

// ------------------- PMF TO QUANTIZED CDF -------------------
inline void pmf_to_quantized_cdf_cpu_kernel(const Tensor pmfs,
                                            const Tensor pmf_lengths,
                                            const Tensor tail_masses,
                                            const Tensor quantized_cdfs) {
  /* NOTE(begaintj): ported from `ryg_rans` public implementation. Not optimal
   * although it's only run once per model after training. See TF/compression
   * implementation for an optimized version. */
  const int num_pmf = pmfs.size(0);
  for (int m = 0; m < num_pmf; ++m) {
    const int pmf_length = pmf_lengths[m].item<int>();
    const float tail_mass = tail_masses[m].item<float>();

    const float *pmf_begin = pmfs[m].data_ptr<float>();
    const float *pmf_end = pmf_begin + pmf_length;

    int *quantized_cdf_begin = quantized_cdfs[m].data_ptr<int>();
    int *quantized_cdf_end = quantized_cdf_begin + pmf_length + 2;

    quantized_cdf_begin[0] = 0; /* freq 0 */
    quantized_cdf_end[-1] =
        std::round(tail_mass * (1 << precision)); /* freq of tail mass */

    std::transform(pmf_begin, pmf_end, quantized_cdf_begin + 1,
                   [=](float p) { return std::round(p * (1 << precision)); });

    const uint32_t total =
        std::accumulate(quantized_cdf_begin, quantized_cdf_end, 0);
    if (total == 0) {
      throw std::domain_error(
          "Invalid `pmf`: at least one element must have a "
          "non-zero probability.");
    }

    std::transform(
        quantized_cdf_begin, quantized_cdf_end, quantized_cdf_begin,
        [=](uint32_t p) {
          return ((static_cast<uint64_t>(1 << precision) * p) / total);
        });

    std::partial_sum(quantized_cdf_begin, quantized_cdf_end,
                     quantized_cdf_begin);
    quantized_cdf_end[-1] = 1 << precision;

    for (int i = 0; i < static_cast<int>(pmf_length + 1); ++i) {
      if (quantized_cdf_begin[i] == quantized_cdf_begin[i + 1]) {
        /* Try to steal frequency from low-frequency symbols */
        uint32_t best_freq = ~0u;
        int best_steal = -1;
        for (int j = 0; j < static_cast<int>(pmf_length + 2) - 1; ++j) {
          uint32_t freq = quantized_cdf_begin[j + 1] - quantized_cdf_begin[j];
          if (freq > 1 && freq < best_freq) {
            best_freq = freq;
            best_steal = j;
          }
        }

        assert(best_steal != -1);

        if (best_steal < i) {
          for (int j = best_steal + 1; j <= i; ++j) {
            quantized_cdf_begin[j]--;
          }
        } else {
          assert(best_steal > i);
          for (int j = i + 1; j <= best_steal; ++j) {
            quantized_cdf_begin[j]++;
          }
        }
      }
    }

    assert(quantized_cdf_begin[0] == 0);
    assert(quantized_cdf_end[-1] == (1 << precision));
    for (int i = 0; i < static_cast<int>(pmf_length + 2) - 1; ++i) {
      assert(quantized_cdf_begin[i + 1] > quantized_cdf_begin[i]);
    }
  }
}

// ------------------- MULTITHREADING -------------------

#define COMPRESS_HEADER_BYTES 2
#define OTHER_HEADER_BITS_NUM 3
#define MAX_NUM_WITH_BITS(bits) (std::pow(2, (bits)) - 1)
#define MAX_NUM_THREADS (MAX_NUM_WITH_BITS(16 - OTHER_HEADER_BITS_NUM))
#define GET_BEGIN_IDX(i, total_length, num_threads) \
  (i * total_length / num_threads)
#define GET_END_IDX(i, total_length, num_threads) \
  (i == num_threads - 1) ? total_length           \
                         : GET_BEGIN_IDX((i + 1), total_length, num_threads)

enum class NBytesMode { OneByte = 0, TwoBytes = 1, FourBytes = 2 };
struct RansHeader {
  uint16_t num_threads;
  bool encode_with_cpu;
  NBytesMode nbytes_mode;
};

inline void check_num_threads(int num_threads) {
  int cpu_max_threads = std::thread::hardware_concurrency();
  TORCH_CHECK(num_threads > 0 && num_threads <= cpu_max_threads,
              "num_threads must be greater than 0 and less than or equal to ",
              cpu_max_threads);
}

inline std::string compress_rans_header(const RansHeader &header) {
  uint16_t compressed_header = 0;
  compressed_header |=
      header.num_threads
      << 3;  // left shift 3 bits, store num_threads in bits 3-15
  compressed_header |=
      header.encode_with_cpu
      << 2;  // left shift 2 bits, store encode_with_cpu in bit 2
  compressed_header |= static_cast<decltype(compressed_header)>(
      header.nbytes_mode);  // left shift 0 bits, store nbytes_mode in bits 0-1
  return std::string((char *)&compressed_header, sizeof(compressed_header));
}

inline RansHeader decompress_rans_header(std::string compressed_string) {
  uint16_t compressed_header = *(uint16_t *)compressed_string.data();
  RansHeader header;
  header.num_threads = (compressed_header >> 3) &
                       0x1FFF;  // right shift 3 bits, then take bits 3-15
  header.encode_with_cpu =
      (compressed_header >> 2) & 0x1;  // right shift 2 bits, then take bit 2
  header.nbytes_mode = static_cast<decltype(header.nbytes_mode)>(
      compressed_header & 0x3);  // right shift 0 bits, then take bits 0-1
  return header;
}

inline std::string compress_nbytes(uint32_t *nbytes, NBytesMode nbytes_mode,
                                   int num_nbytes) {
  std::string compressed;
  if (num_nbytes == 1) return compressed;  // no need to compress

  if (nbytes_mode == NBytesMode::OneByte) {
    // Use uint8_t for compression
    std::vector<uint8_t> compressedData(num_nbytes);
    for (int i = 0; i < num_nbytes; ++i) {
      compressedData[i] = static_cast<uint8_t>(nbytes[i]);
    }
    compressed.resize(num_nbytes);
    memcpy(&compressed[0], &compressedData[0], num_nbytes);
  } else if (nbytes_mode == NBytesMode::TwoBytes) {
    // Use uint16_t for compression
    std::vector<uint16_t> compressedData(num_nbytes);
    for (int i = 0; i < num_nbytes; ++i) {
      compressedData[i] = static_cast<uint16_t>(nbytes[i]);
    }
    compressed.resize(num_nbytes * 2);
    memcpy(&compressed[0], &compressedData[0], num_nbytes * 2);
  } else {
    // Use uint32_t, no need for compression
    compressed.resize(num_nbytes * 4);
    memcpy(&compressed[0], nbytes, num_nbytes * 4);
  }

  return compressed;
}

inline void decompress_nbytes(const std::string &compressed, uint32_t *nbytes,
                              int num_nbytes, NBytesMode nbytes_mode) {
  if (num_nbytes == 1) return;  // no need to decompress

  if (nbytes_mode == NBytesMode::OneByte) {
    // Use uint8_t for decompression
    std::vector<uint8_t> decompressedData(num_nbytes);
    memcpy(&decompressedData[0], &compressed[0], num_nbytes);

    for (int i = 0; i < num_nbytes; ++i) {
      nbytes[i] = static_cast<uint32_t>(decompressedData[i]);
    }
  } else if (nbytes_mode == NBytesMode::TwoBytes) {
    // Use uint16_t for decompression
    std::vector<uint16_t> decompressedData(num_nbytes);
    memcpy(&decompressedData[0], &compressed[0], num_nbytes * 2);

    for (int i = 0; i < num_nbytes; ++i) {
      nbytes[i] = static_cast<uint32_t>(decompressedData[i]);
    }
  } else {
    // Use uint32_t, no need for decompression
    memcpy(nbytes, &compressed[0], num_nbytes * 4);
  }
}

inline NBytesMode get_nbytes_mode(uint32_t max_nbyte) {
  if (max_nbyte <= UINT8_MAX) {
    return NBytesMode::OneByte;
  } else if (max_nbyte <= UINT16_MAX) {
    return NBytesMode::TwoBytes;
  } else {
    return NBytesMode::FourBytes;
  }
}

inline int get_nbytes_size(NBytesMode nbytes_mode) {
  if (nbytes_mode == NBytesMode::OneByte) {
    return 1;
  } else if (nbytes_mode == NBytesMode::TwoBytes) {
    return 2;
  } else {
    return 4;
  }
}

// ------------------- CHECK INPUTS -------------------
inline void check_rans_encode_input(const Tensor &symbols,
                                    const Tensor &indexes, const Tensor &cdfs,
                                    const Tensor &cdfs_sizes,
                                    const Tensor &offsets, int num_threads) {
  // Check if the number of threads is valid
  check_num_threads(num_threads);

  // Check if the tensors are on the same device
  TORCH_CHECK(symbols.device() == indexes.device() &&
                  symbols.device() == cdfs.device() &&
                  symbols.device() == cdfs_sizes.device() &&
                  symbols.device() == offsets.device(),
              "All tensors must be on the same device.");

  // Check if the tensors have the correct dtype
  TORCH_CHECK(symbols.scalar_type() == ScalarType::Int,
              "symbols must be a tensor of integers");
  TORCH_CHECK(indexes.scalar_type() == ScalarType::Int,
              "indexes must be a tensor of integers");
  TORCH_CHECK(cdfs.scalar_type() == ScalarType::Int,
              "cdfs must be a tensor of integers");
  TORCH_CHECK(cdfs_sizes.scalar_type() == ScalarType::Int,
              "cdfs_sizes must be a tensor of integers");
  TORCH_CHECK(offsets.scalar_type() == ScalarType::Int,
              "offsets must be a tensor of integers");
  TORCH_CHECK(cdfs.size(0) == cdfs_sizes.size(0),
              "Size mismatch between cdfs and cdfs_sizes: cdfs.size(0) = ",
              cdfs.size(0), " while cdfs_sizes.size(0) = ", cdfs_sizes.size(0));

  // Check if the cdfs tensor is continuous
  TORCH_CHECK(symbols.is_contiguous(), "symbols tensor must be contiguous");
  TORCH_CHECK(indexes.is_contiguous(), "indexes tensor must be contiguous");
  TORCH_CHECK(cdfs.is_contiguous(), "cdfs tensor must be contiguous");
  TORCH_CHECK(cdfs_sizes.is_contiguous(),
              "cdfs_sizes tensor must be contiguous");
  TORCH_CHECK(offsets.is_contiguous(), "offsets tensor must be contiguous");

  // Check if the tensors have the correct shape
  TORCH_CHECK(symbols.dim() == 1 && indexes.dim() == 1 &&
                  cdfs_sizes.dim() == 1 && offsets.dim() == 1 &&
                  cdfs.dim() == 2,
              "The tensors must have the following shapes: "
              "symbols (N), indexes (N), cdfs_sizes (M), offsets (M), "
              "cdfs (M x K).");
}

inline void check_rans_decode_input(const std::string &encoded,
                                    const Tensor &indexes, const Tensor &cdfs,
                                    const Tensor &cdfs_sizes,
                                    const Tensor &offsets) {
  // Check if the tensors are on the same device
  TORCH_CHECK(indexes.device() == cdfs.device() &&
                  indexes.device() == cdfs_sizes.device() &&
                  indexes.device() == offsets.device(),
              "All tensors must be on the same device.");

  // Check if the tensors have the correct dtype
  TORCH_CHECK(indexes.scalar_type() == ScalarType::Int,
              "indexes must be a tensor of integers");
  TORCH_CHECK(cdfs.scalar_type() == ScalarType::Int,
              "cdfs must be a tensor of integers");
  TORCH_CHECK(cdfs_sizes.scalar_type() == ScalarType::Int,
              "cdfs_sizes must be a tensor of integers");
  TORCH_CHECK(offsets.scalar_type() == ScalarType::Int,
              "offsets must be a tensor of integers");
  TORCH_CHECK(cdfs.size(0) == cdfs_sizes.size(0),
              "Size mismatch between cdfs and cdfs_sizes: cdfs.size(0) = ",
              cdfs.size(0), " while cdfs_sizes.size(0) = ", cdfs_sizes.size(0));

  // Check if the cdfs tensor is continuous
  TORCH_CHECK(indexes.is_contiguous(), "indexes tensor must be contiguous");
  TORCH_CHECK(cdfs.is_contiguous(), "cdfs tensor must be contiguous");
  TORCH_CHECK(cdfs_sizes.is_contiguous(),
              "cdfs_sizes tensor must be contiguous");
  TORCH_CHECK(offsets.is_contiguous(), "offsets tensor must be contiguous");

  // Check if the tensors have the correct shape
  TORCH_CHECK(cdfs_sizes.dim() == 1 && offsets.dim() == 1 && cdfs.dim() == 2,
              "The tensors must have the following shapes: "
              "indexes (N), cdfs_sizes (M), offsets (M), "
              "cdfs (M x K).");
}

inline void check_pmf_to_quantized_cdf_input(const Tensor &pmfs,
                                             const Tensor &pmf_lengths,
                                             const Tensor &tail_masses) {
  // Check all pmfs larger than 0
  TORCH_CHECK(pmfs.min().item().toFloat() >= 0,
              "All pmfs must be larger than 0");

  // Check if the tensors are on the same device
  TORCH_CHECK(pmfs.device() == pmf_lengths.device() &&
                  pmfs.device() == tail_masses.device(),
              "All tensors must be on the same device.");

  // Check if the tensors have the correct dtype
  TORCH_CHECK(pmfs.scalar_type() == ScalarType::Float,
              "pmfs must be a tensor of floats");
  TORCH_CHECK(pmf_lengths.scalar_type() == ScalarType::Int,
              "pmf_lengths must be a tensor of integers");
  TORCH_CHECK(tail_masses.scalar_type() == ScalarType::Float,
              "tail_masses must be a tensor of floats");

  // Check if the cdfs tensor is continuous
  TORCH_CHECK(pmfs.is_contiguous(), "pmfs tensor must be contiguous");
  TORCH_CHECK(pmf_lengths.is_contiguous(),
              "pmf_lengths tensor must be contiguous");
  TORCH_CHECK(tail_masses.is_contiguous(),
              "tail_masses tensor must be contiguous");

  // Check if the tensors have the correct shape
  TORCH_CHECK(
      pmfs.dim() == 2 && pmf_lengths.dim() == 1 && tail_masses.dim() == 1,
      "The tensors must have the following shapes: "
      "pmfs (N x M), pmf_lengths (N), tail_masses (N).");
}

#endif  // RANS_HPP
