#ifndef MMCV_OPS_CSRC_COMMON__UTIL_HPP_
#define MMCV_OPS_CSRC_COMMON__UTIL_HPP_
const int SIZE = 8;

c10::SmallVector<int64_t, SIZE> array_to_vector(c10::IntArrayRef shape) {
    c10::SmallVector<int64_t, SIZE> shape_small_vec;
    for (uint64_t i = 0; i < shape.size(); i++) {
        shape_small_vec.emplace_back(shape[i]);
    }

    return shape_small_vec;
}

#endif  // MMCV_OPS_CSRC_COMMON__UTIL_HPP_
