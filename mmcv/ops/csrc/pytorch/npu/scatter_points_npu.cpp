#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;

typedef enum { SUM = 0, MEAN = 1, MAX = 2} reduce_t;

static std::map<int64_t, std::string> REDUCE_TYPE_MAP = {{0, "sum"},
                                                         {1, "mean"},
                                                         {2, "max"}};

inline void npu_dynamic_scatter_check(const at::Tensor &coors,
                                      reduce_t reduce_type) {
  TORCH_CHECK(reduce_type == 0 || reduce_type == 1 || reduce_type == 2,
              "reduce_type must be 0(sum) or 1(mean) or 2(max).");
  TORCH_CHECK(coors.size(1) == 3,
              "npu_dynamic_scatter only support coors.size(1) == 3.");
}

std::tuple<at::Tensor, at::Tensor> get_hash_key_and_coefficient(
    const at::Tensor &coors) {
  auto coors_dtype = coors.dtype();
  auto coors_dim = coors.size(1);
  auto coors_max =
      std::get<0>(at::max(coors.to(at::kLong), 0, false)).to(coors_dtype);
  coors_max = at::add(coors_max, 
                      at::ones(coors_dim,
                               coors_max.options().dtype(coors_dtype)),
                      1);
  auto cof_tensor = at::ones({1}, coors_max.options().dtype(coors_dtype));
  auto tmp = cof_tensor;
  for (auto i = coors_dim - 1; i > 0; i--) {
    tmp = at::mul(coors_max[i], tmp);
    cof_tensor = at::cat({tmp, cof_tensor}, 0);
  }
  cof_tensor = cof_tensor.reshape({1, coors_dim});

  auto coors_clean = coors.masked_fill(coors.lt(0).any(-1, true), -1);
  auto cof_mul = at::mul(coors_clean, cof_tensor);
  auto hash_key = at::sum(cof_mul, 1);
  return {hash_key, cof_tensor};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_dim_simulation(
    const at::Tensor &coors) {
  at::Tensor out_coors_unique2;
  at::Tensor coors_map;
  at::Tensor reduce_count;
  at::Tensor hash_key;
  at::Tensor cof_tensor;
  at::Tensor out_coors;
  std::tie(hash_key, cof_tensor) = get_hash_key_and_coefficient(coors);
  std::tie(out_coors_unique2, coors_map, reduce_count) =
      at::_unique2(hash_key, true, true, true);

  c10::optional<c10::string_view> rounding_mode = "trunc";
  std::vector<at::Tensor> out_coors_tensors;
  for (auto i = 0; i < cof_tensor.numel() - 1; i++) {
    auto out_coors_0 =
        at::div(out_coors_unique2, cof_tensor[0][i], rounding_mode);
    out_coors_unique2 =
        at::sub(out_coors_unique2, at::mul(out_coors_0,
                                           cof_tensor[0][i]));
    out_coors_tensors.push_back(out_coors_0);
  }
  out_coors_tensors.push_back(out_coors_unique2);
  out_coors = at::stack(at::TensorList(out_coors_tensors), 1);
  out_coors = out_coors.to(coors.dtype());
  return {out_coors, coors_map, reduce_count};
}

std::vector<torch::Tensor> dynamic_point_to_voxel_forward_npu(
    const at::Tensor &feats,
    const at::Tensor &coors,
    reduce_t reduce_type) {
  npu_dynamic_scatter_check(coors, reduce_type);
  auto num_input = feats.size(0);
  auto num_feats = feats.size(1);
  if (num_input == 0) {
    return {feats.clone().detach(), coors.clone().detach(),
            coors.new_empty({0}, at::kInt),
            coors.new_empty({0}, at::kInt)};
  }

  at::Tensor out_coors;
  at::Tensor coors_map;
  at::Tensor reduce_count;
  std::tie(out_coors, coors_map, reduce_count) =
      unique_dim_simulation(coors);

  coors_map = coors_map.to(at::kInt);
  reduce_count = reduce_count.to(at::kInt);


  if (out_coors[0][0].lt(0).item<bool>()) {
    out_coors = out_coors.slice(0, 1);
    reduce_count = reduce_count.slice(0, 1);
    coors_map = coors_map - 1;
  }
    
  auto reduced_feats = at::empty({out_coors.size(0), num_feats},
                                  feats.options());
  const char *reduce_type_string =
      const_cast<char *>(REDUCE_TYPE_MAP[reduce_type] == "max" ? "max" : "sum");

  EXEC_NPU_CMD(aclnnDynamicScatter, feats, coors_map, reduce_type_string,
               reduced_feats);

  if (reduce_type == 1) {
    reduced_feats /= reduce_count.unsqueeze(-1).to(reduced_feats.dtype());
  }

  return {reduced_feats, out_coors, coors_map, reduce_count};
}

std::vector<torch::Tensor> dynamic_point_to_voxel_forward_impl(
    const torch::Tensor& feats, const torch::Tensor& coors,
    const reduce_t reduce_type);

REGISTER_NPU_IMPL(dynamic_point_to_voxel_forward_impl,
                  dynamic_point_to_voxel_forward_npu);
