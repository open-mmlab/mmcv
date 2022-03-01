/*************************************************************************
 * Copyright (C) 2021 Cambricon.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "gen_case.hpp"

#include <errno.h>
#include <error.h>
#include <string.h>
#include <sys/stat.h>
#include <iomanip>

#include "env_time.hpp"

// CNNL_GEN_CASE=1/2/3/ON/FILE/FILE_WITH_DATA/PRINT : Enable the gen_case module
#define IS_GEN_CASE                              \
  ((gen_case_mode_ > 0 && gen_case_mode_ < 4) || \
   (gen_case_env_ > 0 && gen_case_env_ < 4))

// CNNL_GEN_CASE=2/FILE_WITH_DATA : Dump the true values.
#define IS_DUMP_DATA                                                      \
  ((gen_case_mode_ == 2 && gen_case_env_ == 0) || (gen_case_env_ == 2) || \
   (dump_data_env_ && gen_case_env_ == 1))

// CNNL_GEN_CASE=3/PRINT : Only print message on screen.
#define IS_ONLY_SHOW                                                      \
  ((gen_case_mode_ == 3 && gen_case_env_ == 0) || (gen_case_env_ == 3) || \
   (only_show_env_ && gen_case_env_ == 1))

namespace cnnl {
namespace gen_case {

int gen_case_mode_ = 0;

// Get CNNL_GEN_CASE from env. CNNL_GEN_CASE has the highest priority.
// CNNL_GEN_CASE=ON/1/FILE:
//     Generate gen_case file without input data.
// CNNL_GEN_CASE=2/FILE_WITH_DATA:
//     Generate gen_case file with input data.
// CNNL_GEN_CASE=3/PRINT:
//     Print gen_case on screen.
__attribute__((__unused__)) int gen_case_env_ =
    getIntEnvVar("CNNL_GEN_CASE", 0);

// CNNL_GEN_CASE_OP_NAME=op_name : Only generate the designated op_name's
// prototxt.
__attribute__((__unused__)) std::string op_name_ =
    getStringEnvVar("CNNL_GEN_CASE_OP_NAME", "all");

// CNNL_GEN_CASE_DUMP_DATA=ON : Dump the true values.
__attribute__((__unused__)) bool dump_data_env_ =
    getBoolEnvVar("CNNL_GEN_CASE_DUMP_DATA", false);

// CNNL_GEN_CASE_ONLY_SHOW=ON : Only print message on screen.
__attribute__((__unused__)) bool only_show_env_ =
    getBoolEnvVar("CNNL_GEN_CASE_ONLY_SHOW", false);

// CNNL_GEN_CASE_ONLY_FOLDER=ON : Only generate op_name folder without prototxt.
__attribute__((__unused__)) bool is_only_folder_ =
    getBoolEnvVar("CNNL_GEN_CASE_ONLY_FOLDER", false);

inline bool isGenCaseOn() { return IS_GEN_CASE; }

void genCaseModeSet(const int mode) {
  if (mode > 0 && mode < 4) {
    gen_case_mode_ = mode;
    CNLOG(INFO) << "[gen_case] Set GEN_CASE mode to " << mode << ".";
  } else {
    gen_case_mode_ = 0;
    CNLOG(INFO) << "[gen_case] Set GEN_CASE mode to " << mode << ".";
  }
}

void genCaseModeSet(const std::string &mode) {
  if (mode == "1" || mode == "FILE" || mode == "ON") {
    gen_case_mode_ = 1;
    CNLOG(INFO) << "[gen_case] Set GEN_CASE mode to " << mode << ".";
  } else if (mode == "2" || mode == "FILE_WITH_DATA") {
    gen_case_mode_ = 2;
    CNLOG(INFO) << "[gen_case] Set GEN_CASE mode to " << mode << ".";
  } else if (mode == "3" || mode == "PRINT") {
    gen_case_mode_ = 3;
    CNLOG(INFO) << "[gen_case] Set GEN_CASE mode to " << mode << ".";
  } else {
    gen_case_mode_ = 0;
    CNLOG(INFO) << "[gen_case] Set GEN_CASE mode to " << mode << ".";
  }
}

float cvtHalfToFloat(const int16_t src) {
  if (sizeof(int16_t) == 2) {
    int re = src;
    float f = 0.;
    int sign = (re >> 15) ? (-1) : 1;
    int exp = (re >> 10) & 0x1f;
    int eff = re & 0x3ff;
    constexpr float half_max = 65504.;
    constexpr float half_min = -65504.;  // or to be defined as infinity
    if (exp == 0x1f && sign == 1) {
      // add upper bound of half. 0x7bff： 0 11110 1111111111 =  65504
      return half_max;
    } else if (exp == 0x1f && sign == -1) {
      // add lower bound of half. 0xfbff： 1 11110 1111111111 = -65504
      return half_min;
    }
    if (exp > 0) {
      exp -= 15;
      eff = eff | 0x400;
    } else {
      exp = -14;
    }
    int sft;
    sft = exp - 10;
    if (sft < 0) {
      f = (float)sign * eff / (1 << (-sft));
    } else {
      f = ((float)sign) * (1 << sft) * eff;
    }
    return f;
  } else if (sizeof(int16_t) == 4) {
    // using float
    return src;
  }
}

void saveDataToFile(const std::string &file_name, const void *data,
                    const at::ScalarType ScalarType, const int64_t count) {
  std::ofstream case_file;
  case_file.open(file_name.c_str(), std::ios::app);
  switch (ScalarType) {
    case at::kHalf: {
      for (int64_t i = 0; i < count; ++i) {
        case_file << "  value_f: " << std::dec
                  << cvtHalfToFloat(((int16_t *)data)[i]) << std::endl;
      }
    }; break;
    case at::kFloat: {
      for (int64_t i = 0; i < count; ++i) {
        case_file << "  value_f: " << std::dec << ((float *)data)[i]
                  << std::endl;
      }
    }; break;
    case at::kDouble: {
      for (int64_t i = 0; i < count; ++i) {
        case_file << "  value_l: " << std::dec << ((double *)data)[i]
                  << std::endl;
      }
    }; break;
    case at::kChar: {
      for (int64_t i = 0; i < count; ++i) {
        case_file << "  value_i: " << std::dec << (int32_t)((int8_t *)data)[i]
                  << std::endl;
      }
    }; break;
    case at::kByte: {
      for (int64_t i = 0; i < count; ++i) {
        case_file << "  value_ui: " << std::dec
                  << (uint32_t)((uint8_t *)data)[i] << std::endl;
      }
    }; break;
    case at::kShort: {
      for (int64_t i = 0; i < count; ++i) {
        case_file << "  value_i: " << std::dec << ((int16_t *)data)[i]
                  << std::endl;
      }
    }; break;
    case at::kInt: {
      for (int64_t i = 0; i < count; ++i) {
        case_file << "  value_i: " << std::dec << ((int32_t *)data)[i]
                  << std::endl;
      }
    }; break;
    case at::kLong: {
      for (int64_t i = 0; i < count; ++i) {
        case_file << "  value_l: " << std::dec << ((int64_t *)data)[i]
                  << std::endl;
      }
    }; break;
    default: { CNLOG(ERROR) << "[gen case]: unsupported data type"; }; break;
  }
}

std::string genCaseStart(const std::string &op_name,
                         const std::string &op_type) {
  if (!getBoolOpName(op_name, op_name_) || !IS_GEN_CASE) {
    return "NULL";
  }

  if (IS_ONLY_SHOW) {
    std::string file_name;
    file_name = "[tid" + std::to_string(syscall(SYS_gettid)) + "][gen_case][" +
                op_name + "] ";
    return file_name;
  }

  // Create folder name by op_name.
  char current_dir[PATH_MAX];
  if (getcwd(current_dir, sizeof(current_dir)) == NULL) {
    CNLOG(ERROR) << "[gen_case]: get current directory failed! (" << errno
                 << ": " << strerror(errno) << ")";
    return "NULL";
  }
  std::string folder_name = current_dir;
  folder_name = folder_name + "/gen_case/" + op_name;

  if (mkdirRecursive(folder_name.c_str()) != 0) {
    CNLOG(ERROR) << "[gen_case]: mkdir folder failed for " << folder_name
                 << " ! (" << errno << ": " << strerror(errno) << ")";
    return "NULL";
  }

  if (is_only_folder_) {
    return "NULL";
  }

  // Get current time for file name.
  static platform::EnvTime *env_time = platform::EnvTime::Default();
  uint64_t now_micros = env_time->NowMicros();
  int32_t micros_remainder = static_cast<int32_t>(now_micros % 1000000);
  time_t current_time = time(NULL);
  char char_current_time[64];
  strftime(char_current_time, sizeof(char_current_time), "%Y%m%d_%H_%M_%S_",
           localtime(&current_time));
  std::string string_current_time = char_current_time;
  std::string string_micros_remainder = std::to_string(micros_remainder);
  while (string_micros_remainder.size() < 6) {
    string_micros_remainder = "0" + string_micros_remainder;
  }

  // Get current device index.
  int dev_index = -1;
  cnrtGetDevice(&dev_index);

  // Create file name by op_name and current time.
  std::string file_name = folder_name + "/" + op_name + "_" +
                          string_current_time + string_micros_remainder +
                          "_tid" + std::to_string(syscall(SYS_gettid)) +
                          "_device" + std::to_string(dev_index) + ".prototxt";

  std::ofstream case_file;
  if (!case_file.is_open()) {
    case_file.open(file_name.c_str(), std::ios::app);
    if (case_file) {
      case_file << "op_name: \"" + op_name + "\"" << std::endl;
      case_file << "op_type: " + op_type << std::endl;
      CNLOG(INFO) << "[gen_case]: Generate " + file_name;
    }
    case_file.close();
  }
  return file_name;
}

// Check if tensor need stride process.
bool ifNeedTensorStrideProcess(const Tensor &tensor) {
  int stride_base = 1;
  if (tensor.stride(0) == -1) {
    return false;
  }
  for (int i = tensor.dim() - 1; i >= 0; i--) {
    if (tensor.stride(i) != stride_base) {
      return true;
    }
    stride_base *= tensor.size(i);
  }
  return false;
}

cnnlTensorLayout_t genCaseSuggestLayout(const Tensor &input) {
  auto suggest_memory_format = input.suggest_memory_format();
  cnnlTensorLayout_t layout = CNNL_LAYOUT_ARRAY;
  switch (input.dim()) {
    case 4: {
      layout = (suggest_memory_format == at::MemoryFormat::ChannelsLast)
                   ? CNNL_LAYOUT_NHWC
                   : CNNL_LAYOUT_NCHW;
      if (layout == CNNL_LAYOUT_NCHW && input.size(1) == 1) {
        layout = CNNL_LAYOUT_NHWC;
      }
    }; break;
    case 5: {
      layout = (suggest_memory_format == at::MemoryFormat::ChannelsLast3d)
                   ? CNNL_LAYOUT_NDHWC
                   : CNNL_LAYOUT_NCDHW;
      if (layout == CNNL_LAYOUT_NCDHW && input.size(1) == 1) {
        layout = CNNL_LAYOUT_NDHWC;
      }
    }; break;
    default: { layout = CNNL_LAYOUT_ARRAY; }; break;
  }
  return layout;
}

std::string dumpSizeAndStride(const Tensor &tensor) {
  auto layout = genCaseSuggestLayout(tensor);
  std::string dims_str = "";
  // Write the dims of shape module.
  if (layout == CNNL_LAYOUT_NHWC) {
    dims_str += ("    dims: " + std::to_string(tensor.size(0)) + '\n' +
                 "    dims: " + std::to_string(tensor.size(2)) + '\n' +
                 "    dims: " + std::to_string(tensor.size(3)) + '\n' +
                 "    dims: " + std::to_string(tensor.size(1)) + '\n');
  } else if (layout == CNNL_LAYOUT_NDHWC) {
    dims_str += ("    dims: " + std::to_string(tensor.size(0)) + '\n' +
                 "    dims: " + std::to_string(tensor.size(2)) + '\n' +
                 "    dims: " + std::to_string(tensor.size(3)) + '\n' +
                 "    dims: " + std::to_string(tensor.size(4)) + '\n' +
                 "    dims: " + std::to_string(tensor.size(1)) + '\n');
  } else {
    for (int i = 0; i < tensor.dim(); i++) {
      dims_str += ("    dims: " + std::to_string(tensor.size(i)) + '\n');
    }
  }

  if (tensor.numel() != 1) {
    if (ifNeedTensorStrideProcess(tensor)) {
      // Write the dim_stride of shape module.
      if (layout == CNNL_LAYOUT_NHWC) {
        dims_str +=
            ("    dim_stride: " + std::to_string(tensor.stride(0)) + '\n' +
             "    dim_stride: " + std::to_string(tensor.stride(2)) + '\n' +
             "    dim_stride: " + std::to_string(tensor.stride(3)) + '\n' +
             "    dim_stride: " + std::to_string(tensor.stride(1)) + '\n');
      } else if (layout == CNNL_LAYOUT_NDHWC) {
        dims_str +=
            ("    dim_stride: " + std::to_string(tensor.stride(0)) + '\n' +
             "    dim_stride: " + std::to_string(tensor.stride(2)) + '\n' +
             "    dim_stride: " + std::to_string(tensor.stride(3)) + '\n' +
             "    dim_stride: " + std::to_string(tensor.stride(4)) + '\n' +
             "    dim_stride: " + std::to_string(tensor.stride(1)) + '\n');
      } else {
        for (int i = 0; i < tensor.dim(); i++) {
          dims_str +=
              ("    dim_stride: " + std::to_string(tensor.stride(i)) + '\n');
        }
      }
    }
  }
  return dims_str;
}

void genCaseData(const bool op_dump_data, std::string *file_name,
                 const bool is_input, std::string id, const Tensor &tensor,
                 const float upper_bound, const float lower_bound) {
  if (!IS_GEN_CASE || *file_name == "NULL") {
    return;
  }

  // Get layout string
  std::string layout_string =
      convertLayout2String(genCaseSuggestLayout(tensor));

  // Get dtype string
  std::string dtype_string =
      convertDataType2String(torch_mlu::getCnnlDataType(tensor.dtype()));

  // Get onchip dtype string
  std::string onchip_dtype_string = "DTYPE_INVALID";

  // Get position scale and offset.
  int position = 0;
  float scale = 1;
  int offset = 0;

  // If IS_ONLY_SHOW == true, then print data message on screen.
  if (IS_ONLY_SHOW) {
    if (is_input) {
      *file_name = *file_name + "input{";
    } else {
      *file_name = *file_name + "output{";
    }
    if (tensor.data_ptr() == NULL) {
      *file_name = *file_name + "NULL} ";
      return;
    }
    *file_name = *file_name + id;
    *file_name = *file_name + "," + layout_string;
    *file_name = *file_name + "," + dtype_string;
    if (onchip_dtype_string != "DTYPE_INVALID") {
      *file_name = *file_name + "," + onchip_dtype_string;
    }
    *file_name = *file_name + ",[";
    for (int i = 0; i < tensor.dim(); i++) {
      if (i < tensor.dim() - 1) {
        *file_name = *file_name + std::to_string(tensor.size(i)) + ",";
      } else {
        *file_name = *file_name + std::to_string(tensor.size(i));
      }
    }
    *file_name = *file_name + "]} ";
    return;
  }

  std::ofstream case_file;
  if (!case_file.is_open()) {
    case_file.open((*file_name).c_str(), std::ios::app);
  }

  // Write the title of input/output module.
  if (is_input) {
    case_file << "input {" << std::endl;
  } else {
    if (tensor.data_ptr() == NULL) {
      return;
    }
    case_file << "output {" << std::endl;
  }

  // Write the id and the title of shape module.
  if (tensor.data_ptr() == NULL) {
    id = "NULL";
    case_file << "  id: \"" + id + "\"" << std::endl;
    case_file << "  shape: {" << std::endl;
    case_file << "    dims: 1" << std::endl;
    case_file << "  }" << std::endl << "  layout: LAYOUT_ARRAY" << std::endl;
    case_file << "  dtype: DTYPE_FLOAT" << std::endl;
  } else {
    case_file << "  id: \"" + id + "\"" << std::endl;
    case_file << "  shape: {" << std::endl;

    // Write the dims of shape module.
    case_file << dumpSizeAndStride(tensor);

    // Write the layout module.
    case_file << "  }" << std::endl
              << "  layout: " + layout_string << std::endl;

    // Write the dtype module.
    case_file << "  dtype: " + dtype_string << std::endl;

    // Write the onchip dtype module.
    if (onchip_dtype_string != "DTYPE_INVALID") {
      case_file << "  onchip_dtype: " + onchip_dtype_string << std::endl;
    }

    // Write the position, scale and offset module.
    case_file << "  position: " << position << std::endl;
    case_file << "  scale: " << scale << std::endl;
    case_file << "  offset: " << offset << std::endl;
  }

  bool need_dump_data = op_dump_data || IS_DUMP_DATA;
  if (is_input) {
    if (need_dump_data && tensor.numel() != 0) {
      // Write true values from device_data.
      saveDataToFile(*file_name, tensor.cpu().data_ptr(), tensor.scalar_type(),
                     tensor.numel());
    } else {
      // Write the random_data module.
      case_file << "  random_data: {" << std::endl;
      case_file << "    seed: 233" << std::endl;
      case_file << "    upper_bound: " << std::to_string(upper_bound)
                << std::endl;
      case_file << "    lower_bound: " << std::to_string(lower_bound)
                << std::endl;
      case_file << "    distribution: UNIFORM" << std::endl;
      case_file << "  }" << std::endl;
    }
  }
  case_file << "}" << std::endl;
  case_file.close();
}

void genCaseData(std::string *file_name, const bool is_input, std::string id,
                 const Tensor &tensor, const float upper_bound,
                 const float lower_bound, const std::string distribution) {
  if (!IS_GEN_CASE || *file_name == "NULL") {
    return;
  }

  // Get layout string
  std::string layout_string =
      convertLayout2String(genCaseSuggestLayout(tensor));

  // Get dtype string
  std::string dtype_string =
      convertDataType2String(torch_mlu::getCnnlDataType(tensor.dtype()));

  // Get onchip dtype string
  std::string onchip_dtype_string = "DTYPE_INVALID";

  // Get position scale and offset.
  int position = 0;
  float scale = 1;
  int offset = 0;

  // If IS_ONLY_SHOW == true, then print data message on screen.
  if (IS_ONLY_SHOW) {
    if (is_input) {
      *file_name = *file_name + "input{";
    } else {
      *file_name = *file_name + "output{";
    }
    if (tensor.data_ptr() == NULL) {
      *file_name = *file_name + "NULL} ";
      return;
    }
    *file_name = *file_name + id;
    *file_name = *file_name + "," + layout_string;
    *file_name = *file_name + "," + dtype_string;
    if (onchip_dtype_string != "DTYPE_INVALID") {
      *file_name = *file_name + "," + onchip_dtype_string;
    }
    *file_name = *file_name + ",[";
    for (int i = 0; i < tensor.dim(); i++) {
      if (i < tensor.dim() - 1) {
        *file_name = *file_name + std::to_string(tensor.size(i)) + ",";
      } else {
        *file_name = *file_name + std::to_string(tensor.size(i));
      }
    }
    *file_name = *file_name + "]} ";
    return;
  }

  std::ofstream case_file;
  if (!case_file.is_open()) {
    case_file.open((*file_name).c_str(), std::ios::app);
  }

  // Write the title of input/output module.
  int dim_total = tensor.numel();
  if (is_input) {
    case_file << "input {" << std::endl;
  } else {
    if (tensor.data_ptr() == NULL) {
      return;
    }
    case_file << "output {" << std::endl;
  }

  // Write the id and the title of shape module.
  if (tensor.data_ptr() == NULL) {
    id = "NULL";
    case_file << "  id: \"" + id + "\"" << std::endl;
    case_file << "  shape: {" << std::endl;
    case_file << "    dims: 1" << std::endl;
    case_file << "  }" << std::endl << "  layout: LAYOUT_ARRAY" << std::endl;
    case_file << "  dtype: DTYPE_FLOAT" << std::endl;
  } else {
    case_file << "  id: \"" + id + "\"" << std::endl;
    case_file << "  shape: {" << std::endl;

    // Write the dims of shape module.
    case_file << dumpSizeAndStride(tensor);

    // Write the layout module.
    case_file << "  }" << std::endl
              << "  layout: " + layout_string << std::endl;

    // Write the dtype module.
    case_file << "  dtype: " + dtype_string << std::endl;

    // Write the onchip dtype module.
    if (onchip_dtype_string != "DTYPE_INVALID") {
      case_file << "  onchip_dtype: " + onchip_dtype_string << std::endl;
    }

    // Write the position, scale and offset module.
    case_file << "  position: " << position << std::endl;
    case_file << "  scale: " << scale << std::endl;
    case_file << "  offset: " << offset << std::endl;
  }

  if (is_input) {
    // Write true values from device_data.
    if (IS_DUMP_DATA && tensor.data_ptr() != NULL) {
      saveDataToFile(*file_name, tensor.cpu().data_ptr(), tensor.scalar_type(),
                     tensor.numel());
    } else {
      // Write the random_data module.
      case_file << "  random_data: {" << std::endl;
      case_file << "    seed: 233" << std::endl;
      if (distribution == "UNIFORM") {
        case_file << "    upper_bound: " << std::to_string(upper_bound)
                  << std::endl;
        case_file << "    lower_bound: " << std::to_string(lower_bound)
                  << std::endl;
        case_file << "    distribution: UNIFORM" << std::endl;
      } else if (distribution == "GAUSSIAN") {
        case_file << "    mu: " << std::to_string(upper_bound) << std::endl;
        case_file << "    sigma: " << std::to_string(lower_bound) << std::endl;
        case_file << "    distribution: GAUSSIAN" << std::endl;
      } else {
        CNLOG(INFO) << "distribution only supports UNIFORM or GAUSSIAN, "
                       "distribution will use UNIFORM";
        case_file << "    upper_bound: " << std::to_string(upper_bound)
                  << std::endl;
        case_file << "    lower_bound: " << std::to_string(lower_bound)
                  << std::endl;
        case_file << "    distribution: UNIFORM" << std::endl;
      }
      case_file << "  }" << std::endl;
    }
  }
  case_file << "}" << std::endl;
  case_file.close();
}

void genCaseOpParam(const int flag, std::string *file_name,
                    const std::string &op_name, const std::string &param_name,
                    const int value) {
  if (!IS_GEN_CASE || *file_name == "NULL") {
    return;
  }

  // If IS_ONLY_SHOW == true, then print data message on screen.
  if (IS_ONLY_SHOW) {
    if (flag == 0 || flag == 3) {
      *file_name = *file_name + "params{";
      *file_name = *file_name + param_name + ":" + std::to_string(value);
    } else {
      *file_name = *file_name + "," + param_name + ":" + std::to_string(value);
    }
    if (flag == 2 || flag == 3) {
      *file_name = *file_name + "}";
    }
    return;
  }

  std::ofstream case_file;
  if (!case_file.is_open()) {
    case_file.open((*file_name).c_str(), std::ios::app);
  }

  // Write op_name_param.
  if (flag == 0 || flag == 3) {
    case_file << op_name + "_param: {" << std::endl;
  }

  // Write param with value.
  case_file << "  " + param_name + ": " << (int)value << std::endl;

  if (flag == 2 || flag == 3) {
    case_file << "}" << std::endl;
  }
  case_file.close();
}

void genCaseOpParam(const int flag, std::string *file_name,
                    const std::string &op_name, const std::string &param_name,
                    const float value) {
  if (!IS_GEN_CASE || *file_name == "NULL") {
    return;
  }

  // If IS_ONLY_SHOW == true, then print data message on screen.
  if (IS_ONLY_SHOW) {
    if (flag == 0 || flag == 3) {
      *file_name = *file_name + "params{";
      *file_name = *file_name + param_name + ":" + std::to_string(value);
    } else {
      *file_name = *file_name + "," + param_name + ":" + std::to_string(value);
    }
    if (flag == 2 || flag == 3) {
      *file_name = *file_name + "}";
    }
    return;
  }

  std::ofstream case_file;
  if (!case_file.is_open()) {
    case_file.open((*file_name).c_str(), std::ios::app);
  }

  // Write op_name_param.
  if (flag == 0 || flag == 3) {
    case_file << op_name + "_param: {" << std::endl;
  }

  // Write param with value.
  case_file << std::setprecision(8) << "  " + param_name + ": " << (float)value
            << std::endl;

  if (flag == 2 || flag == 3) {
    case_file << "}" << std::endl;
  }
  case_file.close();
}

void genCaseOpParam(const int flag, std::string *file_name,
                    const std::string &op_name, const std::string &param_name,
                    const std::string &value) {
  if (!IS_GEN_CASE || *file_name == "NULL") {
    return;
  }

  // If IS_ONLY_SHOW == true, then print data message on screen.
  if (IS_ONLY_SHOW) {
    if (flag == 0 || flag == 3) {
      *file_name = *file_name + "params{";
      *file_name = *file_name + param_name + ":" + value;
    } else {
      *file_name = *file_name + "," + param_name + ":" + value;
    }
    if (flag == 2 || flag == 3) {
      *file_name = *file_name + "}";
    }
    return;
  }

  std::ofstream case_file;
  if (!case_file.is_open()) {
    case_file.open((*file_name).c_str(), std::ios::app);
  }

  // Write op_name_param.
  if (flag == 0 || flag == 3) {
    case_file << op_name + "_param: {" << std::endl;
  }

  // Write param with value.
  case_file << "  " + param_name + ": " << value << std::endl;

  if (flag == 2 || flag == 3) {
    case_file << "}" << std::endl;
  }
  case_file.close();
}

void genCaseOpParam(const int flag, std::string *file_name,
                    const std::string &op_name, const std::string &param_name,
                    const int value, const bool is_subparam) {
  if (!IS_GEN_CASE || *file_name == "NULL") {
    return;
  }

  // If IS_ONLY_SHOW == true, then print data message on screen.
  if (IS_ONLY_SHOW) {
    if (flag == 0 || flag == 3) {
      *file_name = *file_name + "params{";
      *file_name = *file_name + param_name + ":" + std::to_string(value);
    } else {
      *file_name = *file_name + "," + param_name + ":" + std::to_string(value);
    }
    if (flag == 2 || flag == 3) {
      *file_name = *file_name + "}";
    }
    return;
  }

  std::ofstream case_file;
  if (!case_file.is_open()) {
    case_file.open((*file_name).c_str(), std::ios::app);
  }

  // Write op_name_param.
  if (flag == 0 || flag == 3) {
    if (is_subparam) {
      case_file << op_name + ": {" << std::endl;
    } else {
      case_file << op_name + "_param: {" << std::endl;
    }
  }

  // Write param with value.
  case_file << "  " + param_name + ": " << value << std::endl;

  if (flag == 2 || flag == 3) {
    case_file << "}" << std::endl;
  }
  case_file.close();
}

void genCaseOpParam(const int flag, std::string *file_name,
                    const std::string &op_name, const std::string &param_name,
                    const int *value, const int num) {
  if (!IS_GEN_CASE || *file_name == "NULL") {
    return;
  }

  // If IS_ONLY_SHOW == true, then print data message on screen.
  if (IS_ONLY_SHOW) {
    if (flag == 0 || flag == 3) {
      *file_name = *file_name + "params{";
      *file_name = *file_name + param_name + ":";
      *file_name = *file_name + "[";
      for (int i = 0; i < num; i++) {
        if (i < num - 1) {
          *file_name = *file_name + std::to_string(value[i]) + ",";
        } else {
          *file_name = *file_name + std::to_string(value[i]);
        }
      }
      *file_name = *file_name + "]";
    } else {
      *file_name = *file_name + "," + param_name + ":";
      *file_name = *file_name + "[";
      for (int i = 0; i < num; i++) {
        if (i < num - 1) {
          *file_name = *file_name + std::to_string(value[i]) + ",";
        } else {
          *file_name = *file_name + std::to_string(value[i]);
        }
      }
      *file_name = *file_name + "]";
    }
    if (flag == 2 || flag == 3) {
      *file_name = *file_name + "}";
    }
    return;
  }

  std::ofstream case_file;
  if (!case_file.is_open()) {
    case_file.open((*file_name).c_str(), std::ios::app);
  }

  // Write op_name_param.
  if (flag == 0 || flag == 3) {
    case_file << op_name + "_param: {" << std::endl;
  }

  // Write param with value.
  for (int i = 0; i < num; i++) {
    case_file << "  " + param_name + ": " << value[i] << std::endl;
  }

  if (flag == 2 || flag == 3) {
    case_file << "}" << std::endl;
  }
  case_file.close();
}

void genCaseOpParam(const int flag, std::string *file_name,
                    const std::string &op_name, const std::string &param_name,
                    const float *value, const int num) {
  if (!IS_GEN_CASE || *file_name == "NULL") {
    return;
  }

  // If IS_ONLY_SHOW == true, then print data message on screen.
  if (IS_ONLY_SHOW) {
    if (flag == 0 || flag == 3) {
      *file_name = *file_name + "params{";
      *file_name = *file_name + param_name + ":";
      *file_name = *file_name + "[";
      for (int i = 0; i < num; i++) {
        if (i < num - 1) {
          *file_name = *file_name + std::to_string(value[i]) + ",";
        } else {
          *file_name = *file_name + std::to_string(value[i]);
        }
      }
      *file_name = *file_name + "]";
    } else {
      *file_name = *file_name + "," + param_name + ":";
      *file_name = *file_name + "[";
      for (int i = 0; i < num; i++) {
        if (i < num - 1) {
          *file_name = *file_name + std::to_string(value[i]) + ",";
        } else {
          *file_name = *file_name + std::to_string(value[i]);
        }
      }
      *file_name = *file_name + "]";
    }
    if (flag == 2 || flag == 3) {
      *file_name = *file_name + "}";
    }
    return;
  }

  std::ofstream case_file;
  if (!case_file.is_open()) {
    case_file.open((*file_name).c_str(), std::ios::app);
  }

  // Write op_name_param.
  if (flag == 0 || flag == 3) {
    case_file << op_name + "_param: {" << std::endl;
  }

  // Write param with value.
  for (int i = 0; i < num; i++) {
    case_file << std::setprecision(8) << "  " + param_name + ": " << value[i]
              << std::endl;
  }

  if (flag == 2 || flag == 3) {
    case_file << "}" << std::endl;
  }
  case_file.close();
}

void genCaseTestParam(std::string *file_name, const bool is_diff1,
                      const bool is_diff2, const bool is_diff3,
                      const float diff1_threshold, const float diff2_threshold,
                      const float diff3_threshold) {
  if (!IS_GEN_CASE || *file_name == "NULL") {
    return;
  }

  // If IS_ONLY_SHOW == true, then print data message on screen.
  if (IS_ONLY_SHOW) {
    CNLOG(INFO) << *file_name;
    return;
  }

  std::ofstream case_file;
  if (!case_file.is_open()) {
    case_file.open((*file_name).c_str(), std::ios::app);
  }

  // Write test_param module.
  case_file << "test_param: {" << std::endl;

  if (is_diff1) {
    case_file << "  error_func: DIFF1" << std::endl;
  }
  if (is_diff2) {
    case_file << "  error_func: DIFF2" << std::endl;
  }
  if (is_diff3) {
    case_file << "  error_func: DIFF3" << std::endl;
  }

  if (is_diff1) {
    case_file << "  error_threshold: " << diff1_threshold << std::endl;
  }
  if (is_diff2) {
    case_file << "  error_threshold: " << diff2_threshold << std::endl;
  }
  if (is_diff3) {
    case_file << "  error_threshold: " << diff3_threshold << std::endl;
  }

  case_file << "  baseline_device: CPU" << std::endl;
  case_file << "}" << std::endl;

  case_file.close();
}

bool getBoolEnvVar(const std::string &str, const bool default_para) {
  const char *env_raw_ptr = std::getenv(str.c_str());
  if (env_raw_ptr == NULL) {
    return default_para;
  }
  std::string env_var = std::string(env_raw_ptr);
  std::transform(env_var.begin(), env_var.end(), env_var.begin(), ::toupper);
  return (env_var == "1" || env_var == "ON" || env_var == "YES" ||
          env_var == "TRUE");
}

int getIntEnvVar(const std::string &str, const int default_para) {
  const char *env_raw_ptr = std::getenv(str.c_str());
  if (env_raw_ptr == NULL) {
    return default_para;
  }
  std::string env_var = std::string(env_raw_ptr);
  std::transform(env_var.begin(), env_var.end(), env_var.begin(), ::toupper);
  if (env_var == "1" || env_var == "FILE" || env_var == "ON") {
    return 1;
  } else if (env_var == "2" || env_var == "FILE_WITH_DATA") {
    return 2;
  } else if (env_var == "3" || env_var == "PRINT") {
    return 3;
  } else {
    return 0;
  }
}

std::string getStringEnvVar(const std::string &str,
                            const std::string default_para) {
  const char *env_raw_ptr = std::getenv(str.c_str());
  if (env_raw_ptr == NULL) {
    return default_para;
  }
  std::string env_var = std::string(env_raw_ptr);
  return env_var;
}

bool getBoolOpName(const std::string &str, const std::string &op_name) {
  if (op_name.find(str) != std::string::npos ||
      op_name.find("all") != std::string::npos) {
    if (op_name.find("-" + str) != std::string::npos) {
      return false;
    } else {
      return true;
    }
  } else {
    return false;
  }
}

std::string convertDataType2String(const cnnlDataType_t dtype) {
  // Get dtype string
  std::string dtype_string = "DTYPE_INVALID";
  switch (dtype) {
    case CNNL_DTYPE_HALF: {
      dtype_string = "DTYPE_HALF";
    }; break;
    case CNNL_DTYPE_FLOAT: {
      dtype_string = "DTYPE_FLOAT";
    }; break;
    case CNNL_DTYPE_INT8: {
      dtype_string = "DTYPE_INT8";
    }; break;
    case CNNL_DTYPE_UINT8: {
      dtype_string = "DTYPE_UINT8";
    }; break;
    case CNNL_DTYPE_UINT16: {
      dtype_string = "DTYPE_INT16";
    }; break;
    case CNNL_DTYPE_INT31: {
      dtype_string = "DTYPE_INT31";
    }; break;
    case CNNL_DTYPE_INT32: {
      dtype_string = "DTYPE_INT32";
    }; break;
    case CNNL_DTYPE_INT64: {
      dtype_string = "DTYPE_INT64";
    }; break;
    case CNNL_DTYPE_BOOL: {
      dtype_string = "DTYPE_BOOL";
    }; break;
    default: { dtype_string = "DTYPE_INVALID"; }; break;
  }
  return dtype_string;
}

std::string convertLayout2String(const cnnlTensorLayout_t layout) {
  // Get layout string
  std::string layout_string = "LAYOUT_INVALID";
  switch (layout) {
    case CNNL_LAYOUT_NCHW: {
      layout_string = "LAYOUT_NCHW";
    }; break;
    case CNNL_LAYOUT_NHWC: {
      layout_string = "LAYOUT_NHWC";
    }; break;
    case CNNL_LAYOUT_HWCN: {
      layout_string = "LAYOUT_HWCN";
    }; break;
    case CNNL_LAYOUT_NDHWC: {
      layout_string = "LAYOUT_NDHWC";
    }; break;
    case CNNL_LAYOUT_NCDHW: {
      layout_string = "LAYOUT_NCDHW";
    }; break;
    case CNNL_LAYOUT_ARRAY: {
      layout_string = "LAYOUT_ARRAY";
    }; break;
    case CNNL_LAYOUT_TNC: {
      layout_string = "LAYOUT_TNC";
    }; break;
    case CNNL_LAYOUT_NTC: {
      layout_string = "LAYOUT_NTC";
    }; break;
    case CNNL_LAYOUT_NLC: {
      layout_string = "LAYOUT_NLC";
    }; break;
    case CNNL_LAYOUT_NC: {
      layout_string = "LAYOUT_NC";
    }; break;
    default: { layout_string = "LAYOUT_INVALID"; }; break;
  }
  return layout_string;
}

int mkdirRecursive(const char *pathname) {
  auto mkdirIfNotExist = [](const char *pathname) -> int {
    struct stat dir_stat = {};
    if (stat(pathname, &dir_stat) != 0) {
      if (mkdir(pathname, 0777) != 0) {
        return errno;
      }
      return 0;
    } else if (!S_ISDIR(dir_stat.st_mode)) {
      return ENOTDIR;
    }
    return 0;
  };

  // ensure pathname is not null
  const char path_token = '/';
  size_t pos = 0;
  const std::string pathname_view(pathname);
  while (pos < pathname_view.size()) {
    auto find_path_token = pathname_view.find(path_token, pos);
    if (std::string::npos == find_path_token) {
      return mkdirIfNotExist(pathname_view.c_str());
    }
    int ret =
        mkdirIfNotExist(pathname_view.substr(0, find_path_token + 1).c_str());
    if (ret) return ret;
    pos = find_path_token + 1;
  }
  return 0;
}

}  // namespace gen_case
}  // namespace cnnl
// void cnnlSetGenCaseMode(int mode) { cnnl::gen_case::genCaseModeSet(mode); }
