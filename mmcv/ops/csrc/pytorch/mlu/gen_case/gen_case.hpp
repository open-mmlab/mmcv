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
#ifndef GEN_CASE_HPP_
#define GEN_CASE_HPP_

#include <limits.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "pytorch_mlu_helper.hpp"

#define CNNL_GEN_CASE_ON cnnl::gen_case::isGenCaseOn()

#define GEN_CASE_START(op_name, op_type) \
  std::string gen_case_file_name =       \
      cnnl::gen_case::genCaseStart(op_name, op_type)

#define GEN_CASE_DATA(is_input, id, tensor, upper_bound, lower_bound)   \
  cnnl::gen_case::genCaseData(false, &gen_case_file_name, is_input, id, \
                              tensor, upper_bound, lower_bound)

#define GEN_CASE_DATA_REAL(is_input, id, tensor)                               \
  cnnl::gen_case::genCaseData(true, &gen_case_file_name, is_input, id, tensor, \
                              0, 0)

// when distribution is "GAUSSIAN", upper_bound indicates mu, lower_bound
// indicates sigma.
#define GEN_CASE_DATA_v2(is_input, id, tensor, upper_bound, lower_bound, \
                         distribution)                                   \
  cnnl::gen_case::genCaseData(&gen_case_file_name, is_input, id, tensor, \
                              upper_bound, lower_bound, distribution)

#define GEN_CASE_OP_PARAM_SINGLE(flag, op_name, param_name, value)   \
  cnnl::gen_case::genCaseOpParam(flag, &gen_case_file_name, op_name, \
                                 param_name, value)

#define GEN_CASE_OP_PARAM_SINGLE_SUB(flag, op_name, param_name, value, is_sub) \
  cnnl::gen_case::genCaseOpParam(flag, &gen_case_file_name, op_name,           \
                                 param_name, value, is_sub)

#define GEN_CASE_OP_PARAM_ARRAY(flag, op_name, param_name, value, num) \
  cnnl::gen_case::genCaseOpParam(flag, &gen_case_file_name, op_name,   \
                                 param_name, value, num)

#define GEN_CASE_TEST_PARAM(is_diff1, is_diff2, is_diff3, diff1_threshold,     \
                            diff2_threshold, diff3_threshold)                  \
  cnnl::gen_case::genCaseTestParam(&gen_case_file_name, is_diff1, is_diff2,    \
                                   is_diff3, diff1_threshold, diff2_threshold, \
                                   diff3_threshold)

namespace cnnl {
namespace gen_case {

inline bool isGenCaseOn();
void genCaseModeSet(const int mode);
void genCaseModeSet(const std::string &mode);
float cvtHalfToFloat(const int16_t src);
void saveDataToFile(const std::string &gen_case_file_name, const void *data,
                    const at::ScalarType ScalarType, const int64_t count);
std::string genCaseStart(const std::string &op_name,
                         const std::string &op_type);
std::string convertLayout2String(const cnnlTensorLayout_t);
std::string convertDataType2String(const cnnlDataType_t dtype);
void genCaseData(const bool op_dump_data, std::string *file_name,
                 const bool is_input, std::string id, const Tensor &tensor,
                 const float upper_bound, const float lower_bound);
void genCaseData(std::string *file_name, const bool is_input, std::string id,
                 const Tensor &tensor, const float upper_bound,
                 const float lower_bound, const std::string distribution);
void genCaseOpParam(const int flag, std::string *file_name,
                    const std::string &op_name, const std::string &param_name,
                    const int value);
void genCaseOpParam(const int flag, std::string *gen_case_file_name,
                    const std::string &op_name, const std::string &param_name,
                    const float value);
void genCaseOpParam(const int flag, std::string *gen_case_file_name,
                    const std::string &op_name, const std::string &param_name,
                    const std::string &value);
void genCaseOpParam(const int flag, std::string *gen_case_file_name,
                    const std::string &op_name, const std::string &param_name,
                    const int value, const bool is_subparam);
void genCaseOpParam(const int flag, std::string *gen_case_file_name,
                    const std::string &op_name, const std::string &param_name,
                    const int *value, const int num);
void genCaseOpParam(const int flag, std::string *gen_case_file_name,
                    const std::string &op_name, const std::string &param_name,
                    const float *value, const int num);
void genCaseTestParam(std::string *gen_case_file_name, const bool is_diff1,
                      const bool is_diff2, const bool is_diff3,
                      const float diff1_threshold, const float diff2_threshold,
                      const float diff3_threshold);
int getIntEnvVar(const std::string &str, const int default_para);
bool getBoolEnvVar(const std::string &str, const bool default_para);
std::string getStringEnvVar(const std::string &str,
                            const std::string default_para);
bool getBoolOpName(const std::string &str, const std::string &op_name);
int mkdirRecursive(const char *pathname);
}  // namespace gen_case
}  // namespace cnnl
#endif  // GEN_CASE_HPP_
