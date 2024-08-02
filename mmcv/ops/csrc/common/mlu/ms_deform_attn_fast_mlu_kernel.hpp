/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef MS_DEFORM_ATTN_FORWARD_FAST_MLU_KERNEL_HPP_
#define MS_DEFORM_ATTN_FORWARD_FAST_MLU_KERNEL_HPP_
void KernelMsDeformAttnForwardFast(
    cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
    const cnrtDataType_t d_type, const char *data_value_gdram,
    const char *data_spatial_shapes_gdram,
    const char *data_level_start_index_gdram,
    const char *data_sampling_loc_gdram, const char *data_attn_weight_gdram,
    const int32_t batch_size, const int32_t num_keys, const int32_t num_heads,
    const int32_t channels, const int32_t num_levels, const int32_t num_queries,
    const int32_t num_points, char *data_col_gdram);
#endif  // MS_DEFORM_ATTN_FORWARD_FAST_MLU_KERNEL_HPP_
