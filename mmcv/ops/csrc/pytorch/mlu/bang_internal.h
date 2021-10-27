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
#ifndef BANG_INTERNEL_H_
#define BANG_INTERNEL_H_

void KernelNms(cnrtDim3_t k_dim,
               cnrtFunctionType_t k_type,
               cnrtQueue_t queue,
               const cnrtDataType_t data_type_input,
               const void *boxes_ptr,
               const void *scores_ptr,
               const int input_num_boxes,
               const int input_stride,
               const int max_output_boxes,
               const float iou_threshold,
               const float offset,
               void *workspace_ptr,
               void *output_size_ptr,
               void *output_ptr);

#endif  // BANG_INTERNEL_H_
