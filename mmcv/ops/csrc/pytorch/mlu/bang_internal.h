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

void KernelBBoxOverlaps(cnrtDim3_t k_dim,
                        cnrtFunctionType_t k_type,
                        cnrtQueue_t queue,
                        const cnrtDataType_t d_type,
                        const void *bboxes1,
                        const void *bboxes2,
                        void *ious,
                        const int32_t num_bbox1,
                        const int32_t num_bbox2,
                        const int32_t mode,
                        const bool aligned,
                        const int32_t offset);

#endif  // BANG_INTERNEL_H_
