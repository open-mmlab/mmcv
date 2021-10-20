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

void KernelFocalLossSigmoidForward(cnrtDim3_t k_dim,
                                   cnrtFunctionType_t k_type,
                                   cnrtQueue_t queue,
                                   const cnrtDataType_t d_type,
                                   const void *input,
                                   const void *target,
                                   const void *weight,
                                   const int32_t N,
                                   const int32_t C,
                                   const float alpha,
                                   const float gamma,
                                   void *output);

void KernelFocalLossSigmoidBackward(cnrtDim3_t k_dim,
                                    cnrtFunctionType_t k_type,
                                    cnrtQueue_t queue,
                                    const cnrtDataType_t d_type,
                                    const void *input,
                                    const void *target,
                                    const void *weight,
                                    const float gamma,
                                    const float alpha,
                                    const int32_t dim_n,
                                    const int32_t deal_n,
                                    const int32_t dim_c,
                                    void *output);

#endif  // BANG_INTERNEL_H_
