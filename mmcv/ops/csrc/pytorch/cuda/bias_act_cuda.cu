// Modified from
// https://github.com/NVlabs/stylegan3/blob/main/torch_utils/ops/bias_act.cpp

// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <c10/util/Half.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "../bias_act.h"

//------------------------------------------------------------------------

//------------------------------------------------------------------------
// Helpers.

template <class T> struct InternalType;
template <> struct InternalType<double>     { typedef double scalar_t; };
template <> struct InternalType<float>      { typedef float  scalar_t; };
template <> struct InternalType<c10::Half>  { typedef float  scalar_t; };

//------------------------------------------------------------------------
// CUDA kernel.

template <class T, int A>
__global__ void bias_act_kernel(bias_act_kernel_params p)
{
    typedef typename InternalType<T>::scalar_t scalar_t;
    int G                 = p.grad;
    scalar_t alpha        = (scalar_t)p.alpha;
    scalar_t gain         = (scalar_t)p.gain;
    scalar_t clamp        = (scalar_t)p.clamp;
    scalar_t one          = (scalar_t)1;
    scalar_t two          = (scalar_t)2;
    scalar_t expRange     = (scalar_t)80;
    scalar_t halfExpRange = (scalar_t)40;
    scalar_t seluScale    = (scalar_t)1.0507009873554804934193349852946;
    scalar_t seluAlpha    = (scalar_t)1.6732632423543772848170429916717;

    // Loop over elements.
    int xi = blockIdx.x * p.loopX * blockDim.x + threadIdx.x;
    for (int loopIdx = 0; loopIdx < p.loopX && xi < p.sizeX; loopIdx++, xi += blockDim.x)
    {
        // Load.
        scalar_t x = (scalar_t)((const T*)p.x)[xi];
        scalar_t b = (p.b) ? (scalar_t)((const T*)p.b)[(xi / p.stepB) % p.sizeB] : 0;
        scalar_t xref = (p.xref) ? (scalar_t)((const T*)p.xref)[xi] : 0;
        scalar_t yref = (p.yref) ? (scalar_t)((const T*)p.yref)[xi] : 0;
        scalar_t dy = (p.dy) ? (scalar_t)((const T*)p.dy)[xi] : one;
        scalar_t yy = (gain != 0) ? yref / gain : 0;
        scalar_t y = 0;

        // Apply bias.
        ((G == 0) ? x : xref) += b;

        // linear
        if (A == 1)
        {
            if (G == 0) y = x;
            if (G == 1) y = x;
        }

        // relu
        if (A == 2)
        {
            if (G == 0) y = (x > 0) ? x : 0;
            if (G == 1) y = (yy > 0) ? x : 0;
        }

        // lrelu
        if (A == 3)
        {
            if (G == 0) y = (x > 0) ? x : x * alpha;
            if (G == 1) y = (yy > 0) ? x : x * alpha;
        }

        // tanh
        if (A == 4)
        {
            if (G == 0) { scalar_t c = exp(x); scalar_t d = one / c; y = (x < -expRange) ? -one : (x > expRange) ? one : (c - d) / (c + d); }
            if (G == 1) y = x * (one - yy * yy);
            if (G == 2) y = x * (one - yy * yy) * (-two * yy);
        }

        // sigmoid
        if (A == 5)
        {
            if (G == 0) y = (x < -expRange) ? 0 : one / (exp(-x) + one);
            if (G == 1) y = x * yy * (one - yy);
            if (G == 2) y = x * yy * (one - yy) * (one - two * yy);
        }

        // elu
        if (A == 6)
        {
            if (G == 0) y = (x >= 0) ? x : exp(x) - one;
            if (G == 1) y = (yy >= 0) ? x : x * (yy + one);
            if (G == 2) y = (yy >= 0) ? 0 : x * (yy + one);
        }

        // selu
        if (A == 7)
        {
            if (G == 0) y = (x >= 0) ? seluScale * x : (seluScale * seluAlpha) * (exp(x) - one);
            if (G == 1) y = (yy >= 0) ? x * seluScale : x * (yy + seluScale * seluAlpha);
            if (G == 2) y = (yy >= 0) ? 0 : x * (yy + seluScale * seluAlpha);
        }

        // softplus
        if (A == 8)
        {
            if (G == 0) y = (x > expRange) ? x : log(exp(x) + one);
            if (G == 1) y = x * (one - exp(-yy));
            if (G == 2) { scalar_t c = exp(-yy); y = x * c * (one - c); }
        }

        // swish
        if (A == 9)
        {
            if (G == 0)
                y = (x < -expRange) ? 0 : x / (exp(-x) + one);
            else
            {
                scalar_t c = exp(xref);
                scalar_t d = c + one;
                if (G == 1)
                    y = (xref > halfExpRange) ? x : x * c * (xref + d) / (d * d);
                else
                    y = (xref > halfExpRange) ? 0 : x * c * (xref * (two - d) + two * d) / (d * d * d);
                yref = (xref < -expRange) ? 0 : xref / (exp(-xref) + one) * gain;
            }
        }

        // Apply gain.
        y *= gain * dy;

        // Clamp.
        if (clamp >= 0)
        {
            if (G == 0)
                y = (y > -clamp & y < clamp) ? y : (y >= 0) ? clamp : -clamp;
            else
                y = (yref > -clamp & yref < clamp) ? y : 0;
        }

        // Store.
        ((T*)p.y)[xi] = (T)y;
    }
}

//------------------------------------------------------------------------
// CUDA kernel selection.

template <class T> void* choose_bias_act_kernel(const bias_act_kernel_params& p)
{
    if (p.act == 1) return (void*)bias_act_kernel<T, 1>;
    if (p.act == 2) return (void*)bias_act_kernel<T, 2>;
    if (p.act == 3) return (void*)bias_act_kernel<T, 3>;
    if (p.act == 4) return (void*)bias_act_kernel<T, 4>;
    if (p.act == 5) return (void*)bias_act_kernel<T, 5>;
    if (p.act == 6) return (void*)bias_act_kernel<T, 6>;
    if (p.act == 7) return (void*)bias_act_kernel<T, 7>;
    if (p.act == 8) return (void*)bias_act_kernel<T, 8>;
    if (p.act == 9) return (void*)bias_act_kernel<T, 9>;
    return NULL;
}

//------------------------------------------------------------------------
// Template specializations.

template void* choose_bias_act_kernel<double>       (const bias_act_kernel_params& p);
template void* choose_bias_act_kernel<float>        (const bias_act_kernel_params& p);
template void* choose_bias_act_kernel<c10::Half>    (const bias_act_kernel_params& p);

//------------------------------------------------------------------------


static bool has_same_layout(torch::Tensor x, torch::Tensor y)
{
    if (x.dim() != y.dim())
        return false;
    for (int64_t i = 0; i < x.dim(); i++)
    {
        if (x.size(i) != y.size(i))
            return false;
        if (x.size(i) >= 2 && x.stride(i) != y.stride(i))
            return false;
    }
    return true;
}

//------------------------------------------------------------------------

static torch::Tensor bias_act_op(torch::Tensor x, torch::Tensor b, torch::Tensor xref, torch::Tensor yref, torch::Tensor dy, int grad, int dim, int act, float alpha, float gain, float clamp)
{
    // Validate arguments.
    TORCH_CHECK(x.is_cuda(), "x must reside on CUDA device");
    TORCH_CHECK(b.numel() == 0 || (b.dtype() == x.dtype() && b.device() == x.device()), "b must have the same dtype and device as x");
    TORCH_CHECK(xref.numel() == 0 || (xref.sizes() == x.sizes() && xref.dtype() == x.dtype() && xref.device() == x.device()), "xref must have the same shape, dtype, and device as x");
    TORCH_CHECK(yref.numel() == 0 || (yref.sizes() == x.sizes() && yref.dtype() == x.dtype() && yref.device() == x.device()), "yref must have the same shape, dtype, and device as x");
    TORCH_CHECK(dy.numel() == 0 || (dy.sizes() == x.sizes() && dy.dtype() == x.dtype() && dy.device() == x.device()), "dy must have the same dtype and device as x");
    TORCH_CHECK(x.numel() <= INT_MAX, "x is too large");
    TORCH_CHECK(b.dim() == 1, "b must have rank 1");
    TORCH_CHECK(b.numel() == 0 || (dim >= 0 && dim < x.dim()), "dim is out of bounds");
    TORCH_CHECK(b.numel() == 0 || b.numel() == x.size(dim), "b has wrong number of elements");
    TORCH_CHECK(grad >= 0, "grad must be non-negative");

    // Validate layout.
    TORCH_CHECK(x.is_non_overlapping_and_dense(), "x must be non-overlapping and dense");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(xref.numel() == 0 || has_same_layout(xref, x), "xref must have the same layout as x");
    TORCH_CHECK(yref.numel() == 0 || has_same_layout(yref, x), "yref must have the same layout as x");
    TORCH_CHECK(dy.numel() == 0 || has_same_layout(dy, x), "dy must have the same layout as x");

    // Create output tensor.
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    torch::Tensor y = torch::empty_like(x);
    TORCH_CHECK(has_same_layout(y, x), "y must have the same layout as x");

    // Initialize CUDA kernel parameters.
    bias_act_kernel_params p;
    p.x     = x.data_ptr();
    p.b     = (b.numel()) ? b.data_ptr() : NULL;
    p.xref  = (xref.numel()) ? xref.data_ptr() : NULL;
    p.yref  = (yref.numel()) ? yref.data_ptr() : NULL;
    p.dy    = (dy.numel()) ? dy.data_ptr() : NULL;
    p.y     = y.data_ptr();
    p.grad  = grad;
    p.act   = act;
    p.alpha = alpha;
    p.gain  = gain;
    p.clamp = clamp;
    p.sizeX = (int)x.numel();
    p.sizeB = (int)b.numel();
    p.stepB = (b.numel()) ? (int)x.stride(dim) : 1;

    // Choose CUDA kernel.
    void* kernel;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "upfirdn2d_cuda", [&]
    {
        kernel = choose_bias_act_kernel<scalar_t>(p);
    });
    TORCH_CHECK(kernel, "no CUDA kernel found for the specified activation func");

    // Launch CUDA kernel.
    p.loopX = 4;
    int blockSize = 4 * 32;
    int gridSize = (p.sizeX - 1) / (p.loopX * blockSize) + 1;
    void* args[] = {&p};
    AT_CUDA_CHECK(cudaLaunchKernel(kernel, gridSize, blockSize, args, 0, at::cuda::getCurrentCUDAStream()));
    return y;
}
