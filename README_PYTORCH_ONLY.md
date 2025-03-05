# PyTorch-Only MMCV

This project aims to replace all CUDA and C++ implementations in MMCV with pure PyTorch equivalents. This makes it easier to use MMCV in environments where compiling custom CUDA kernels is difficult or impossible.

## Current Status

Several core operations have been reimplemented in pure PyTorch:

- NMS and variants (nms, soft-nms, nms-match, quadrilateral-nms)
- ROI operations (ROI pooling and ROI align)

See [pytorch_only_summary.md](pytorch_only_summary.md) for a complete list of implemented and yet-to-be-implemented operations.

## Usage

Most operations work exactly the same way as before. However, there are some limitations:

1. **Backward Passes**: The backward passes are not properly implemented for some operations, which means they may not work correctly for training. They should work fine for inference.
2. **Performance**: These pure PyTorch implementations are likely to be slower than the original CUDA-optimized versions.
3. **Numerical Precision**: There may be slight numerical differences compared to the original implementations.

## Adding More Implementations

To add more pure PyTorch implementations:

1. Create a file in `mmcv/ops/pure_pytorch_<operation>.py` with your implementation
2. Modify the corresponding operation file to use your implementation
3. Update the `remove_cuda_deps.py` script to automatically detect and replace uses of your operation
4. Update the `pytorch_only_summary.md` file

### Implementation Guidelines

1. Try to match the exact API of the original function
2. Prioritize correctness over performance 
3. Document any limitations or differences from the original implementation
4. Include a proper backward pass if possible

## Testing Your Implementations

To test your implementations:

1. Find or create a test case in the `tests/` directory
2. Run the test to ensure your implementation produces similar results to the original

## Contributing

Contributions to complete the remaining operations are welcome! Please follow these steps:

1. Choose an operation from `pytorch_only_summary.md` that's not yet implemented
2. Implement it following the guidelines above
3. Test your implementation
4. Submit a pull request with your changes

## Limitations

This project is primarily intended for:

1. Environments where compiling CUDA extensions is difficult
2. Inference use cases rather than training
3. Applications where performance is not the top priority

For high-performance or training-critical applications, you may want to use the original MMCV with CUDA extensions.

## License

This project follows the same license as MMCV. See the LICENSE file for details.