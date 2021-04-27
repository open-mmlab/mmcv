#ifndef ONNXRUNTIME_CORNER_POOL_H
#define ONNXRUNTIME_CORNER_POOL_H

#include <onnxruntime_cxx_api.h>

struct MMCVTopPoolKernel {
    public:
        MMCVTopPoolKernel(Ort::CustomOpApi ort, const OrtKernelInfo* info): ort_(ort) {
            // create allocator
            allocator_ = Ort::AllocatorWithDefaultOptions();
        }

        void Compute(OrtKernelContext* context);

    private:
        Ort::CustomOpApi ort_;
        Ort::AllocatorWithDefaultOptions allocator_;
};

struct MMCVTopPoolCustomOp : Ort::CustomOpBase<MMCVTopPoolCustomOp, MMCVTopPoolKernel> {
    void *CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo* info) {
        return new MMCVTopPoolKernel(api, info);
    }

    const char* GetName() const { return "MMCVTopPool"; }

    size_t GetInputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetInputType(size_t) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }

    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }

    // force cpu
    const char* GetExecutionProviderType() const {
        return "CPUExecutionProvider";
    }
};
#endif  // ONNXRUNTIME_CORNER_POOL_H
