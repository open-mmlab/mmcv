#include "onnxruntime_register.h"

#include "ort_mmcv_utils.h"
#include "soft_nms.h"

const char *c_MMCVOpDomain = "mmcv";
SoftNmsOp c_SoftNmsOp;

OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                          const OrtApiBase *api) {
  OrtCustomOpDomain *domain = nullptr;
  const OrtApi *ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain(c_MMCVOpDomain, &domain)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_SoftNmsOp)) {
    return status;
  }

  return ortApi->AddCustomOpDomain(options, domain);
}
