#ifndef XLA_TORCH_XLA_CSRC_XLA_CUSTOM_CALL_H_
#define XLA_TORCH_XLA_CSRC_XLA_CUSTOM_CALL_H_

#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include <cuda_runtime.h>
#include <cuda.h>

void argmin_custom(void* out, const void** in);
XLA_REGISTER_CUSTOM_CALL_TARGET(argmin_custom, "Host");

void do_custom_gpu_call(CUstream stream, void** buffers,
                    const char* opaque, size_t opaque_len);
XLA_REGISTER_CUSTOM_CALL_TARGET(do_custom_gpu_call, "CUDA");

#endif  // XLA_TORCH_XLA_CSRC_XLA_CUSTOM_CALL_H_
