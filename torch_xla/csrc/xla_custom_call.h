#ifndef XLA_TORCH_XLA_CSRC_XLA_CUSTOM_CALL_H_
#define XLA_TORCH_XLA_CSRC_XLA_CUSTOM_CALL_H_

#include "xla/service/custom_call_target_registry.h"

// Creates a CPU HLO instruction for argmin using custom-calls.
void argmin_custom(void* out, const void** in);
XLA_REGISTER_CUSTOM_CALL_TARGET(argmin_custom, "Host");

// Creates a GPU HLO instruction for sum using custom-calls.
// void do_custom_gpu_call(CUstream stream, void** buffers,
//                     const char* opaque, size_t opaque_len);

#endif  // XLA_TORCH_XLA_CSRC_XLA_CUSTOM_CALL_H_