#ifndef XLA_TORCH_XLA_CSRC_XLA_CUSTOM_CALL_H_
#define XLA_TORCH_XLA_CSRC_XLA_CUSTOM_CALL_H_

#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"

void argmin_custom(void* out, const void** in);
XLA_REGISTER_CUSTOM_CALL_TARGET(argmin_custom, "Host");


#endif  // XLA_TORCH_XLA_CSRC_XLA_CUSTOM_CALL_H_