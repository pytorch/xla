#ifndef XLA_TORCH_XLA_CSRC_ATEN_CPU_FALLBACK_H_
#define XLA_TORCH_XLA_CSRC_ATEN_CPU_FALLBACK_H_

#include <ATen/native/CPUFallback.h>

namespace torch_xla {

void xla_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);

std::vector<std::string> GetFallbackOperations();

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_ATEN_CPU_FALLBACK_H_
