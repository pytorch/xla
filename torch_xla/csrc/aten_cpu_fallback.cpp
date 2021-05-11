#include "torch_xla/csrc/aten_cpu_fallback.h"

#include <ATen/native/cpu_fallback.h>

#include <tensorflow/compiler/xla/xla_client/debug_macros.h>
#include <tensorflow/compiler/xla/xla_client/metrics.h>
#include <tensorflow/compiler/xla/xla_client/tf_logging.h>
#include <torch_xla/csrc/function_call_tracker.h>

namespace torch_xla {

void xla_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  XLA_FN_TRACK(3);
  XLA_COUNTER(op.schema().name(), 1);

  auto& args = op.schema().arguments();
  auto arguments = torch::jit::last(stack, args.size());

  // Log each tensor argument.
  for (int64_t idx = 0; idx < arguments.size(); ++idx) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      TF_VLOG(3) << ivalue.toTensor().toString();
    }
  }

  // Call the actual boxed CPU fallback.
  at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(_, XLA, m) {
   m.fallback(torch::CppFunction::makeFromBoxedFunction<&xla_cpu_fallback>());
}

} // namespace torch_xla
