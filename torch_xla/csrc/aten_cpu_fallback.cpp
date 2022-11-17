#include "torch_xla/csrc/aten_cpu_fallback.h"

#include <third_party/xla_client/debug_macros.h>
#include <third_party/xla_client/metrics.h>
#include <third_party/xla_client/tf_logging.h>
#include <torch_xla/csrc/function_call_tracker.h>

#include <unordered_map>

namespace torch_xla {

// TODO(jwtan): Replace this with torch::lazy::Counter. We need
// _cpu_fallback_counters to remain as xla::metrics::Counter to support
// xla::metrics::CreatePerformanceReport(). For more information, see NOTE:
// [TORCH_LAZY_COUNTER v.s. XLA_COUNTER].
static std::unordered_map<std::string, ::xla::metrics::Counter*>
    _cpu_fallback_counters;

void xla_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  std::cout << "WONJOO: at aten_cpu_fallback.cpp, xla_cpu_fallback1" << std::endl;
  XLA_FN_TRACK(3);
  const auto name = c10::toString(op.operator_name());

  // Manually applying the XLA_COUNTER macro.
  // We need to do it ourselves and explicitly keep a mapping of counters
  // because this boxed fallback kernel is used by multiple operators,
  // and the macro stamps out a static Counter object with a fixed name
  // at the code location that it was called.
  if (_cpu_fallback_counters.find(name) == _cpu_fallback_counters.end()) {
    _cpu_fallback_counters[name] = new ::xla::metrics::Counter(name);
  }
  _cpu_fallback_counters[name]->AddValue(1);

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

}  // namespace torch_xla
