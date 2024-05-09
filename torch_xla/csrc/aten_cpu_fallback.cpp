#include "torch_xla/csrc/aten_cpu_fallback.h"

#include <unordered_map>

#include "torch_xla/csrc/function_call_tracker.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/metrics.h"
#include "torch_xla/csrc/runtime/tf_logging.h"
#include "torch/csrc/jit/python/pybind_utils.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace torch_xla {

// TODO(jwtan): Replace this with torch::lazy::Counter. We need
// _cpu_fallback_counters to remain as torch_xla::runtime::metrics::Counter to
// support torch_xla::runtime::metrics::CreatePerformanceReport(). For more
// information, see NOTE: [TORCH_LAZY_COUNTER v.s. XLA_COUNTER].
static std::unordered_map<std::string, ::torch_xla::runtime::metrics::Counter*>
    _cpu_fallback_counters;


std::pair<py::object, py::dict> parseIValuesToPyArgsKwargs(
    const c10::OperatorHandle& op,
    const std::vector<c10::IValue>& arguments) {
  TORCH_CHECK(
      PyGILState_Check(),
      "GIL must be held before you call parseIValuesToPyArgsKwargs");
  const auto& schema = op.schema();
  py::dict kwargs;
  // About all the pointers:
  //
  // f(int x, int y = 0, *, int z = 0)
  //                                  ^- arguments.size()
  //                        ^- kwarg_only_start
  //          ^- positional_default_start
  //   ^- 0

  // Find the split point between kwarg-only and regular.  Since most functions
  // don't have kwarg-only arguments, it is more efficient to scan from the
  // right (but ideally, this would just be precomputed in FunctionSchema
  // itself).  (NB: minus one in the loop is because we're testing if the
  // *next* argument is kwarg-only before we advance the starting index)
  int64_t kwarg_only_start = static_cast<int64_t>(arguments.size());
  for (; kwarg_only_start > 0; kwarg_only_start--) {
    const auto& arg = schema.arguments()[kwarg_only_start - 1];
    if (!arg.kwarg_only()) {
      break;
    }
  }

  // Find the first positional argument that isn't defaulted
  auto is_default = [&](size_t idx) -> bool {
    const auto& arg = schema.arguments()[idx];
    if (!arg.default_value().has_value()) {
      return false;
    }
    const auto& default_ivalue = *arg.default_value();
    const auto& ivalue = arguments[idx];
    if (default_ivalue != ivalue) {
      return false;
    }
    return true;
  };

  int64_t positional_default_start = kwarg_only_start;
  for (; positional_default_start > 0; positional_default_start--) {
    if (!is_default(positional_default_start - 1)) {
      break;
    }
  }

  auto args =
      py::reinterpret_steal<py::object>(PyTuple_New(positional_default_start));

  auto schemaAwareToPyObject = [&](size_t idx) -> py::object {
    const auto& arg = schema.arguments()[idx];
    auto match = [&](c10::TypeKind kind) {
      const auto& t = arg.real_type();
      if (t->kind() == kind)
        return true;
      if (auto opt_t = t->cast<c10::OptionalType>()) {
        if (opt_t->getElementType()->kind() == kind)
          return true;
      }
      return false;
    };
    if (arguments[idx].isNone()) {
      return py::none();
    } else if (match(c10::ScalarTypeType::Kind)) {
      auto* obj =
          torch::getTHPDtype(static_cast<c10::ScalarType>(arguments[idx].toInt()));
      return py::reinterpret_borrow<py::object>(
          reinterpret_cast<PyObject*>(obj));
    } else if (match(c10::LayoutType::Kind)) {
      auto* obj =
          torch::getTHPLayout(static_cast<c10::Layout>(arguments[idx].toInt()));
      return py::reinterpret_borrow<py::object>(
          reinterpret_cast<PyObject*>(obj));
    } else if (match(c10::MemoryFormatType::Kind)) {
      return py::cast(static_cast<c10::MemoryFormat>(arguments[idx].toInt()));
    } else {
      return torch::jit::toPyObject(arguments[idx]);
    }
  };

  // Populate positional arguments
  for (const auto idx : c10::irange(positional_default_start)) {
    PyTuple_SET_ITEM(
        args.ptr(), idx, schemaAwareToPyObject(idx).release().ptr());
  }

  // Populate keyword arguments
  for (const auto idx : c10::irange(kwarg_only_start, arguments.size())) {
    // But don't populate default keyword arguments
    if (is_default(idx))
      continue;
    const auto& arg = schema.arguments()[idx];
    kwargs[py::cast(arg.name())] = schemaAwareToPyObject(idx);
  }
  return std::make_pair(std::move(args), std::move(kwargs));
}

void pushPyOutToStack(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    py::object out,
    const char* msg) {
  TORCH_CHECK(
      PyGILState_Check(), "GIL must be held before you call pushPyOutToStack");
  auto schema_returns = op.schema().returns();
  const auto num_returns = schema_returns.size();
  if (num_returns == 0) {
    // Check that we got a None return from Python. Anything else is an error.
    TORCH_CHECK(
        out.is_none(),
        "Expected ",
        msg,
        " for ",
        op.operator_name(),
        " to return None but it returned something else instead.");
  } else if (num_returns == 1) {
    torch::jit::push(
        stack, torch::jit::toIValue(out.ptr(), schema_returns[0].real_type()));
  } else {
    auto outs = py::cast<py::sequence>(out);
    for (const auto idx : c10::irange(outs.size())) {
      torch::jit::push(
          stack,
          torch::jit::toIValue(
              outs[idx].ptr(), schema_returns[idx].real_type()));
    }
  }
}

void xla_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  XLA_FN_TRACK(3);
  const auto name = c10::toString(op.operator_name());

  if (op.hasSchema()) {
    py::gil_scoped_acquire gil;
    auto decomps = py::module_::import("torch_xla.core.decompositions");
    auto fn = decomps.attr("get_decomposition")(name);
    if (!fn.is_none()) {
      auto arguments = torch::jit::pop(*stack, op.schema().arguments().size());
      auto a = parseIValuesToPyArgsKwargs(op, arguments);
      auto out = fn(std::move(a.first), std::move(a.second));
      if (!out.is_none()) {
        pushPyOutToStack(op, stack, out, "xla_cpu_fallback");
        return;
      }
    }
  }

  // Manually applying the XLA_COUNTER macro.
  // We need to do it ourselves and explicitly keep a mapping of counters
  // because this boxed fallback kernel is used by multiple operators,
  // and the macro stamps out a static Counter object with a fixed name
  // at the code location that it was called.
  if (_cpu_fallback_counters.find(name) == _cpu_fallback_counters.end()) {
    _cpu_fallback_counters[name] =
        new ::torch_xla::runtime::metrics::Counter(name);
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
  // Set error_on_views as XLA should take care
  // of all view ops after functionalization.
  at::native::cpu_fallback(op, stack, true);
}

TORCH_LIBRARY_IMPL(_, XLA, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&xla_cpu_fallback>());
}

}  // namespace torch_xla
