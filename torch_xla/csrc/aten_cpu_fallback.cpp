#include "torch_xla/csrc/aten_cpu_fallback.h"

#include <c10/cuda/CUDAFunctions.h>
#include <ATen/DLConvertor.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <ATen/ops/_copy_from_and_resize.h>
#include <ATen/ops/_to_cpu.h>

#include <sstream>
#include <unordered_map>
#include <vector>

#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/dl_convertor.h"
#include "torch_xla/csrc/function_call_tracker.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/metrics.h"
#include "torch_xla/csrc/runtime/tf_logging.h"
#include "torch_xla/csrc/xla_graph_executor.h"

namespace torch_xla {

// TODO(jwtan): Replace this with torch::lazy::Counter. We need
// _fallback_counters to remain as torch_xla::runtime::metrics::Counter to
// support torch_xla::runtime::metrics::CreatePerformanceReport(). For more
// information, see NOTE: [TORCH_LAZY_COUNTER v.s. XLA_COUNTER].
static std::unordered_map<std::string, ::torch_xla::runtime::metrics::Counter*>
    _fallback_counters;

// Get all the executed fallback operations.
// In other words, get all of them whose counters are not zero.
std::vector<std::string> GetFallbackOperations() {
  std::vector<std::string> fallback;
  for (auto const& pair : _fallback_counters) {
    if (pair.second->Value() != 0) {
      fallback.push_back(pair.first);
    }
  }
  return fallback;
}

bool UseCUDAFallback() {
  return runtime::sys_util::GetEnvBool("XLA_FALLBACK_CUDA", false);
}

static bool validate_tensor_list(const c10::List<at::Tensor>& tensorlist) {
  return std::any_of(tensorlist.begin(), tensorlist.end(),
                     [](const at::Tensor& tensor) { return tensor.defined(); });
}

static std::vector<at::Tensor> to_cuda(const at::TensorList& tensors,
                                       at::DeviceIndex* index) {
  // Synchronize tensors, so that we are able to grab their data pointer.
  std::vector<XLATensorPtr> xla_tensors(tensors.size());
  std::transform(
      tensors.begin(), tensors.end(), xla_tensors.begin(),
      [](const at::Tensor& tensor) { return bridge::GetXlaTensor(tensor); });
  XLAGraphExecutor::Get()->SyncTensorsGraph(&xla_tensors, {}, true, true,
                                            false);

  // Use DLPack for sharing the XLA storage with a newly created CUDA tensor.
  std::vector<at::Tensor> cuda_tensors(tensors.size());
  std::transform(tensors.begin(), tensors.end(), cuda_tensors.begin(),
                 [=](const at::Tensor& tensor) {
                   // Grab the DLManagedTensor.
                     DLManagedTensor* managed = torch_xla::toDLPack(tensor);
                   // Remember the device index, so that we can synchronize it
                   // later. Also ensure that there's only one.
                   at::DeviceIndex this_index =
                       managed->dl_tensor.device.device_id;
                   if (*index == -1) {
                     *index = this_index;
                   } else {
                     TORCH_CHECK(*index == this_index,
                                 "fallback using Multi-GPUs not supported");
                   }
                   // Create the CUDA tensor.
                   return at::fromDLPack(
                       managed, [=](void*) { managed->deleter(managed); });
                 });
  return cuda_tensors;
}

static at::Tensor to_xla(const at::Tensor& tensor) {
  return torch_xla::fromDLPack(at::toDLPack(tensor));
}

static void torch_cuda_synchronize(at::DeviceIndex index) {
  at::DeviceIndex current = c10::cuda::current_device();
  c10::cuda::set_device(index);
  c10::cuda::device_synchronize();
  c10::cuda::set_device(current);
}

void cuda_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack,
                   bool error_on_views) {
  auto& schema_args = op.schema().arguments();
  const auto num_arguments = schema_args.size();
  auto arguments = torch::jit::last(stack, num_arguments);
  const auto arguments_begin = stack->size() - num_arguments;

  std::vector<at::Tensor> tensor_args;
  std::vector<int> tensor_args_indices;

  std::vector<c10::List<at::Tensor>> tensorlist_args;
  std::vector<int> tensorlist_args_indices;

  at::DeviceIndex device_index = -1;

  // Step 1: Convert all non-CUDA tensor inputs into CUDA tensors
  // and put them on the stack at the correct indices.
  for (const auto idx : c10::irange(arguments.size())) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      tensor_args.push_back(ivalue.toTensor());
      tensor_args_indices.push_back(idx);
    } else if (ivalue.isTensorList()) {
      // Note: we copy each TensorList argument to CUDA individually out of
      // convenience, but XLA would benefit from materializing all tensor and
      // TensorList args onto the CUDA at the same time. We can improve this if
      // we need better perf for XLA's CUDA fallbacks.
      tensorlist_args.push_back(ivalue.toTensorList());
      tensorlist_args_indices.push_back(idx);
      auto cuda_ivalue = c10::IValue(c10::List<at::Tensor>(
          to_cuda(ivalue.toTensorList().vec(), &device_index)));
      (*stack)[arguments_begin + idx] = std::move(cuda_ivalue);
      tensorlist_args.push_back(ivalue.toTensorList());
    } else if (ivalue.isOptionalTensorList()) {
      auto opt_tensors = ivalue.toOptionalTensorList().vec();
      std::vector<at::Tensor> need_convert_tensors;
      std::vector<int> need_convert_tensors_index;
      for (auto i : c10::irange(opt_tensors.size())) {
        if (!opt_tensors[i].has_value() || !opt_tensors[i]->defined()) continue;
        need_convert_tensors.push_back(opt_tensors[i].value());
        need_convert_tensors_index.push_back(i);
      }
      auto cuda_tensors = to_cuda(need_convert_tensors, &device_index);
      for (const auto i : c10::irange(need_convert_tensors_index.size())) {
        auto idx = need_convert_tensors_index[i];
        opt_tensors[idx] = cuda_tensors[i];
      }
      (*stack)[arguments_begin + idx] = c10::IValue(opt_tensors);
    }
  }
  // XLA requires all of the tensor arguments to be gathered up and converted to
  // CUDA together.
  auto cuda_tensors = to_cuda(tensor_args, &device_index);

  for (const auto i : c10::irange(tensor_args_indices.size())) {
    auto idx = tensor_args_indices[i];
    (*stack)[arguments_begin + idx] = c10::IValue(cuda_tensors[i]);
  }

  // Step 2: Call the underlying CUDA implementation of the operator
  op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CUDA), stack);

  // Step 4: Convert any CUDA output tensors back to the original input device.
  // For mutable alias'd outputs, we also need to take special care
  // to move the ORIGINAL input tensor back onto the stack, in place of
  // the temporary CUDA output tensor that we created.
  //
  // See [CUDA Fallback Does Not Handle View Operators]
  const auto& schema_returns = op.schema().returns();
  const auto& num_returns = schema_returns.size();
  auto returns = torch::jit::last(stack, num_returns);
  const auto returns_begin = stack->size() - num_returns;

  // Synchronize the device before actually converting back to XLA.
  torch_cuda_synchronize(device_index);

  for (const auto idx : c10::irange(returns.size())) {
    const c10::AliasInfo* alias_info = schema_returns[idx].alias_info();
    if (alias_info != nullptr && alias_info->isWrite()) {
      // Case (1): mutable alias case.
      // Move the input ivalue directly onto the stack in place of
      // the existing cuda output tensor.
      bool found_alias = false;
      if (returns[idx].isTensor() && returns[idx].toTensor().defined()) {
        // We could store some extra metadata on the function schema to avoid
        // the loop here if we need to improve perf.
        for (const auto i : c10::irange(tensor_args_indices.size())) {
          auto input_tensor_idx = tensor_args_indices[i];
          const auto& input_tensor = cuda_tensors[i];
          const c10::AliasInfo* input_alias_info =
              schema_args[input_tensor_idx].alias_info();
          // Checked above; adding assert to guard against breakage of the below
          // condition due to changing the above if test.
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(alias_info != nullptr);
          if (input_tensor.defined() && (alias_info == input_alias_info ||
                                         (input_alias_info != nullptr &&
                                          *alias_info == *input_alias_info))) {
            // We've found the original input tensor that aliases with the
            // current output. Wrap it in an IValue and put it directly on the
            // stack.
            (*stack)[returns_begin + idx] = c10::IValue(tensor_args[i]);
            found_alias = true;
            break;
          }
        }
      } else if (returns[idx].isTensorList() &&
                 validate_tensor_list(returns[idx].toTensorList())) {
        for (const auto i : c10::irange(tensorlist_args_indices.size())) {
          auto input_tensor_idx = tensorlist_args_indices[i];
          const c10::AliasInfo* input_alias_info =
              schema_args[input_tensor_idx].alias_info();
          // Checked above; adding assert to guard against breakage of the below
          // condition due to changing the above if test.
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(alias_info != nullptr);
          if (validate_tensor_list(tensorlist_args[i]) &&
              (alias_info == input_alias_info ||
               (input_alias_info != nullptr &&
                *alias_info == *input_alias_info))) {
            // We've found the original input tensor that aliases with the
            // current output. Wrap it in an IValue and put it directly on the
            // stack.
            (*stack)[returns_begin + idx] = c10::IValue(tensorlist_args[i]);
            found_alias = true;
            break;
          }
        }
      }
      TORCH_CHECK(
          found_alias, "The operator ", op.schema().operator_name(),
          " appears to have invalid alias information. ",
          "Found a return tensor argument with a mismatched mutable alias: ",
          schema_returns[idx]);
    } else {
      if (alias_info != nullptr && !alias_info->isWrite()) {
        // Case (3): immutable alias (view) case.
        TORCH_CHECK(
            false, "The operator ", op.schema().operator_name(),
            " appears to be a view operator, ",
            "but it has no implementation for the backend \"xla\". ",
            "View operators don't support ",
            "since the tensor's storage cannot be shared across devices.");
      }
      // Case (2): copy case.
      // Copy the CUDA output tensor to the original device.
      if (returns[idx].isTensor() && returns[idx].toTensor().defined()) {
        (*stack)[returns_begin + idx] =
            c10::IValue(to_xla(returns[idx].toTensor()));
      } else if (returns[idx].isTensorList() &&
                 validate_tensor_list(returns[idx].toTensorList())) {
        const auto& cuda_tensors = returns[idx].toTensorList().vec();
        std::vector<at::Tensor> tensors;
        tensors.reserve(cuda_tensors.size());

        for (const auto& tensor : cuda_tensors) {
          tensors.push_back(to_xla(tensor));
        }
        (*stack)[returns_begin + idx] =
            c10::IValue(c10::List<at::Tensor>(tensors));
      }
    }
  }
}

void xla_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  XLA_FN_TRACK(3);
  const auto name = c10::toString(op.operator_name());

  // Manually applying the XLA_COUNTER macro.
  // We need to do it ourselves and explicitly keep a mapping of counters
  // because this boxed fallback kernel is used by multiple operators,
  // and the macro stamps out a static Counter object with a fixed name
  // at the code location that it was called.
  if (_fallback_counters.find(name) == _fallback_counters.end()) {
    _fallback_counters[name] =
        new ::torch_xla::runtime::metrics::Counter(name);
  }
  _fallback_counters[name]->AddValue(1);

  auto& args = op.schema().arguments();
  auto arguments = torch::jit::last(stack, args.size());

  // Log each tensor argument.
  for (int64_t idx = 0; idx < arguments.size(); ++idx) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      TF_VLOG(3) << ivalue.toTensor().toString();
    }
  }

  if (UseCUDAFallback()) {
    cuda_fallback(op, stack, true);
  } else {
    // Call the actual boxed CPU fallback.
    // Set error_on_views as XLA should take care
    // of all view ops after functionalization.
    at::native::cpu_fallback(op, stack, true);
  }
}

TORCH_LIBRARY_IMPL(_, XLA, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&xla_fallback>());
}

}  // namespace torch_xla
