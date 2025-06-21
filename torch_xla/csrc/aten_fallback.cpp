#include "torch_xla/csrc/aten_fallback.h"

#include <ATen/DLConvertor.h>
#include <ATen/ops/_copy_from_and_resize.h>
#include <ATen/ops/_to_cpu.h>
#include <torch/csrc/utils/device_lazy_init.h>

#include <optional>
#include <unordered_map>
#include <vector>

#include "torch_xla/csrc/aten_cuda_functions.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/dl_convertor.h"
#include "torch_xla/csrc/function_call_tracker.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/metrics.h"
#include "torch_xla/csrc/runtime/runtime.h"
#include "torch_xla/csrc/runtime/tf_logging.h"
#include "torch_xla/csrc/xla_graph_executor.h"

namespace torch_xla {

// List of operations that should be fallbacked to CPU instead of GPU.
static std::unordered_set<std::string> _force_fallback_on_cpu{
    // This operation is a simple memory access that transforms the given
    // 1-element tensor into a Scalar.
    //
    // Although it makes sense to run this operation on CPU (since the
    // output will get copied back to CPU anyway), this also fixes a
    // particular issue with moco benchmark.
    // More details: https://github.com/pytorch/xla/issues/7647
    "aten::_local_scalar_dense",
};

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

// Most of the functions for the CUDA fallback are a modified version of
// PyTorch's at::native::cpu_fallback function.
//
// Source: aten/src/ATen/native/CPUFallback.cpp
//
// While a better solution would be to adapt PyTorch's function to be device
// agnostic, the changes are not small enough that would make sense for adding
// just one more device. Therefore, we copied the needed functions in this file.
//
// Before each modified function below, we shall specify what has changed,
// if there was any.

// Decide whether to run OpenXLA fallback operations on CUDA.
bool UseOpenXLAFallbackOnCUDA(const c10::OperatorHandle& op) {
  // In order to run OpenXLA fallback operations on CUDA, the conditions below
  // must be true:

  //   1. XLA_FALLBACK_CPU environment variable is NOT set
  bool dont_fallback_on_cpu =
      !runtime::sys_util::GetEnvBool("XLA_FALLBACK_CPU", false);

  //   2. The current ComputationClient DeviceType is CUDA. Basically, we don't
  //      support running OpenXLA fallback operations on CUDA if the current
  //      PyTorch/XLA DeviceType is not CUDA.
  bool device_is_cuda =
      runtime::GetComputationClientOrDie()->GetDeviceType().getType() ==
      XlaDeviceType::CUDA;

  //   3. PyTorch must have been compiled with CUDA support. Otherwise, our
  //      phony implementation in aten_cuda_functions.cpp will return 0 for the
  //      call below.
  bool pytorch_device_is_not_zero = c10::cuda::device_count() > 0;

  //   4. There is a kernel registered for the CUDA dispatch key, for this
  //      operation.
  bool has_cuda_kernel = op.hasKernelForDispatchKey(c10::DispatchKey::CUDA);

  //   5. The operation is not in the set of operations that should be forcefuly
  //      fallbacked on CPU.
  bool dont_force_fallback_on_cpu =
      _force_fallback_on_cpu.find(c10::toString(op.operator_name())) ==
      _force_fallback_on_cpu.end();

  return dont_fallback_on_cpu && device_is_cuda && pytorch_device_is_not_zero &&
         has_cuda_kernel && dont_force_fallback_on_cpu;
}

struct DeviceInfo {
  DeviceInfo(c10::Device device, c10::DeviceIndex i = -1)
      : common_device(device), index(i) {}

  // Synchronizes the CUDA device being used by PyTorch.
  void synchronize() {
    TORCH_CHECK(index != -1, "No defined XLA tensors found for CUDA fallback.");
    // Save the current PyTorch device, in case it's not the same as the
    // recorded tensor device.
    c10::DeviceIndex current = c10::cuda::current_device();
    c10::cuda::set_device(index);
    c10::cuda::device_synchronize();
    c10::cuda::set_device(current);
  }

  // Common device for all XLA tensors.
  //
  // CUDA OpenXLA fallback is supported only when all XLA tensors live in
  // the same XLA device. This field should be updated and checked every
  // time we convert an XLA tensor argument into a CUDA tensor.
  c10::Device common_device;

  // CUDA device index where the tensors live in.
  //
  // This is used for synchronizing the device where the fallback operation
  // was called. This should ensure completion of the CUDA computation, in
  // order to be used by another XLA computation.
  c10::DeviceIndex index;
};

// Change: use of std::any_of instead of iterating with a for-loop.
static bool validate_tensor_list(const c10::List<at::Tensor>& tensorlist) {
  return std::any_of(tensorlist.begin(), tensorlist.end(),
                     [](const at::Tensor& tensor) { return tensor.defined(); });
}

// Retrieve the inner XLATensorPtr, and check it lives inside CUDA.
static XLATensorPtr get_xla_cuda_tensor(const at::Tensor& tensor) {
  XLATensorPtr xla_tensor = bridge::GetXlaTensor(tensor);
  const torch::lazy::BackendDevice& device = xla_tensor->GetDevice();
  TORCH_CHECK(device.type() == static_cast<int8_t>(XlaDeviceType::CUDA),
              "OpenXLA CUDA fallback only supports XLA:CUDA tensors. Found a "
              "tensor of another device: ",
              device.toString());
  return xla_tensor;
}

static bool is_valid_xla_tensor(const at::Tensor& tensor) {
  return tensor.defined() && tensor.is_xla();
}

static at::Tensor to_cuda_tensor(const at::Tensor& tensor,
                                 std::optional<DeviceInfo>& info) {
  // Skip undefined or non-XLA tensors.
  if (!is_valid_xla_tensor(tensor)) {
    return tensor;
  }

  // Grab the DLManagedTensor.
  DLManagedTensor* managed = torch_xla::toDLPack(tensor);
  c10::DeviceIndex index = managed->dl_tensor.device.device_id;

  if (info.has_value()) {
    TORCH_CHECK(info->common_device == tensor.device() && info->index == index,
                "fallback supports only single XLA device.");
  } else {
    info = std::make_optional(DeviceInfo(tensor.device(), index));
  }

  // Create the CUDA tensor.
  return at::fromDLPack(managed, [=](void*) { managed->deleter(managed); });
}

// Former 'to_cpu'.
// In order to move tensors from XLA to CUDA, we make use of the DLPack API.
//
//   1. Synchronize the XLA tensors, so that we can access their data pointer
//   2. Use DLPack in order to create a CUDA tensor
static std::vector<at::Tensor> to_cuda(const at::TensorList& tensors,
                                       std::optional<DeviceInfo>& info) {
  // Synchronize tensors, so that we are able to grab their data pointer.
  std::vector<XLATensorPtr> xla_tensors;
  for (auto& tensor : tensors) {
    if (is_valid_xla_tensor(tensor)) {
      xla_tensors.push_back(get_xla_cuda_tensor(tensor));
    }
  }
  XLAGraphExecutor::Get()->SyncTensorsGraph(
      &xla_tensors, /*devices=*/{}, /*wait=*/true, /*sync_ltc_data=*/true,
      /*warm_up_cache_only=*/false);

  // Use DLPack for sharing the XLA storage with a newly created CUDA tensor.
  std::vector<at::Tensor> cuda_tensors(tensors.size());
  std::transform(
      tensors.begin(), tensors.end(), cuda_tensors.begin(),
      [&](const at::Tensor& tensor) { return to_cuda_tensor(tensor, info); });
  return cuda_tensors;
}

// Copy back the results from CUDA to XLA.
// Assumes that we have already synchronized CUDA.
static at::Tensor to_xla(const at::Tensor& tensor) {
  return torch_xla::fromDLPack(at::toDLPack(tensor));
}

// Former 'cpu_fallback'.
// Changes:
//
//   1. Track the device index being used. Rationale: we synchronize the device
//      before crossing device borders for correctness.
//
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

  std::vector<c10::IValue> tensorlist_cuda_args;

  // This fallback only works if all XLA:CUDA tensor arguments are
  // on the same CUDA device.
  //
  // We keep track of said device, so that after actually running
  // the operation on PyTorch CUDA eager-mode, we synchronize the
  // device.
  //
  // This variable is updated over the course of 'to_cuda' calls.
  std::optional<DeviceInfo> info;

  // Initialize CUDA device.
  torch::utils::device_lazy_init(at::kCUDA);

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
      auto cuda_ivalue = c10::IValue(
          c10::List<at::Tensor>(to_cuda(ivalue.toTensorList().vec(), info)));
      tensorlist_cuda_args.push_back(cuda_ivalue);
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
      auto cuda_tensors = to_cuda(need_convert_tensors, info);
      for (const auto i : c10::irange(need_convert_tensors_index.size())) {
        auto idx = need_convert_tensors_index[i];
        opt_tensors[idx] = cuda_tensors[i];
      }
      (*stack)[arguments_begin + idx] = c10::IValue(opt_tensors);
    } else if (ivalue.isDevice()) {
      c10::Device device = ivalue.toDevice();
      if (info.has_value()) {
        TORCH_CHECK(info->common_device == device, "XLA tensors live in ",
                    info->common_device, " but found target device: ", device);
      } else {
        info->common_device = device;
      }
      (*stack)[arguments_begin + idx] = c10::IValue(c10::Device(at::kCUDA));
    }
  }
  // XLA requires all of the tensor arguments to be gathered up and converted to
  // CUDA together.
  auto cuda_tensors = to_cuda(tensor_args, info);

  for (const auto i : c10::irange(tensor_args_indices.size())) {
    auto idx = tensor_args_indices[i];
    (*stack)[arguments_begin + idx] = c10::IValue(cuda_tensors[i]);
  }

  // Step 2: Call the underlying CUDA implementation of the operator
  op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CUDA), stack);

  // Synchronize the device before actually converting back to XLA.
  TORCH_CHECK(info.has_value());
  info->synchronize();

  // Step 3: We need to take special care to handle mutable aliases properly:
  // If any input tensors are mutable aliases, we need to
  // directly copy the updated data on the CUDA tensors back to the original
  // inputs.
  for (const auto i : c10::irange(tensor_args_indices.size())) {
    auto tensor_idx = tensor_args_indices[i];
    const c10::AliasInfo* alias_info = schema_args[tensor_idx].alias_info();
    if (alias_info != nullptr && alias_info->isWrite()) {
      at::_copy_from_and_resize(cuda_tensors[i], tensor_args[i]);
    }
  }

  // We also need to explicit reapply input mutations to inputs that are lists
  // of tensors
  for (const auto i : c10::irange(tensorlist_args_indices.size())) {
    auto tensorlist_idx = tensorlist_args_indices[i];
    const c10::AliasInfo* alias_info = schema_args[tensorlist_idx].alias_info();
    if (alias_info != nullptr && alias_info->isWrite()) {
      const auto& cuda_tensors = tensorlist_cuda_args[i].toTensorList().vec();
      for (const auto idx : c10::irange(tensorlist_args[i].size())) {
        at::_copy_from_and_resize(cuda_tensors[idx], tensorlist_args[i][idx]);
      }
    }
  }

  // Step 4: Convert any CUDA output tensors back to the original input device.
  // For mutable alias'd outputs, we also need to take special care
  // to move the ORIGINAL input tensor back onto the stack, in place of
  // the temporary CUDA output tensor that we created.
  //
  // See [CPU Fallback Does Not Handle View Operators]
  const auto& schema_returns = op.schema().returns();
  const auto& num_returns = schema_returns.size();
  auto returns = torch::jit::last(stack, num_returns);
  const auto returns_begin = stack->size() - num_returns;

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
    _fallback_counters[name] = new ::torch_xla::runtime::metrics::Counter(name);
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

  if (UseOpenXLAFallbackOnCUDA(op)) {
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
