#ifndef XLA_TORCH_XLA_CSRC_ATEN_XLA_BRIDGE_H_
#define XLA_TORCH_XLA_CSRC_ATEN_XLA_BRIDGE_H_

#include <vector>

#include <ATen/Device.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/tensor.h"

namespace torch_xla {
namespace bridge {

// TODO(ysiraichi): remove this function once codegen does not need it.
//
// We still need this function because lazy codegen needs a function that
// returns a value of type `T`, which can be:
//
//   1. cast to boolean; and
//   2. accessed with "->"
//
// e.g. pointers and optional types
//
// A StatusOr type fulfills only (2), so we can't use it there. In order
// to do so, we have to change upstream accordingly.
//
[[deprecated(
    "Use GetXlaTensor(), instead. "
    "This function returns an null-initialized `XLATensorPtr`, instead of "
    "propagating errors with StatusOr values.")]]  //
XLATensorPtr
TryGetXlaTensor(const at::Tensor& tensor);

// Retrieves the underlying `XLATensorPtr` from `tensor`.
//
// This function does the following things in order to retrieve
// (if exists) the underlying `XLATensorPtr`:
//
//   1. Synchronizes the tensor, if it's a tensor wrapped in a functional tensor
//   2. Retrieves the inner `XLATensorImpl` instance
//   3. Finally, retrieves the `XLATensor` that lives inside `XLATensorImpl`
//
// An error might ocurr if, after unwrapping the wrapper functional tensor
// (if exists), the `TensorImpl` of the unwrapped tensor is not a
// `XLATensorImpl`. This might happen if:
//
//   1. `tensor` lives in another device
//   2. `tensor` wasn't created within this project
//      (e.g. meta tensors whose device is XLA)
//
absl::StatusOr<absl_nonnull XLATensorPtr> GetXlaTensor(
    const at::Tensor& tensor);

// Same as above, applied to a list of tensors.
absl::StatusOr<std::vector<absl_nonnull XLATensorPtr>> GetXlaTensors(
    const at::ITensorListRef& tensors);

// Retrieves the underlying `XLATensorPtr` from `tensor`.
//
// If `tensor` is not an actual XLA tensor, this function will craft a
// specialized error message for PyTorch operations or Python API
// functions, i.e. functions where the parameter name makes sense for
// the end user.
absl::StatusOr<absl_nonnull XLATensorPtr> GetInputXlaTensor(
    const at::Tensor& tensor, std::string_view param);

bool IsXlaTensor(const at::Tensor& tensor);

// Replaces the XLA tensor embedded within `tensor`'s XLA TensorImpl with
// `new_xla_tensor`.
//
// Fails if `tensor` is not an XLA tensor.
absl::Status ReplaceXlaTensor(const at::Tensor& tensor,
                              XLATensorPtr new_xla_tensor);

// Replaces the XLA tensor embedded within the `tensors` XLA TensorImpl
// with `new_xla_tensors`.
//
// Fails if any of `tensors` is not an XLA tensor, or if the number of `tensors`
// does not match the number of `new_xla_tensors`.
absl::Status ReplaceXlaTensor(const std::vector<at::Tensor>& tensors,
                              const std::vector<XLATensorPtr> new_xla_tensors);

torch_xla::XLATensorPtr GetXlaTensorOrCreateForWrappedNumber(
    const at::Tensor& tensor, const torch::lazy::BackendDevice& device);

// If tensor is an XLA tensor type, returns the XLATensorPtr embedded within it,
// otherwise creates a new XLA tensor type with tensor as data.
XLATensorPtr GetOrCreateXlaTensor(const at::Tensor& tensor,
                                  const torch::lazy::BackendDevice& device);

XLATensorPtr GetOrCreateXlaTensor(const std::optional<at::Tensor>& tensor,
                                  const torch::lazy::BackendDevice& device);

std::vector<XLATensorPtr> GetOrCreateXlaTensors(
    absl::Span<const at::Tensor> tensors,
    const torch::lazy::BackendDevice& device);

// Creates a vector of at::Tensor objects extracted from a list of XLA tensors.
std::vector<at::Tensor> XlaCreateTensorList(const at::ITensorListRef& tensors);

// Creates a vector of std::optional<at::Tensor> objects extracted from a list
// of optional XLA tensors.
std::vector<std::optional<at::Tensor>> XlaCreateOptTensorList(
    const std::vector<std::optional<at::Tensor>>& tensors);

void XlaUpdateTensors(absl::Span<const at::Tensor> dest_xla_tensors,
                      absl::Span<const at::Tensor> source_cpu_tensors,
                      absl::Span<const size_t> indices);

// Tries to extract the device out of the XLA tensor. Returns nullopt if the
// input is not an XLA tensor.
std::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const at::Tensor& tensor);

std::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const std::optional<at::Tensor>& tensor);

std::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const at::ITensorListRef& tensors);

std::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const std::vector<at::Tensor>& tensors);

std::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const at::TensorOptions& tensor_options);

std::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const c10::Device& device);

std::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const std::optional<c10::Device>& device = std::nullopt);

std::vector<torch::lazy::BackendDevice> GetBackendDevices();

torch::lazy::BackendDevice AtenDeviceToXlaDevice(const c10::Device& device);

c10::Device XlaDeviceToAtenDevice(const torch::lazy::BackendDevice& device);

std::string ToXlaString(const c10::Device& device);

// Returns the default `BackendDevice`.
// This function returns an error if the `ComputationClient` wasn't correctly
// initialized.
const absl::StatusOr<torch::lazy::BackendDevice * absl_nonnull>&
GetDefaultDevice();

c10::Device AtenDefaultDevice();

torch::lazy::BackendDevice GetCurrentDevice();

c10::Device GetCurrentAtenDevice();

static inline torch::lazy::BackendDevice GetDeviceOrCurrent(
    const torch::lazy::BackendDevice* device) {
  return device != nullptr ? *device : GetCurrentDevice();
}

c10::Device SetCurrentDevice(const c10::Device& device);

torch::lazy::BackendDevice SetCurrentDevice(
    const torch::lazy::BackendDevice& device);

at::Tensor XlaToAtenTensor(XLATensorPtr xla_tensor,
                           const at::TensorOptions& tensor_options);

// Creates an ATen tensor with XLA type id from an XLATensorPtr.
at::Tensor AtenFromXlaTensor(XLATensorPtr xla_tensor,
                             bool skip_functionalization = false);

std::vector<at::Tensor> AtenFromXlaTensors(
    absl::Span<const XLATensorPtr> xla_tensors);

// Creates an XLA tensor holding the data in tensor, on the given device.
at::Tensor CreateXlaTensor(
    at::Tensor tensor, const std::optional<torch::lazy::BackendDevice>& device);

// Given a vector of at::Tensor creates a vector of XLA tensors on the given
// device.
std::vector<at::Tensor> CreateXlaTensors(
    const std::vector<at::Tensor>& tensors,
    const std::optional<torch::lazy::BackendDevice>& device);

template <typename T, typename... Args>
std::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const T& tensor, const Args&... forward_tensors) {
  auto optional_device = GetXlaDevice(tensor);
  if (optional_device) {
    return optional_device;
  }
  return GetXlaDevice(forward_tensors...);
}

template <size_t... Indices>
auto TupleAtenFromXlaTensorsImpl(const std::vector<XLATensorPtr>& tensors,
                                 std::index_sequence<Indices...>) {
  return std::make_tuple(AtenFromXlaTensor(tensors[Indices])...);
}

template <size_t N>
auto TupleAtenFromXlaTensors(const std::vector<XLATensorPtr>& tensors) {
  return TupleAtenFromXlaTensorsImpl(tensors, std::make_index_sequence<N>{});
}

// Returns the deepest base tensor for a given tensor.
// If the base tensor is not defined, returns the tensor itself.
const at::Tensor& GetRootBase(const at::Tensor& tensor);
// Sets the base tensor of a given XLATensor. Convenient function
// to be used when returning tensors.
XLATensorPtr SetBaseTensor(XLATensorPtr tensor, const at::Tensor& base);

}  // namespace bridge
}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_ATEN_XLA_BRIDGE_H_
