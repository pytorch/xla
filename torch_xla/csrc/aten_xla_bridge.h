#ifndef XLA_TORCH_XLA_CSRC_ATEN_XLA_BRIDGE_H_
#define XLA_TORCH_XLA_CSRC_ATEN_XLA_BRIDGE_H_

#include <ATen/Device.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>

#include <vector>

#include "absl/types/span.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/tensor.h"

namespace torch_xla {
namespace bridge {

XLATensorPtr TryGetXlaTensor(const at::Tensor& tensor);

bool IsXlaTensor(const at::Tensor& tensor);

// Extracts the XLATensorPtr out of our version of at::Tensor. Throws an
// exception if tensor is not an XLA tensor.
XLATensorPtr GetXlaTensor(const at::Tensor& tensor);

// Replaces the XLA tensor embedded within the XLA TensorImpl with the new
// version.
void ReplaceXlaTensor(const at::Tensor& tensor, XLATensorPtr new_xla_tensor);

// Same as above, applied to a list of tensors.
std::vector<XLATensorPtr> GetXlaTensors(const at::ITensorListRef& tensors);

torch_xla::XLATensorPtr GetXlaTensorOrCreateForWrappedNumber(
    const at::Tensor& tensor, const torch::lazy::BackendDevice& device);

// If tensor is an XLA tensor type, returns the XLATensorPtr embedded within it,
// otherwise creates a new XLA tensor type with tensor as data.
XLATensorPtr GetOrCreateXlaTensor(const at::Tensor& tensor,
                                  const torch::lazy::BackendDevice& device);

XLATensorPtr GetOrCreateXlaTensor(const c10::optional<at::Tensor>& tensor,
                                  const torch::lazy::BackendDevice& device);

std::vector<XLATensorPtr> GetOrCreateXlaTensors(
    absl::Span<const at::Tensor> tensors,
    const torch::lazy::BackendDevice& device);

// Creates a vector of at::Tensor objects extracted from a list of XLA tensors.
std::vector<at::Tensor> XlaCreateTensorList(const at::ITensorListRef& tensors);

// Creates a vector of c10::optional<at::Tensor> objects extracted from a list
// of optional XLA tensors.
std::vector<c10::optional<at::Tensor>> XlaCreateOptTensorList(
    const std::vector<c10::optional<at::Tensor>>& tensors);

void XlaUpdateTensors(absl::Span<const at::Tensor> dest_xla_tensors,
                      absl::Span<const at::Tensor> source_cpu_tensors,
                      absl::Span<const size_t> indices);

// Tries to extract the device out of the XLA tensor. Returns nullopt if the
// input is not an XLA tensor.
c10::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const at::Tensor& tensor);

c10::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const c10::optional<at::Tensor>& tensor);

c10::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const at::ITensorListRef& tensors);

c10::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const std::vector<at::Tensor>& tensors);

c10::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const at::TensorOptions& tensor_options);

c10::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const c10::Device& device);

c10::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const c10::optional<c10::Device>& device = c10::nullopt);

std::vector<torch::lazy::BackendDevice> GetBackendDevices();

torch::lazy::BackendDevice AtenDeviceToXlaDevice(const c10::Device& device);

c10::Device XlaDeviceToAtenDevice(const torch::lazy::BackendDevice& device);

std::string ToXlaString(const c10::Device& device);

c10::Device AtenDefaultDevice();

c10::Device SetCurrentDevice(const c10::Device& device);

torch::lazy::BackendDevice SetCurrentDevice(
    const torch::lazy::BackendDevice& device);

c10::Device GetCurrentAtenDevice();

at::Tensor XlaToAtenTensor(XLATensorPtr xla_tensor,
                           const at::TensorOptions& tensor_options);

// Creates an ATen tensor with XLA type id from an XLATensorPtr.
at::Tensor AtenFromXlaTensor(XLATensorPtr xla_tensor);

std::vector<at::Tensor> AtenFromXlaTensors(
    absl::Span<const XLATensorPtr> xla_tensors);

// Creates an XLA tensor holding the data in tensor, on the given device.
at::Tensor CreateXlaTensor(
    at::Tensor tensor, const c10::optional<torch::lazy::BackendDevice>& device);

// Given a vector of at::Tensor creates a vector of XLA tensors on the given
// device.
std::vector<at::Tensor> CreateXlaTensors(
    const std::vector<at::Tensor>& tensors,
    const c10::optional<torch::lazy::BackendDevice>& device);

template <typename T, typename... Args>
c10::optional<torch::lazy::BackendDevice> GetXlaDevice(
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

}  // namespace bridge
}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_ATEN_XLA_BRIDGE_H_