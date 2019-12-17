#pragma once

#include <ATen/Device.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>

#include <vector>

#include "tensorflow/core/lib/gtl/array_slice.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/tensor.h"

namespace torch_xla {
namespace bridge {

c10::optional<XLATensor> TryGetXlaTensor(const at::Tensor& tensor);

// Extracts the XLATensor out of our version of at::Tensor. Throws an exception
// if tensor is not an XLA tensor.
XLATensor GetXlaTensor(const at::Tensor& tensor);

// Replaces the XLA tensor embedded within the XLA TensorImpl with the new
// version.
void ReplaceXlaTensor(const at::Tensor& tensor, XLATensor new_xla_tensor);

// Same as above, applied to a list of tensors.
std::vector<XLATensor> GetXlaTensors(
    tensorflow::gtl::ArraySlice<const at::Tensor> tensors);

// If tensor is an XLA tensor type, returns the XLATensor embedded within it,
// otherwise creates a new XLA tensor type with tensor as data.
XLATensor GetOrCreateXlaTensor(const at::Tensor& tensor, const Device& device);

// Creates a vector of at::Tensor objects extracted from a list of XLA tensors.
std::vector<at::Tensor> XlaCreateTensorList(const at::TensorList& tensors);

void XlaUpdateTensors(
    tensorflow::gtl::ArraySlice<const at::Tensor> dest_xla_tensors,
    tensorflow::gtl::ArraySlice<const at::Tensor> source_cpu_tensors,
    tensorflow::gtl::ArraySlice<const size_t> indices);

// Tries to extract the device out of the XLA tensor. Returns nullopt if the
// input is not an XLA tensor.
c10::optional<Device> GetXlaDevice(const at::Tensor& tensor);

c10::optional<Device> GetXlaDevice(const at::TensorList& tensors);

c10::optional<Device> GetXlaDevice(const at::TensorOptions& tensor_options);

c10::optional<Device> GetXlaDevice(const c10::Device& device);

Device AtenDeviceToXlaDevice(const c10::Device& device);

c10::Device XlaDeviceToAtenDevice(const Device& device);

std::string ToXlaString(const c10::Device& device);

c10::Device AtenDefaultDevice();

c10::Device SetCurrentDevice(const c10::Device& device);

Device SetCurrentDevice(const Device& device);

c10::Device GetCurrentAtenDevice();

at::Tensor XlaToAtenTensor(XLATensor xla_tensor,
                           const at::TensorOptions& tensor_options);

// Creates an ATen tensor with XLA type id from an XLATensor.
at::Tensor AtenFromXlaTensor(XLATensor xla_tensor);

std::vector<at::Tensor> AtenFromXlaTensors(
    tensorflow::gtl::ArraySlice<const XLATensor> xla_tensors);

// Creates an XLA tensor holding the data in tensor, on the given device.
at::Tensor CreateXlaTensor(at::Tensor tensor,
                           const c10::optional<Device>& device);

// Given a vector of at::Tensor creates a vector of XLA tensors on the given
// device.
std::vector<at::Tensor> CreateXlaTensors(const std::vector<at::Tensor>& tensors,
                                         const c10::optional<Device>& device);

}  // namespace bridge
}  // namespace torch_xla
