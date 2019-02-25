#pragma once

#include <vector>

#include <ATen/Device.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/Type.h>

#include "tensorflow/core/lib/gtl/array_slice.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/tensor.h"

namespace torch_xla {
namespace bridge {

// Extracts the XLATensor out of our version of at::Tensor. Throws an exception
// if tensor is not an XLA tensor.
XLATensor GetXlaTensor(const at::Tensor& tensor);

// Same as above, applied to a list of tensors.
std::vector<XLATensor> GetXlaTensors(
    tensorflow::gtl::ArraySlice<const at::Tensor> tensors);

// Like GetXlaTensor(), but if tensor is a variable, unwraps it and access the
// underline tensor.
XLATensor GetXlaTensorUnwrap(const at::Tensor& tensor);

// If tensor is an XLA tensor type, returns the XLATensor embedded within it,
// otherwise creates a new XLA tensor type with tensor as data.
XLATensor GetOrCreateXlaTensor(const at::Tensor& tensor, const Device& device);

// Creates a vector of at::Tensor objects extracted from a list of XLA tensors.
// If the writeable vector is not nullptr, it must be the same size as tensors,
// and the corresponding bool tells whether the ATEN tensor to be retrieved
// should the a writeable copy.
std::vector<at::Tensor> XlaCreateTensorList(const at::TensorList& tensors,
                                            const std::vector<bool>* writeable);

// Creates an at::Tensor out of an XLA tensor, but making the XLA tensor to
// discard any device side data. Throws if tensor is not an XLA tensor.
at::Tensor XlaToAtenMutableTensor(const at::Tensor& tensor);

// Extracts the device out of the XLA tensor. Throws an exception if tensor is
// not an XLA tensor.
c10::optional<Device> GetXlaDevice(const at::Tensor& tensor);

c10::optional<Device> GetXlaDevice(const at::TensorList& tensors);

c10::optional<Device> GetXlaDevice(const at::TensorOptions& tensor_options);

c10::optional<Device> GetXlaDevice(const c10::Device& device);

Device AtenDeviceToXlaDevice(const c10::Device& device);

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
