#pragma once

#include <vector>

#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/Type.h>

#include "device.h"
#include "tensor.h"

namespace torch_xla {
namespace bridge {

// Extracts the XLATensor out of our version of at::Tensor. Throws an exception
// if tensor is not an XLA tensor.
XLATensor& GetXlaTensor(const at::Tensor& tensor);

// Creates a vector of at::Tensor objects extracted from a list of XLA tensors.
std::vector<at::Tensor> XlaCreateTensorList(const at::TensorList& tensors);

// Creates an at::Tensor out of an XLA tensor. Throws if tensor is not an XLA
// tensor.
at::Tensor XlaToAtenTensor(const at::Tensor& tensor);

// Creates an at::Tensor out of an XLA tensor, but making the XLA tensor to
// discard any device side data. Throws if tensor is not an XLA tensor.
at::Tensor XlaToAtenMutableTensor(const at::Tensor& tensor);

// Given a vector of at::Tensor creates a vector of XLA tensors on the given
// device.
std::vector<at::Tensor> CreateXlaTensors(
    const std::vector<at::Tensor>& tensors, const Device& device);

// Extracts the device out of the XLA tensor. Throws an exception if tensor is
// not an XLA tensor.
Device XlaTensorDevice(const at::Tensor& tensor);

// Creates an XLA tensor holding the data in tensor, on the given device.
at::Tensor CreateXlaTensor(const at::Tensor& tensor, const Device& device);

}  // namespace bridge
}  // namespace torch_xla
