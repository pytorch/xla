#pragma once

#include <vector>

#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/Type.h>

#include "device.h"
#include "tensor.h"

namespace torch_xla {
namespace bridge {

// Helper function which creates an empty CPU ATEN tensor.
at::Tensor CreateEmptyTensor(at::IntList size,
                             const at::TensorOptions& options);

// Extracts the XLATensor out of our version of at::Tensor. Throws an exception
// if tensor is not an XLA tensor.
XLATensor& GetXlaTensor(const at::Tensor& tensor);

// Creates a vector of at::Tensor objects extracted from a list of XLA tensors.
// If the writeable vector is not nullptr, it must be the same size as tensors,
// and the corresponding bool tells whether the ATEN tensor to be retrieved
// should the a writeable copy.
std::vector<at::Tensor> XlaCreateTensorList(const at::TensorList& tensors,
                                            const std::vector<bool>* writeable);

// Creates an at::Tensor out of an XLA tensor. Throws if tensor is not an XLA
// tensor.
at::Tensor XlaToAtenTensor(const at::Tensor& tensor);

// Creates an at::Tensor out of an XLA tensor, but making the XLA tensor to
// discard any device side data. Throws if tensor is not an XLA tensor.
at::Tensor XlaToAtenMutableTensor(const at::Tensor& tensor);

// Given a vector of at::Tensor creates a vector of XLA tensors on the given
// device.
std::vector<at::Tensor> CreateXlaTensors(const std::vector<at::Tensor>& tensors,
                                         const Device& device);

// Extracts the device out of the XLA tensor. Throws an exception if tensor is
// not an XLA tensor.
Device XlaTensorDevice(const at::Tensor& tensor);

static inline Device XlaTensorDevice(const at::TensorList& tensors) {
  return XlaTensorDevice(tensors.at(0));
}

Device XlaTensorDevice(const at::TensorOptions& tensor_options);

// Creates an ATen tensor with XLA type id from an XLATensor.
at::Tensor AtenFromXlaTensor(XLATensor xla_tensor);

// Creates an XLA tensor holding the data in tensor, on the given device.
at::Tensor CreateXlaTensor(const at::Tensor& tensor, const Device& device);

}  // namespace bridge
}  // namespace torch_xla
