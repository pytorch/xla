#pragma once

#include <string>
#include <vector>

#include "device.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "torch/csrc/autograd/variable.h"

namespace torch_xla {

// Create an XLA shape with the given dimensions and type, suitable to be used
// in the specified device type. The type of device can affect the choice of the
// XLA layout.
xla::Shape MakeArrayShapeFromDimensions(const at::IntList& tensor_dimensions,
                                        xla::PrimitiveType type,
                                        DeviceType device_type);

// Converts an ATEN tensor to an XLA literal.
at::Tensor MakeTensorFromXlaLiteral(const xla::Literal& literal);

// Uploads an ATEN tensor data to the device and fetches the corresponding
// device data handle.
std::shared_ptr<xla::ComputationClient::Data> TensorToXlaData(
    const at::Tensor& tensor, const Device& device);

// Retrieves the device data handles by parallel uploading data onto the
// corresponding devices.
std::vector<std::shared_ptr<xla::ComputationClient::Data>> CreateTensorsData(
    const std::vector<at::Tensor>& tensors,
    const std::vector<std::string>& devices);

// Creates an XLA literal out of an ATEN tensor. If shape is specified, that
// shape+layout will be used, otherwise one will be generated out of the ATEN
// tensor shape.
xla::Literal GetTensorLiteral(const at::Tensor& tensor,
                              const xla::Shape* shape);

// If "shape" is a tuple, return the element shapes, otherwise return a
// singleton list containing the original shape.
std::vector<xla::Shape> GetComponentShapes(const xla::Shape& shape);

// Create a shape with "device_type" compatible layout from the given "shape".
xla::Shape MakeShapeWithDeviceLayout(const xla::Shape& shape,
                                     DeviceType device_type);

}  // namespace torch_xla
