#pragma once

#include <string>
#include <vector>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "torch/csrc/autograd/variable.h"
#include "torch_xla/csrc/device.h"

namespace torch_xla {
namespace detail {

// Checks whether BF16 should be used as default floating point type for XLA
// computations.
bool UseBF16();

}  // namespace detail

std::vector<xla::int64> ComputeShapeStrides(const xla::Shape& shape);

// Create an XLA shape with the given dimensions and type, suitable to be used
// in the specified device type. The type of device can affect the choice of the
// XLA layout.
xla::Shape MakeArrayShapeFromDimensions(
    tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
    xla::PrimitiveType type, DeviceType device_type);

xla::Shape MakeArrayShapeFromDimensions(const at::IntList& dimensions,
                                        xla::PrimitiveType type,
                                        DeviceType device_type);

// Converts an XLA literal to an at::Tensor of the given element type.
at::Tensor MakeTensorFromXlaLiteral(const xla::Literal& literal,
                                    at::ScalarType dest_element_type);

// Uploads an ATEN tensor data to the device and fetches the corresponding
// device data handle.
xla::ComputationClient::DataPtr TensorToXlaData(const at::Tensor& tensor,
                                                const Device& device);

// Retrieves the device data handles by parallel uploading data onto the
// corresponding devices.
std::vector<xla::ComputationClient::DataPtr> CreateTensorsData(
    const std::vector<at::Tensor>& tensors,
    const std::vector<std::string>& devices);

// Creates an XLA literal out of an ATEN tensor. If shape is specified, that
// shape+layout will be used, otherwise one will be generated out of the ATEN
// tensor shape. The device argument (can be nullptr for the default device)
// tells the API that the created Literal will be sent to such device.
xla::Literal GetTensorLiteral(const at::Tensor& tensor, const xla::Shape* shape,
                              const Device* device);

// If "shape" is a tuple, return the element shapes, otherwise return a
// singleton list containing the original shape.
std::vector<xla::Shape> GetComponentShapes(const xla::Shape& shape);

// Create a shape with "device_type" compatible layout from the given "shape".
xla::Shape MakeShapeWithDeviceLayout(const xla::Shape& shape,
                                     DeviceType device_type);

// Create the XLA shape to be used within a lowered XLA computation, to
// represent a given tensor data.
xla::Shape CreateComputationShapeFromTensor(const at::Tensor& tensor,
                                            const Device* device);

at::ScalarType TensorTypeFromXlaType(xla::PrimitiveType xla_type);

// Maps an XLA type to the one which can be used on the given device (or the
// default device, id device is nullptr).
xla::PrimitiveType GetDevicePrimitiveType(xla::PrimitiveType type,
                                          const Device* device);

// Converts the given scalar type to an XLA primitive type.
xla::PrimitiveType MakeXlaPrimitiveType(at::ScalarType scalar_type,
                                        const Device* device);

}  // namespace torch_xla
