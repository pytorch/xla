#pragma once

#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/lazy/core/hash.h"
#include "torch_xla/csrc/device.h"

namespace torch_xla {

std::vector<int64_t> ComputeShapeStrides(const xla::Shape& shape);

// Converts an XLA literal to an at::Tensor of the given element type.
at::Tensor MakeTensorFromXlaLiteral(const xla::Literal& literal,
                                    at::ScalarType dest_element_type);

// TODO LTC @wonjoo - Migrate to upstream after Device -> BackendDevice
std::vector<at::Tensor> XlaDataToTensors(
    absl::Span<const xla::ComputationClient::DataPtr> xla_data,
    at::ScalarType dest_element_type);

bool TensorCompare(const at::Tensor& t1, const at::Tensor& t2);

// Uploads an ATEN tensor data to the device and fetches the corresponding
// device data handle.
// TODO LTC @wonjoo - Migrate to upstream after Device -> BackendDevice
xla::ComputationClient::DataPtr TensorToXlaData(const at::Tensor& tensor,
                                                const torch::lazy::BackendDevice& device,
                                                bool transfer_async = false);

torch::lazy::hash_t TensorHash(const at::Tensor& tensor);

// Retrieves the device data handles by parallel uploading data onto the
// corresponding devices.
// TODO LTC @wonjoo - Migrate to upstream after Device -> BackendDevice
std::vector<xla::ComputationClient::DataPtr> CreateTensorsData(
    const std::vector<at::Tensor>& tensors,
    const std::vector<std::string>& devices, bool transfer_async = false);

// Creates an XLA literal out of an ATEN tensor. If shape is specified, that
// shape+layout will be used, otherwise one will be generated out of the ATEN
// tensor shape. The device argument (can be nullptr for the default device)
// tells the API that the created Literal will be sent to such device.
xla::Literal GetTensorLiteral(const at::Tensor& tensor, const xla::Shape* shape,
                              const torch::lazy::BackendDevice* device);

// If "shape" is a tuple, return the element shapes, otherwise return a
// singleton list containing the original shape.
std::vector<xla::Shape> GetComponentShapes(const xla::Shape& shape);

// Create a shape with "device_type" compatible layout from the given "shape".
xla::Shape MakeShapeWithDeviceLayout(const xla::Shape& shape,
                                     XlaDeviceType hw_type);

// Create the XLA shape to be used within a lowered XLA computation, to
// represent a given tensor data.
xla::Shape CreateComputationShapeFromTensor(const at::Tensor& tensor,
                                            const torch::lazy::BackendDevice* device);

at::ScalarType TensorTypeFromXlaType(xla::PrimitiveType xla_type);

xla::PrimitiveType TensorTypeToRawXlaType(at::ScalarType scalar_type);

// Maps an XLA type to the one which can be used on the given device (or the
// default device, id device is nullptr).
xla::PrimitiveType GetDevicePrimitiveType(xla::PrimitiveType type,
                                          const torch::lazy::BackendDevice* device);

// Converts the given scalar type to an XLA primitive type.
xla::PrimitiveType MakeXlaPrimitiveType(at::ScalarType scalar_type,
                                        const torch::lazy::BackendDevice* device);

bool RequiresRawTypeCasting(at::ScalarType scalar_type, const torch::lazy::BackendDevice* device);

xla::PrimitiveType GetShapeDimensionType(const torch::lazy::BackendDevice* device);

}  // namespace torch_xla
