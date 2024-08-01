#ifndef XLA_TORCH_XLA_CSRC_TENSOR_UTIL_H_
#define XLA_TORCH_XLA_CSRC_TENSOR_UTIL_H_

#include <ATen/ExpandUtils.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/lazy/core/hash.h>

#include <string>
#include <vector>

#include "absl/types/span.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/unwrap_data.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/types.h"

namespace torch_xla {

std::vector<int64_t> ComputeShapeStrides(const xla::Shape& shape);

// Converts an XLA literal to an at::Tensor of the given element type.
at::Tensor MakeTensorFromXlaLiteral(const xla::Literal& literal,
                                    at::ScalarType dest_element_type);

// Execution and data transfer are async in PJRT, so TransferFromDevice may
// block until `DataPtr`s are ready. Release the GIL so other threads can
// proceed and unblock any transfers or collective computations.
std::vector<xla::Literal> ReleaseGilAndTransferData(
    absl::Span<const torch::lazy::BackendDataPtr> xla_data);

// TODO LTC @wonjoo - Migrate to upstream after Device -> BackendDevice
std::vector<at::Tensor> XlaDataToTensors(
    absl::Span<const torch::lazy::BackendDataPtr> xla_data,
    absl::Span<const at::ScalarType> dest_element_type);

bool TensorCompare(const at::Tensor& t1, const at::Tensor& t2);

// Uploads an ATEN tensor data to the device and fetches the corresponding
// device data handle.
// TODO LTC @wonjoo - Migrate to upstream after Device -> BackendDevice
torch::lazy::BackendDataPtr TensorToXlaData(
    const at::Tensor& tensor, const torch::lazy::BackendDevice& device);

torch::lazy::hash_t TensorHash(const at::Tensor& tensor);

// Retrieves the device data handles by parallel uploading data onto the
// corresponding devices.
// TODO LTC @wonjoo - Migrate to upstream after Device -> BackendDevice
std::vector<torch::lazy::BackendDataPtr> CreateTensorsData(
    const std::vector<at::Tensor>& tensors,
    const std::vector<std::string>& devices);

// Shard and transfer tensors to devices using `PjRtComputationClient`.
// The client's data transfer to device is asynchronous.
std::vector<torch::lazy::BackendDataPtr> CreateTensorsData(
    const std::vector<at::Tensor>& tensors,
    const std::vector<XLATensor::ShardingSpecPtr>& sharding_specs,
    const std::vector<std::string>& devices);

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

// Copy the tensor's data into the destination buffer.
void PopulateTensorBuffer(const at::Tensor& tensor,
                          const xla::Shape& dest_shape, void* dest_buffer,
                          size_t dest_buffer_size,
                          const torch::lazy::BackendDevice& device);

// Create the XLA shape to be used within a lowered XLA computation, to
// represent a given tensor data.
xla::Shape CreateComputationShapeFromTensor(
    const at::Tensor& tensor, const torch::lazy::BackendDevice* device);

// Make a compatible dtype for the current device
xla::PrimitiveType GetXlaPrimitiveTypeForCurrentDevice(
    xla::PrimitiveType xla_type);

// Converts the given scalar type to an XLA primitive type.
xla::PrimitiveType MakeXlaPrimitiveType(
    at::ScalarType scalar_type, const torch::lazy::BackendDevice* device);

// Convert a lazy shape to device dependent xla shape.
xla::Shape MakeXlaShapeFromLazyShape(torch::lazy::Shape shape,
                                     const torch::lazy::BackendDevice& device);

bool RequiresRawTypeCasting(at::ScalarType scalar_type,
                            const torch::lazy::BackendDevice* device);

xla::PrimitiveType GetShapeDimensionType(
    const torch::lazy::BackendDevice* device);

// The following functions are copied from aten/src/ATen/ExpandUtils.h just to
// replace the expand with expand_copy.
// TODO(alanwaketan): Fix the upstream.
inline std::tuple<c10::MaybeOwned<at::Tensor>, c10::MaybeOwned<at::Tensor>,
                  c10::MaybeOwned<at::Tensor>>
xla_expand_outplace(const at::Tensor& to_expand1, const at::Tensor& to_expand2,
                    const at::Tensor& to_expand3, const char* api_name) {
  at::check_defined({to_expand1, to_expand2, to_expand3}, api_name);
  if (to_expand1.sizes().equals(to_expand2.sizes()) &&
      to_expand1.sizes().equals(to_expand3.sizes())) {
    return std::make_tuple(c10::MaybeOwned<at::Tensor>::borrowed(to_expand1),
                           c10::MaybeOwned<at::Tensor>::borrowed(to_expand2),
                           c10::MaybeOwned<at::Tensor>::borrowed(to_expand3));
  }

  auto expanded_size12 =
      at::infer_size_dimvector(to_expand1.sizes(), to_expand2.sizes());
  auto expanded_size =
      at::infer_size_dimvector(expanded_size12, to_expand3.sizes());
  return std::make_tuple(c10::MaybeOwned<at::Tensor>::owned(
                             at::expand_copy(to_expand1, expanded_size)),
                         c10::MaybeOwned<at::Tensor>::owned(
                             at::expand_copy(to_expand2, expanded_size)),
                         c10::MaybeOwned<at::Tensor>::owned(
                             at::expand_copy(to_expand3, expanded_size)));
}

inline bool tensor_has_dym_dim(at::Tensor t) {
  c10::SymIntArrayRef sym_sizes = t.sym_sizes();
  return !c10::asIntArrayRefSlowOpt(sym_sizes).has_value();
}

inline bool tensors_has_dym_dim(at::TensorList tensors) {
  for (const at::Tensor& t : tensors) {
    if (tensor_has_dym_dim(t)) {
      return true;
    }
  }
  return false;
}

inline std::vector<at::Tensor> xla_expand_outplace_symint_helper(
    at::TensorList to_expand) {
  // expands a list of Tensors; ignores undefined (null) tensors
  bool first = true;
  at::SymDimVector sym_sizes;
  for (const auto i : c10::irange(to_expand.size())) {
    if (!to_expand[i].defined()) {
      continue;
    } else if (first) {
      sym_sizes = to_expand[i].sym_sizes();
      first = false;
    } else {
      sym_sizes =
          at::infer_size_symdimvector(sym_sizes, to_expand[i].sym_sizes());
    }
  }

  std::vector<at::Tensor> result(to_expand.size());
  for (const auto i : c10::irange(to_expand.size())) {
    if (!to_expand[i].defined()) {
      continue;
    } else if (to_expand[i].sym_sizes().equals(sym_sizes)) {
      result[i] = to_expand[i];
    } else {
      result[i] = at::expand_copy_symint(to_expand[i], sym_sizes);
    }
  }
  return result;
}

inline std::vector<at::Tensor> xla_expand_outplace_helper(
    at::TensorList to_expand) {
  bool first = true;
  at::DimVector sizes;
  for (const auto i : c10::irange(to_expand.size())) {
    if (!to_expand[i].defined()) {
      continue;
    } else if (first) {
      sizes = to_expand[i].sizes();
      first = false;
    } else {
      sizes = at::infer_size_dimvector(sizes, to_expand[i].sizes());
    }
  }

  std::vector<at::Tensor> result(to_expand.size());
  for (const auto i : c10::irange(to_expand.size())) {
    if (!to_expand[i].defined()) {
      continue;
    } else if (to_expand[i].sizes().equals(sizes)) {
      result[i] = to_expand[i];
    } else {
      result[i] = at::expand_copy(to_expand[i], sizes);
    }
  }
  return result;
}

inline std::vector<at::Tensor> xla_expand_outplace(at::TensorList to_expand) {
  // expands a list of Tensors; ignores undefined (null) tensors
  if (tensors_has_dym_dim(to_expand)) {
    return xla_expand_outplace_symint_helper(to_expand);
  } else {
    return xla_expand_outplace_helper(to_expand);
  }
}

std::shared_ptr<runtime::ComputationClient::Data> get_data_handle(
    const at::Tensor& input);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_TENSOR_UTIL_H_
