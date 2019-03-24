#include "torch_xla/csrc/ops/tensor_data.h"

#include <sstream>

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

template <typename T>
void PrintValues(std::ostream& stream, const at::Tensor& tensor) {
  at::Tensor flat_tensor = tensor.contiguous();
  const T* data = flat_tensor.data<T>();
  size_t num_elements = flat_tensor.numel();
  stream << "[";
  for (size_t i = 0; i < num_elements; ++i) {
    if (i > 0) {
      stream << ", ";
    }
    stream << data[i];
  }
  stream << "]";
}

void PrintTensorValues(std::ostream& stream, const at::Tensor& tensor) {
  switch (tensor.type().scalarType()) {
    case at::ScalarType::Double:
      return PrintValues<double>(stream, tensor);
    case at::ScalarType::Float:
      return PrintValues<float>(stream, tensor);
    case at::ScalarType::Byte:
      return PrintValues<uint8_t>(stream, tensor);
    case at::ScalarType::Char:
      return PrintValues<int8_t>(stream, tensor);
    case at::ScalarType::Short:
      return PrintValues<int16_t>(stream, tensor);
    case at::ScalarType::Int:
      return PrintValues<int32_t>(stream, tensor);
    case at::ScalarType::Long:
      return PrintValues<int64_t>(stream, tensor);
    default:
      XLA_ERROR() << "Type not supported: " << tensor.type().scalarType();
  }
}

}  // namespace

TensorData::TensorData(at::Tensor tensor, Device device)
    : Node(xla_tensor_data, CreateComputationShapeFromTensor(tensor, &device),
           GetTensorHashSeed(tensor)),
      tensor_(std::move(tensor)),
      device_(std::move(device)) {}

std::string TensorData::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", device=" << device_;
  if (ShouldMakeConstant(tensor_)) {
    ss << ", values=";
    PrintTensorValues(ss, tensor_);
  }
  return ss.str();
}

XlaOpVector TensorData::Lower(LoweringContext* loctx) const {
  if (ShouldMakeConstant(tensor_)) {
    xla::Literal value = GetTensorLiteral(tensor_, &shape(), &device_);
    return ReturnOp(xla::ConstantLiteral(loctx->builder(), value), loctx);
  }
  xla::ComputationClient::DataPtr data = TensorToXlaData(tensor_, device_);
  return ReturnOp(loctx->GetParameter(data), loctx);
}

bool TensorData::ShouldMakeConstant(const at::Tensor& tensor) {
  // Anything which is not a scalar will not be a constant and will be device
  // data.
  return tensor.numel() == 1;
}

size_t TensorData::GetTensorHashSeed(const at::Tensor& tensor) {
  // If we generate device data, the constant seed is OK (as the shape of it
  // will get into the hash mix in the IrNode constructor).
  // But if this is a constant, we need to hash the content of the tensor.
  size_t seed = 0x34acb4b;
  if (ShouldMakeConstant(tensor)) {
    at::Tensor flat_tensor = tensor.contiguous();
    const c10::TensorImpl* impl = flat_tensor.unsafeGetTensorImpl();
    seed = xla::util::DataHash(impl->data(), impl->numel() * impl->itemsize());
  }
  return seed;
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
