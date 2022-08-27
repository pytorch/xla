#include "torch_xla/csrc/torch_util.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ops/constant.h"

namespace torch_xla {

void SymIntElements::SetSymIntNodeElements(c10::SymInt& size) {
  if (size.is_symbolic()) {
    size_t current_index = upper_bounds_.size();
    // c10::SymInt --(convert)--> c10::SymIntNode --(cast)-->
    // lazy::SymIntNodeImpl
    // --(get)--> lazy::NodePtr --(cast)--> lazy::DimensionNode
    c10::SymIntNode symbolicIntNode = size.toSymIntNodeImpl();
    auto* lazySymIntNode =
        dynamic_cast<torch::lazy::SymIntNodeImpl*>(symbolicIntNode.get());
    torch::lazy::NodePtr size_node = lazySymIntNode->node_;
    std::shared_ptr<torch::lazy::DimensionNode> dimension_node =
        std::dynamic_pointer_cast<torch::lazy::DimensionNode>(size_node);
    size_node_map_[current_index] = size_node;
    upper_bounds_.push_back(dimension_node->getStaticValue());
    dynamic_dims_.push_back(dimension_node->isSymbolic());
  } else {
    upper_bounds_.push_back(size.expect_int());
    dynamic_dims_.push_back(size.is_symbolic());
  }
}

torch::lazy::NodePtr SymIntElements::GetNode(size_t index) {
  if (size_node_map_.find(index) != size_node_map_.end()) {
    return size_node_map_[index];
  }
  return nullptr;
}

at::ScalarType GetScalarType(const at::Scalar& scalar) {
  if (scalar.isFloatingPoint()) {
    return at::kDouble;
  } else if (scalar.isIntegral(/*includeBool=*/false)) {
    return at::kLong;
  } else if (scalar.isBoolean()) {
    return at::kBool;
  } else if (scalar.isComplex()) {
    return at::kComplexDouble;
  }
  XLA_ERROR() << "Unknown type for scalar";
}

at::Tensor UnwrapNumber(const at::Tensor& tensor, at::ScalarType dtype) {
  return tensor.unsafeGetTensorImpl()->is_wrapped_number() ? tensor.to(dtype)
                                                           : tensor;
}

}  // namespace torch_xla

namespace torch {
namespace lazy {
torch::lazy::hash_t Hash(const xla::Shape& shape) {
  auto shape_hash = xla::util::ShapeHash(shape);
  return c10::uint128(absl::Uint128High64(shape_hash),
                      absl::Uint128Low64(shape_hash));
}
}  // namespace lazy
}  // namespace torch
