#include "torch_xla/csrc/torch_util.h"

#include "third_party/xla_client/debug_macros.h"
#include "third_party/xla_client/xla_util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ops/constant.h"
#include "torch_xla/csrc/tensor.h"

namespace torch_xla {

void SymIntElements::AddSymIntNodeElements(c10::SymInt& size) {
  if (size.is_symbolic()) {
    // c10::SymInt --(convert)--> c10::SymIntNode --(cast)-->
    // lazy::SymIntNodeImpl
    // --(get)--> lazy::NodePtr --(cast)--> lazy::DimensionNode
    c10::SymNode symbolicIntNode = size.toSymNodeImpl();
    auto* lazySymNode = dynamic_cast<XLASymNodeImpl*>(symbolicIntNode.get());
    torch::lazy::NodePtr size_node = lazySymNode->node();
    std::shared_ptr<torch::lazy::DimensionNode> dimension_node =
        std::dynamic_pointer_cast<torch::lazy::DimensionNode>(size_node);
    size_nodes_.push_back(size_node);
    upper_bounds_.push_back(dimension_node->getStaticValue());
    dynamic_dims_.push_back(dimension_node->isSymbolic());
  } else {
    size_nodes_.push_back(nullptr);
    upper_bounds_.push_back(size.expect_int());
    dynamic_dims_.push_back(size.is_symbolic());
  }
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

at::Tensor MaybeWrapTensorToFunctional(const at::Tensor& tensor) {
  bool disable_functionalization =
      xla::sys_util::GetEnvBool("DISABLE_FUNCTIONALIZATION", false);
  if (disable_functionalization) {
    return tensor;
  }
  return at::functionalization::impl::to_functional_tensor(tensor);
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
