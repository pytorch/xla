#include "torch_xla/csrc/torch_util.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ops/constant.h"

namespace torch_xla {

void SymIntElements::SetSymIntNodeElements(c10::SymInt& size) {
  if (size.is_symbolic()) {
    std::shared_ptr<c10::SymIntNodeImpl> symbolicIntNode =
        size.toSymIntNodeImpl();
    auto lazySymIntNode =
        std::dynamic_pointer_cast<torch::lazy::SymIntNodeImpl>(
            symbolicIntNode);
    auto size_node = lazySymIntNode->node_;
    size_nodes_.push_back(size_node);
    upper_bounds_.push_back(
        std::dynamic_pointer_cast<torch::lazy::DimensionNode>(size_node)
            ->getStaticValue());
    dynamic_dims_.push_back(
        std::dynamic_pointer_cast<torch::lazy::DimensionNode>(size_node)
            ->isDynamic());
  } else {
    auto size_node = torch::lazy::MakeNode<Constant>(std::move(
        XlaHelpers::ScalarLiteral(size.expect_int(), xla::PrimitiveType::S32)));
    size_nodes_.push_back(size_node);
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
