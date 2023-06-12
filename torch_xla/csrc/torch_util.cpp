#include "torch_xla/csrc/torch_util.h"

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ir_builder.h"
#include "torch_xla/csrc/ops/constant.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/xla_util.h"
#include "torch_xla/csrc/tensor.h"

namespace torch_xla {

SymIntElements::SymIntElements(torch::lazy::Value ir) {
  XLAIrBuilder a = XLAIrBuilder();
  xla::Shape shape = GetXlaShape(ir);
  for (int i = 0; i < shape.dimensions().size(); i++) {
    if (shape.is_dynamic_dimension(i)) {
      torch::lazy::NodePtr size_node = a.MakeSizeNode(ir, i);
      size_nodes_.push_back(size_node);
      upper_bounds_.push_back(shape.dimensions(i));
      dynamic_dims_.push_back(true);
    } else {
      size_nodes_.push_back(nullptr);
      upper_bounds_.push_back(shape.dimensions(i));
      dynamic_dims_.push_back(false);
    }
  }
}

void SymIntElements::AddSymIntNodeElements(c10::SymInt& size) {
  if (auto s = size.maybe_as_int()) {
    size_nodes_.push_back(nullptr);
    upper_bounds_.push_back(*s);
    dynamic_dims_.push_back(false);
  } else {
    // c10::SymInt --(convert)--> c10::SymIntNode --(cast)-->
    // lazy::SymIntNodeImpl
    // --(get)--> lazy::NodePtr --(cast)--> lazy::DimensionNode
    auto* lazySymNode =
        dynamic_cast<XLASymNodeImpl*>(size.toSymNodeImplUnowned());
    torch::lazy::NodePtr size_node = lazySymNode->node();
    std::shared_ptr<torch::lazy::DimensionNode> dimension_node =
        std::dynamic_pointer_cast<torch::lazy::DimensionNode>(size_node);
    size_nodes_.push_back(size_node);
    upper_bounds_.push_back(dimension_node->getStaticValue());
    dynamic_dims_.push_back(dimension_node->isSymbolic());
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
      runtime::sys_util::GetEnvBool("XLA_DISABLE_FUNCTIONALIZATION", false);
  if (disable_functionalization) {
    return tensor;
  }
  return at::functionalization::impl::to_functional_tensor(tensor);
}

}  // namespace torch_xla

namespace torch {
namespace lazy {
torch::lazy::hash_t Hash(const xla::Shape& shape) {
  auto shape_hash = torch_xla::runtime::util::ShapeHash(shape);
  return c10::uint128(absl::Uint128High64(shape_hash),
                      absl::Uint128Low64(shape_hash));
}
}  // namespace lazy
}  // namespace torch
