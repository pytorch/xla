#include "torch_xla/csrc/ops/constant.h"

#include <algorithm>
#include <sstream>

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/runtime/xla_util.h"
#include "torch_xla/csrc/torch_util.h"
#include "xla/layout_util.h"
#include "xla/shape_util.h"

namespace torch_xla {

Constant::Constant(xla::Literal value)
    : XlaNode(torch::lazy::OpKind(at::prim::Constant), value.shape(),
              /*num_outputs=*/1, LiteralHash(value)),
      value_(std::move(value)) {}

std::string Constant::ToString() const {
  // The Literal to string conversion produces \n separated content, which we do
  // not want. It can also produce giant strings, but that's a different issue.
  std::string value_as_string = value_.ToStringWithoutShape();
  std::replace(value_as_string.begin(), value_as_string.end(), '\n', ';');
  std::stringstream ss;
  ss << XlaNode::ToString() << ", value=" << value_as_string;
  return ss.str();
}

torch::lazy::NodePtr Constant::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Constant>(value_.Clone());
}

XlaOpVector Constant::Lower(LoweringContext* loctx) const {
  return ReturnOp(xla::ConstantLiteral(loctx->builder(), value_), loctx);
}

// Based on the AbslHash implementation in
// https://github.com/openxla/xla/blob/46baaafb19d6819d01b3f91be78b5a1e8cc9e14f/xla/literal.h#L323-L341
// AbslHash randomizes the seed for each process, so the resulting hash is not
// suitable for persistent storage.
torch::lazy::hash_t LiteralHash(const xla::Literal& l) {
  auto hash = torch::lazy::Hash(l.shape());
  xla::ShapeUtil::ForEachSubshape(l.shape(), [&](const xla::Shape& subshape,
                                                 const xla::ShapeIndex& index) {
    if (!subshape.IsArray()) {
      return;
    }
    XLA_CHECK(xla::LayoutUtil::IsDenseArray(subshape));
    auto data = absl::MakeConstSpan(
        static_cast<const char*>(l.untyped_data(index)), l.size_bytes(index));
    hash = torch::lazy::HashCombine(torch::lazy::Hash(data), hash);
  });
  return hash;
}

}  // namespace torch_xla
