#include "torch_xla/csrc/ops/unique2.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input) {
  xla::Shape input_shape = GetXlaShape(input);
  int64_t num_elements = xla::ShapeUtil::ElementsIn(input_shape);
  xla::PrimitiveType indices_type = GetShapeDimensionType(/*device=*/nullptr);
  xla::Shape unique_elements_shape =
      xla::ShapeUtil::MakeShape(input_shape.element_type(), {num_elements});
  xla::Shape inverse_indices_shape =
      xla::ShapeUtil::MakeShape(indices_type, input_shape.dimensions());
  xla::Shape counts_shape =
      xla::ShapeUtil::MakeShape(indices_type, {num_elements});
  unique_elements_shape.set_dynamic_dimension(0, true);
  counts_shape.set_dynamic_dimension(0, true);
  return xla::ShapeUtil::MakeTupleShape(
      {unique_elements_shape, inverse_indices_shape, counts_shape});
}

}  // namespace

Unique2::Unique2(const torch::lazy::Value& input)
    : XlaNode(torch::lazy::OpKind(at::aten::_unique2), {input},
              [&]() { return NodeOutputShape(input); },
              /*num_outputs=*/3) {}

torch::lazy::NodePtr Unique2::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Unique2>(operands.at(0));
}

XlaOpVector Unique2::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOps(BuildUnique2(input), loctx);
}

}  // namespace torch_xla
