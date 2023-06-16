#include "torch_xla/csrc/ops/masked_select.h"

#include "xla/shape_util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input) {
  const xla::Shape& input_shape = GetXlaShape(input);
  int64_t input_elements = xla::ShapeUtil::ElementsIn(input_shape);
  xla::PrimitiveType size_type = GetShapeDimensionType(/*device=*/nullptr);
  xla::Shape result_shape =
      xla::ShapeUtil::MakeShape(input_shape.element_type(), {input_elements});
  result_shape.set_dynamic_dimension(0, true);
  return xla::ShapeUtil::MakeTupleShape(
      {result_shape, xla::ShapeUtil::MakeShape(size_type, {})});
}

}  // namespace

MaskedSelect::MaskedSelect(const torch::lazy::Value& input,
                           const torch::lazy::Value& mask)
    : XlaNode(torch::lazy::OpKind(at::aten::masked_select), {input, mask},
              NodeOutputShape(input),
              /*num_outputs=*/2) {}

torch::lazy::NodePtr MaskedSelect::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<MaskedSelect>(operands.at(0), operands.at(1));
}

XlaOpVector MaskedSelect::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp mask = loctx->GetOutputOp(operand(1));
  return ReturnOps(BuildMaskedSelect(input, mask), loctx);
}

}  // namespace torch_xla
