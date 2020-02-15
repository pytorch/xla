#include "torch_xla/csrc/ops/masked_select.h"

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input) {
  const xla::Shape& input_shape = input.shape();
  xla::int64 input_elements = xla::ShapeUtil::ElementsIn(input_shape);
  xla::PrimitiveType size_type = GetShapeDimensionType(/*device=*/nullptr);
  xla::Shape result_shape =
      xla::ShapeUtil::MakeShape(input_shape.element_type(), {input_elements});
  result_shape.set_dynamic_dimension(0, true);
  return xla::ShapeUtil::MakeTupleShape(
      {result_shape, xla::ShapeUtil::MakeShape(size_type, {})});
}

}  // namespace

MaskedSelect::MaskedSelect(const Value& input, const Value& mask)
    : Node(ir::OpKind(at::aten::masked_select), {input, mask},
           NodeOutputShape(input),
           /*num_outputs=*/2) {}

NodePtr MaskedSelect::Clone(OpList operands) const {
  return MakeNode<MaskedSelect>(operands.at(0), operands.at(1));
}

XlaOpVector MaskedSelect::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp mask = loctx->GetOutputOp(operand(1));
  return ReturnOps(BuildMaskedSelect(input, mask), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
