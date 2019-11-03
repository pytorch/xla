#include "torch_xla/csrc/ops/nonzero.h"

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input) {
  const xla::Shape& input_shape = input.shape();
  xla::int64 index_elements = xla::ShapeUtil::ElementsIn(input_shape);
  xla::Shape result_shape = xla::ShapeUtil::MakeShape(
      xla::PrimitiveType::S32, {index_elements, input_shape.rank()});
  result_shape.set_dynamic_dimension(0, true);
  return xla::ShapeUtil::MakeTupleShape(
      {result_shape, xla::ShapeUtil::MakeShape(xla::PrimitiveType::S32, {})});
}

}  // namespace

NonZero::NonZero(const Value& input)
    : Node(ir::OpKind(at::aten::nonzero), {input}, NodeOutputShape(input),
           /*num_outputs=*/2) {}

NodePtr NonZero::Clone(OpList operands) const {
  return MakeNode<NonZero>(operands.at(0));
}

XlaOpVector NonZero::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOps(BuildNonZero(input), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
