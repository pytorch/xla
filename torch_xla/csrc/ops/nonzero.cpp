#include "torch_xla/csrc/ops/nonzero.h"

#include "tensorflow/compiler/xla/shape_util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input) {
  const xla::Shape& input_shape = GetXlaShape(input);
  int64_t index_elements = xla::ShapeUtil::ElementsIn(input_shape);
  xla::PrimitiveType size_type = GetShapeDimensionType(/*device=*/nullptr);
  xla::Shape result_shape = xla::ShapeUtil::MakeShape(
      size_type, {index_elements, input_shape.rank()});
  result_shape.set_dynamic_dimension(0, true);
  return xla::ShapeUtil::MakeTupleShape(
      {result_shape, xla::ShapeUtil::MakeShape(size_type, {})});
}

}  // namespace

NonZero::NonZero(const torch::lazy::Value& input,
                 const torch::lazy::Shape& dynamic_shape)
    : XlaNode(torch::lazy::OpKind(at::aten::nonzero), {input}, dynamic_shape,
              NodeOutputShape(input),
              /*num_outputs=*/2),
      dynamic_shape_(dynamic_shape) {}

torch::lazy::NodePtr NonZero::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<NonZero>(operands.at(0), dynamic_shape_);
}

XlaOpVector NonZero::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOps(BuildNonZero(input), loctx);
}

}  // namespace torch_xla
