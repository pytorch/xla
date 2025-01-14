#include "torch_xla/csrc/ops/cummin.h"

#include <torch/csrc/lazy/core/tensor_util.h>

#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/reduction.h"
#include "torch_xla/csrc/shape_helper.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace {

xla::XlaOp LowerCumMin(xla::XlaOp input, int64_t dim) {
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  xla::XlaOp value_init_value = xla::ConstantLiteral(
      input.builder(), xla::LiteralUtil::MaxValue(input_shape.element_type()));
  xla::XlaOp index_init_value = xla::ConstantLiteral(
      input.builder(), xla::LiteralUtil::Zero(xla::PrimitiveType::S32));
  xla::XlaOp iota =
      xla::Iota(input.builder(),
                xla::ShapeUtil::MakeShape(xla::PrimitiveType::S32,
                                          input_shape.dimensions()),
                dim);
  xla::XlaComputation reducer = XlaHelpers::CreateMinAndArgMinComputation(
      input_shape.element_type(), xla::PrimitiveType::S32);
  return BuildCumulativeComputationWithIndices(
      input, iota, dim, reducer, value_init_value, index_init_value);
}

xla::Shape NodeOutputShape(const torch::lazy::Value& input, int64_t dim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::XlaOp values_and_indices = LowerCumMin(operands[0], dim);
    return values_and_indices;
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

}  // namespace

CumMin::CumMin(const torch::lazy::Value& input, int64_t dim)
    : XlaNode(
          torch::lazy::OpKind(at::aten::cummin), {input},
          [&]() { return NodeOutputShape(input, dim); },
          /*num_outputs=*/2, torch::lazy::MHash(dim)),
      dim_(dim) {}

torch::lazy::NodePtr CumMin::Clone(torch::lazy::OpList operands) const {
  return torch_xla::MakeNode<CumMin>(operands.at(0), dim_);
}

XlaOpVector CumMin::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp values_and_indices = LowerCumMin(input, dim_);
  return ReturnOps({xla::GetTupleElement(values_and_indices, 0),
                    xla::GetTupleElement(values_and_indices, 1)},
                   loctx);
}

std::string CumMin::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace torch_xla
