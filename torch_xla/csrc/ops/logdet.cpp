#include "torch_xla/csrc/ops/logdet.h"

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/logdet.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaOp LowerLogDet(const xla::XlaOp& input) {
  xla::XlaOp logdet_result = xla::LogDet(input);
  return logdet_result;
}

xla::Shape NodeOutputShape(const Value& input) {
  const xla::Shape& input_shape = input.shape();
  XLA_CHECK_GE(input_shape.rank(), 2) << input_shape;
  // The input tensor is ...,N,N
  xla::Shape logdet_shape(input_shape);
  logdet_shape.DeleteDimension(input_shape.rank() - 1);
  logdet_shape.DeleteDimension(input_shape.rank() - 2);
  return logdet_shape;
}

}  // namespace

LogDet::LogDet(const Value& input)
    : Node(ir::OpKind(at::aten::logdet), {input},
           [&]() { return NodeOutputShape(input); }) {}

NodePtr LogDet::Clone(OpList operands) const {
  return MakeNode<LogDet>(operands.at(0));
}

XlaOpVector LogDet::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOps(LowerLogDet(input), loctx);
}

std::string LogDet::ToString() const {
  std::stringstream ss;
  ss << Node::ToString();
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
