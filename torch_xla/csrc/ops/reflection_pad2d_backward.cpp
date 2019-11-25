#include "torch_xla/csrc/ops/reflection_pad2d_backward.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(
    const Value& grad_output, const Value& input,
    tensorflow::gtl::ArraySlice<const xla::int64> padding) {
  auto lower_for_shape_fn =
      [&](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    return BuildReflectionPad2dBackward(operands[0], operands[1], padding);
  };
  return InferOutputShape({grad_output.shape(), input.shape()},
                          lower_for_shape_fn);
}

}  // namespace

ReflectionPad2dBackward::ReflectionPad2dBackward(
    const Value& grad_output, const Value& input,
    std::vector<xla::int64> padding)
    : Node(OpKind(at::aten::reflection_pad2d_backward), {grad_output, input},
           [&]() { return NodeOutputShape(grad_output, input, padding); },
           /*num_outputs=*/1, xla::util::MHash(padding)),
      padding_(std::move(padding)) {}

NodePtr ReflectionPad2dBackward::Clone(OpList operands) const {
  return MakeNode<ReflectionPad2dBackward>(operands.at(0), operands.at(1),
                                           padding_);
}

XlaOpVector ReflectionPad2dBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp output =
      BuildReflectionPad2dBackward(grad_output, input, padding_);
  return ReturnOp(output, loctx);
}

std::string ReflectionPad2dBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", padding=(" << absl::StrJoin(padding_, ", ")
     << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
