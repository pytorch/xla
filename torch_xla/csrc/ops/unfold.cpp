#include "torch_xla/csrc/ops/unfold.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {

xla::Shape NodeOutputShape(const Value& input, xla::int64 dimension,
                           xla::int64 size, xla::int64 step) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildUnfold(operands[0], dimension, size, step);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

Unfold::Unfold(const Value& input, xla::int64 dimension, xla::int64 size,
               xla::int64 step)
    : Node(ir::OpKind(at::aten::unfold), {input},
           [&]() { return NodeOutputShape(input, dimension, size, step); },
           /*num_outputs=*/1, xla::util::MHash(dimension, size, step)),
      dimension_(dimension),
      size_(size),
      step_(step) {}

NodePtr Unfold::Clone(OpList operands) const {
  return MakeNode<Unfold>(operands.at(0), dimension_, size_, step_);
}

XlaOpVector Unfold::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildUnfold(input, dimension_, size_, step_);
  return ReturnOp(output, loctx);
}

std::string Unfold::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dimension=" << dimension_ << ", size=" << size_
     << ", step=" << step_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
