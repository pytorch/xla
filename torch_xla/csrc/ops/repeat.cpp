#include "torch_xla/csrc/ops/repeat.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(
    const Value& input, tensorflow::gtl::ArraySlice<const xla::int64> repeats) {
  auto lower_for_shape_fn =
      [repeats](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1);
    return BuildRepeat(operands[0], repeats);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Repeat::Repeat(const Value& input, std::vector<xla::int64> repeats)
    : Node(
          ir::OpKind(at::aten::repeat), {input},
          [&]() { return NodeOutputShape(input, repeats); },
          /*num_outputs=*/1, xla::util::MHash(repeats)),
      repeats_(std::move(repeats)) {}

XlaOpVector Repeat::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildRepeat(input, repeats_);
  return ReturnOp(output, loctx);
}

std::string Repeat::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", repeats=[" << absl::StrJoin(repeats_, ", ")
     << "]";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
