#include "torch_xla/csrc/ops/adaptive_avg_pool2d.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/pooling.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(
    const Value& input,
    tensorflow::gtl::ArraySlice<const xla::int64> output_size) {
  auto lower_for_shape_fn =
      [output_size](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1);
    return BuildAdaptiveAvgPool2d(operands[0], output_size);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

AdaptiveAvgPool2d::AdaptiveAvgPool2d(const Value& input,
                                     std::vector<xla::int64> output_size)
    : Node(ir::OpKind(at::aten::adaptive_avg_pool2d), {input},
           NodeOutputShape(input, output_size),
           /*num_outputs=*/1, xla::util::MHash(output_size)),
      output_size_(std::move(output_size)) {}

XlaOpVector AdaptiveAvgPool2d::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildAdaptiveAvgPool2d(input, output_size_);
  return ReturnOp(output, loctx);
}

std::string AdaptiveAvgPool2d::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", output_size=["
     << absl::StrJoin(output_size_, ", ") << "]";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
