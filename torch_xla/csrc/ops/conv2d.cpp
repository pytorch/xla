#include "torch_xla/csrc/ops/conv2d.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/convolution.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

// The bias doesn't matter for shape inference.
xla::Shape NodeOutputShape(
    const Value& input, const Value& weight,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding) {
  auto lower_for_shape_fn =
      [stride, padding](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    XLA_CHECK(operands.size() == 2 || operands.size() == 3)
        << "Unexpected number of operands: " << operands.size();
    return BuildConvolution(operands[0], operands[1], stride, padding);
  };
  return InferOutputShape({input.shape(), weight.shape()}, lower_for_shape_fn);
}

}  // namespace

Conv2d::Conv2d(const Value& input, const Value& weight, const Value& bias,
               std::vector<xla::int64> stride, std::vector<xla::int64> padding)
    : Node(ir::OpKind(at::aten::convolution), {input, weight, bias},
           NodeOutputShape(input, weight, stride, padding),
           /*num_outputs=*/1, xla::util::MHash(stride, padding)),
      stride_(std::move(stride)),
      padding_(std::move(padding)) {}

Conv2d::Conv2d(const Value& input, const Value& weight,
               std::vector<xla::int64> stride, std::vector<xla::int64> padding)
    : Node(ir::OpKind(at::aten::convolution), {input, weight},
           NodeOutputShape(input, weight, stride, padding)),
      stride_(std::move(stride)),
      padding_(std::move(padding)) {}

XlaOpVector Conv2d::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp kernel = loctx->GetOutputOp(operand(1));
  xla::XlaOp output;
  if (operands().size() == 3) {
    xla::XlaOp bias = loctx->GetOutputOp(operand(2));
    output = BuildConvolutionBias(input, kernel, bias, stride_, padding_);
  } else {
    XLA_CHECK_EQ(operands().size(), 2);
    output = BuildConvolution(input, kernel, stride_, padding_);
  }
  return ReturnOp(output, loctx);
}

std::string Conv2d::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", stride=[" << absl::StrJoin(stride_, ", ")
     << "], padding=[" << absl::StrJoin(padding_, ", ") << "]";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
