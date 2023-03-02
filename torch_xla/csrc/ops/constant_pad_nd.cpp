#include "torch_xla/csrc/ops/constant_pad_nd.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "third_party/xla_client/debug_macros.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/scalar.h"

namespace torch_xla {
namespace {

xla::XlaOp LowerPad(xla::XlaOp input, const at::Scalar& value,
                    absl::Span<const int64_t> pad) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  return xla::Pad(input,
                  XlaHelpers::ScalarValue(value, input_shape.element_type(),
                                          input.builder()),
                  XlaHelpers::MakeXlaPaddingConfigFromNdPadding(pad));
}

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           const at::Scalar& value,
                           absl::Span<const int64_t> pad) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return LowerPad(operands[0], value, pad);
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

}  // namespace

ConstantPadNd::ConstantPadNd(const torch::lazy::Value& input,
                             std::vector<int64_t> pad, const at::Scalar& value)
    : XlaNode(
          torch::lazy::OpKind(at::aten::constant_pad_nd), {input},
          [&]() { return NodeOutputShape(input, value, pad); },
          /*num_outputs=*/1, torch::lazy::MHash(pad, ScalarHash(value))),
      pad_(std::move(pad)),
      value_(value) {}

torch::lazy::NodePtr ConstantPadNd::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<ConstantPadNd>(operands.at(0), pad_, value_);
}

XlaOpVector ConstantPadNd::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = LowerPad(input, value_, pad_);
  return ReturnOp(output, loctx);
}

std::string ConstantPadNd::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", pad=(" << absl::StrJoin(pad_, ", ") << ")"
     << ", value=" << value_;
  return ss.str();
}

}  // namespace torch_xla
