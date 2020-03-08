#include "torch_xla/csrc/ops/constant_pad_nd.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/scalar.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaOp LowerPad(xla::XlaOp input, const at::Scalar& value,
                    absl::Span<const xla::int64> pad) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  return xla::Pad(input,
                  XlaHelpers::ScalarValue(value, input_shape.element_type(),
                                          input.builder()),
                  XlaHelpers::MakeXlaPaddingConfigFromNdPadding(pad));
}

xla::Shape NodeOutputShape(const Value& input, const at::Scalar& value,
                           absl::Span<const xla::int64> pad) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return LowerPad(operands[0], value, pad);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

ConstantPadNd::ConstantPadNd(const Value& input, std::vector<xla::int64> pad,
                             at::Scalar value)
    : Node(ir::OpKind(at::aten::constant_pad_nd), OpList{input},
           [&]() { return NodeOutputShape(input, value, pad); },
           /*num_outputs=*/1, xla::util::MHash(pad, ScalarHash(value))),
      pad_(std::move(pad)),
      value_(value) {}

NodePtr ConstantPadNd::Clone(OpList operands) const {
  return MakeNode<ConstantPadNd>(operands.at(0), pad_, value_);
}

XlaOpVector ConstantPadNd::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = LowerPad(input, value_, pad_);
  return ReturnOp(output, loctx);
}

std::string ConstantPadNd::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", pad=(" << absl::StrJoin(pad_, ", ") << ")"
     << ", value=" << value_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
