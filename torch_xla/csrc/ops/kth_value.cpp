#include "torch_xla/csrc/ops/kth_value.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, xla::int64 k, xla::int64 dim,
                           bool keepdim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::Shape input_shape;
    std::vector<xla::XlaOp> outputs = CreateKthValue(
        XlaHelpers::MakeArray(operands[0], &input_shape), k, dim, keepdim);
    return xla::Tuple(operands[0].builder(),
                      XlaHelpers::MaybeReshapeToScalar(outputs, input_shape));
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

KthValue::KthValue(const Value& input, xla::int64 k, xla::int64 dim,
                   bool keepdim)
    : Node(ir::OpKind(at::aten::kthvalue), {input},
           [&]() { return NodeOutputShape(input, k, dim, keepdim); },
           /*num_outputs=*/2, xla::util::MHash(k, dim, keepdim)),
      k_(k),
      dim_(dim),
      keepdim_(keepdim) {}

NodePtr KthValue::Clone(OpList operands) const {
  return MakeNode<KthValue>(operands.at(0), k_, dim_, keepdim_);
}

XlaOpVector KthValue::Lower(LoweringContext* loctx) const {
  xla::Shape input_shape;
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  std::vector<xla::XlaOp> outputs = CreateKthValue(
      XlaHelpers::MakeArray(input, &input_shape), k_, dim_, keepdim_);
  return ReturnOps(XlaHelpers::MaybeReshapeToScalar(outputs, input_shape),
                   loctx);
}

std::string KthValue::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", k=" << k_ << ", dim=" << dim_
     << ", keepdim=" << keepdim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
