#include "torch_xla/csrc/ops/kth_value.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
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
    return xla::Tuple(operands[0].builder(),
                      CreateKthValue(operands[0], k, dim, keepdim));
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
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOps(CreateKthValue(input, k_, dim_, keepdim_), loctx);
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
