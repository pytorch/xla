#include "torch_xla/csrc/ops/topk.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, int64_t k, int64_t dim,
                           bool largest, bool sorted, bool stable) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::Tuple(operands[0].builder(),
                      CreateTopK(operands[0], k, dim, largest, stable));
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

TopK::TopK(const Value& input, int64_t k, int64_t dim, bool largest,
           bool sorted, bool stable)
    : Node(ir::OpKind(at::aten::topk), {input},
           [&]() {
             return NodeOutputShape(input, k, dim, largest, sorted, stable);
           },
           /*num_outputs=*/2,
           torch::lazy::MHash(k, dim, largest, sorted, stable)),
      k_(k),
      dim_(dim),
      largest_(largest),
      sorted_(sorted),
      stable_(stable) {}

NodePtr TopK::Clone(OpList operands) const {
  return MakeNode<TopK>(operands.at(0), k_, dim_, largest_, sorted_, stable_);
}

XlaOpVector TopK::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOps(CreateTopK(input, k_, dim_, largest_, stable_), loctx);
}

std::string TopK::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", k=" << k_ << ", dim=" << dim_
     << ", largest=" << largest_ << ", sorted=" << sorted_
     << ", stable=" << stable_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
