#include "torch_xla/csrc/ops/amax.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, std::vector<int64_t>& dimensions,
                           bool keepdim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMaxInDims(operands[0], dimensions, keepdim);
  };
  return InferOutputShape({input.xla_shape()}, lower_for_shape_fn);
}

}  // namespace

Amax::Amax(const Value& input, std::vector<int64_t> dimensions, bool keepdim)
    : XlaNode(torch::lazy::OpKind(at::aten::amax), {input},
           [&]() { return NodeOutputShape(input, dimensions, keepdim); },
           /*num_outputs=*/1, torch::lazy::MHash(dimensions, keepdim)),
      dimensions_(std::move(dimensions)),
      keepdim_(keepdim) {}

torch::lazy::NodePtr Amax::Clone(OpList operands) const {
  return ir::MakeNode<Amax>(operands.at(0), dimensions_, keepdim_);
}

XlaOpVector Amax::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildMaxInDims(input, dimensions_, keepdim_), loctx);
}

std::string Amax::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dimensions=" << absl::StrJoin(dimensions_, ", ")
     << ", keepdim=" << keepdim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
