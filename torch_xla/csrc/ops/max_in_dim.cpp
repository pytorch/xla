#include "torch_xla/csrc/ops/max_in_dim.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input, int64_t dim,
                           bool keepdim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::XlaOp values = BuildMaxInDim(operands[0], dim, keepdim);
    xla::XlaOp indices = BuildArgMax(operands[0], dim, keepdim);
    return xla::Tuple(values.builder(), {values, indices});
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

}  // namespace

MaxInDim::MaxInDim(const torch::lazy::Value& input, int64_t dim, bool keepdim)
    : XlaNode(
          torch::lazy::OpKind(at::aten::max), {input},
          [&]() { return NodeOutputShape(input, dim, keepdim); },
          /*num_outputs=*/2, torch::lazy::MHash(dim, keepdim)),
      dim_(dim),
      keepdim_(keepdim) {}

torch::lazy::NodePtr MaxInDim::Clone(torch::lazy::OpList operands) const {
  return torch_xla::MakeNode<MaxInDim>(operands.at(0), dim_, keepdim_);
}

XlaOpVector MaxInDim::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp values = BuildMaxInDim(input, dim_, keepdim_);
  xla::XlaOp indices = BuildArgMax(input, dim_, keepdim_);
  return ReturnOps({values, indices}, loctx);
}

std::string MaxInDim::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dim=" << dim_ << ", keepdim=" << keepdim_;
  return ss.str();
}

}  // namespace torch_xla
