#include "torch_xla/csrc/ops/max_in_dim.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, xla::int64 dim, bool keepdim) {
  auto lower_for_shape_fn =
      [&](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    xla::XlaOp values = BuildMaxInDim(operands[0], dim, keepdim);
    xla::XlaOp indices = BuildArgMax(operands[0], dim, keepdim);
    return xla::Tuple(values.builder(), {values, indices});
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

MaxInDim::MaxInDim(const Value& input, xla::int64 dim, bool keepdim)
    : Node(
          ir::OpKind(at::aten::max), {input},
          [&]() { return NodeOutputShape(input, dim, keepdim); },
          /*num_outputs=*/2, xla::util::MHash(dim, keepdim)),
      dim_(dim),
      keepdim_(keepdim) {}

NodePtr MaxInDim::Clone(OpList operands) const {
  return MakeNode<MaxInDim>(operands.at(0), dim_, keepdim_);
}

XlaOpVector MaxInDim::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp values = BuildMaxInDim(input, dim_, keepdim_);
  xla::XlaOp indices = BuildArgMax(input, dim_, keepdim_);
  return ReturnOps({values, indices}, loctx);
}

std::string MaxInDim::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_ << ", keepdim=" << keepdim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
