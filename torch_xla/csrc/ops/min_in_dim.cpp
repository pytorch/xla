#include "torch_xla/csrc/ops/min_in_dim.h"

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
    xla::XlaOp values = BuildMinInDim(operands[0], dim, keepdim);
    xla::XlaOp indices = BuildArgMin(operands[0], dim, keepdim);
    return xla::Tuple(values.builder(), {values, indices});
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

MinInDim::MinInDim(const Value& input, xla::int64 dim, bool keepdim)
    : Node(
          ir::OpKind(at::aten::min), {input},
          [&]() { return NodeOutputShape(input, dim, keepdim); },
          /*num_outputs=*/2, xla::util::MHash(dim, keepdim)),
      dim_(dim),
      keepdim_(keepdim) {}

XlaOpVector MinInDim::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp values = BuildMinInDim(input, dim_, keepdim_);
  xla::XlaOp indices = BuildArgMin(input, dim_, keepdim_);
  return ReturnOps({values, indices}, loctx);
}

std::string MinInDim::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_ << ", keepdim=" << keepdim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
