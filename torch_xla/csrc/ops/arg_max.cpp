#include "torch_xla/csrc/ops/arg_max.h"

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaOp LowerArgMax(const xla::XlaOp& input, xla::int64 dim, bool keepdim) {
  xla::Shape shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp operand = input;
  if (dim < 0) {
    dim = 0;
    operand = xla::Reshape(operand, {xla::ShapeUtil::ElementsIn(shape)});
    shape = XlaHelpers::ShapeOfXlaOp(operand);
  }
  xla::XlaOp result = xla::ArgMax(operand, xla::PrimitiveType::S64, dim);
  if (keepdim) {
    auto dimensions = xla::util::ToVector<xla::int64>(shape.dimensions());
    dimensions[dim] = 1;
    result = xla::Reshape(result, dimensions);
  }
  return result;
}

xla::Shape NodeOutputShape(const Value& input, xla::int64 dim, bool keepdim) {
  auto lower_for_shape_fn =
      [&](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp { return LowerArgMax(operands[0], dim, keepdim); };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

ArgMax::ArgMax(const Value& input, xla::int64 dim, bool keepdim)
    : Node(ir::OpKind(at::aten::argmax), {input},
           NodeOutputShape(input, dim, keepdim),
           /*num_outputs=*/1, xla::util::MHash(dim, keepdim)),
      dim_(dim),
      keepdim_(keepdim) {}

XlaOpVector ArgMax::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(LowerArgMax(input, dim_, keepdim_), loctx);
}

std::string ArgMax::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_ << ", keepdim=" << keepdim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
