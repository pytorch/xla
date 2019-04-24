#include "torch_xla/csrc/ops/arg_min.h"

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaOp LowerArgMin(const xla::XlaOp& input, xla::int64 dim, bool keepdim) {
  xla::Shape shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp operand = input;
  if (dim < 0) {
    dim = 0;
    operand = xla::Reshape(operand, {xla::ShapeUtil::ElementsIn(shape)});
    shape = XlaHelpers::ShapeOfXlaOp(operand);
  }
  xla::XlaOp result = xla::ArgMinTwoPass(operand, xla::PrimitiveType::S64, dim);
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
      -> xla::XlaOp { return LowerArgMin(operands[0], dim, keepdim); };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

ArgMin::ArgMin(const Value& input, xla::int64 dim, bool keepdim)
    : Node(
          ir::OpKind(at::aten::argmin), {input},
          [&]() { return NodeOutputShape(input, dim, keepdim); },
          /*num_outputs=*/1, xla::util::MHash(dim, keepdim)),
      dim_(dim),
      keepdim_(keepdim) {}

XlaOpVector ArgMin::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(LowerArgMin(input, dim_, keepdim_), loctx);
}

std::string ArgMin::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_ << ", keepdim=" << keepdim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
