#include "torch_xla/csrc/ops/diagonal.h"

#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/matrix.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, xla::int64 offset,
                           xla::int64 dim1, xla::int64 dim2) {
  auto lower_for_shape_fn =
      [offset, dim1,
       dim2](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1);
    return BuildDiagonal(operands[0], offset, dim1, dim2);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Diagonal::Diagonal(const Value& input, xla::int64 offset, xla::int64 dim1,
                   xla::int64 dim2)
    : Node(
          ir::OpKind(at::aten::diagonal), {input},
          [&]() { return NodeOutputShape(input, offset, dim1, dim2); },
          /*num_outputs=*/1, xla::util::MHash(offset, dim1, dim2)),
      offset_(offset),
      dim1_(dim1),
      dim2_(dim2) {}

NodePtr Diagonal::Clone(OpList operands) const {
  return MakeNode<Diagonal>(operands.at(0), offset_, dim1_, dim2_);
}

XlaOpVector Diagonal::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildDiagonal(input, offset_, dim1_, dim2_);
  return ReturnOp(output, loctx);
}

std::string Diagonal::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", offset=" << offset_ << ", dim1=" << dim1_
     << ", dim2=" << dim2_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
