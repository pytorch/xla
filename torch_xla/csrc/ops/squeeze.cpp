#include "torch_xla/csrc/ops/squeeze.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaOp LowerSqueeze(const xla::XlaOp& input, int dim) {
  if (dim == -1) {
    return SqueezeAllTrivialDimensions(input);
  }
  XLA_CHECK_GE(dim, 0);
  return SqueezeTrivialDimension(input, dim);
}

xla::Shape NodeOutputShape(const Value& input, int dim) {
  auto lower_for_shape_fn =
      [dim](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1);
    return LowerSqueeze(operands[0], dim);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Squeeze::Squeeze(const Value& input, int dim)
    : Node(ir::OpKind(at::aten::squeeze), {input}, NodeOutputShape(input, dim),
           /*num_outputs=*/1, xla::util::MHash(dim)),
      dim_(dim) {}

XlaOpVector Squeeze::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = LowerSqueeze(input, dim_);
  return ReturnOp(output, loctx);
}

std::string Squeeze::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << " dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
