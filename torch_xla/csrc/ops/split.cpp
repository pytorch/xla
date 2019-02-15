#include "torch_xla/csrc/ops/split.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, xla::int64 split_size,
                           xla::int64 dim) {
  auto lower_for_shape_fn =
      [split_size, dim](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1);
    return xla::Tuple(operands[0].builder(),
                      BuildSplit(operands[0], split_size, dim));
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Split::Split(const Value& input, xla::int64 split_size, xla::int64 dim)
    : Node(ir::OpKind(at::aten::split), {input},
           NodeOutputShape(input, split_size, dim),
           /*num_outputs=*/
           RoundUpDiv(input.shape().dimensions(dim), split_size),
           xla::util::MHash(split_size, dim)),
      split_size_(split_size),
      dim_(dim) {}

XlaOpVector Split::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  const auto outputs = BuildSplit(input, split_size_, dim_);
  return ReturnOps(outputs, loctx);
}

std::string Split::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", split_size=" << split_size_ << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
