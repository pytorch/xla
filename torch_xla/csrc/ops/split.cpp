#include "torch_xla/csrc/ops/split.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input,
                           const std::vector<xla::int64>& split_sizes,
                           xla::int64 dim) {
  auto lower_for_shape_fn =
      [&](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    return xla::Tuple(operands[0].builder(),
                      BuildSplit(operands[0], split_sizes, dim));
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Split::Split(const Value& input, std::vector<xla::int64> split_sizes,
             xla::int64 dim)
    : Node(ir::OpKind(at::aten::split), {input},
           [&]() { return NodeOutputShape(input, split_sizes, dim); },
           ComputeSplitCount(input.shape().dimensions(dim), split_sizes),
           xla::util::MHash(split_sizes, dim)),
      split_sizes_(std::move(split_sizes)),
      dim_(dim) {}

NodePtr Split::Clone(OpList operands) const {
  return MakeNode<Split>(operands.at(0), split_sizes_, dim_);
}

XlaOpVector Split::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  const auto outputs = BuildSplit(input, split_sizes_, dim_);
  return ReturnOps(outputs, loctx);
}

std::string Split::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", split_sizes=["
     << absl::StrJoin(split_sizes_, ", ") << "], dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
