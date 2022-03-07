#include "torch_xla/csrc/ops/update_slice.h"

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, const Value& source,
                           absl::Span<const int64_t> base_indices) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildUpdateSlice(operands[0], operands[1], base_indices);
  };
  return InferOutputShape({input.shape(), source.shape()}, lower_for_shape_fn);
}

}  // namespace

UpdateSlice::UpdateSlice(const Value& input, const Value& source,
                         absl::Span<const int64_t> base_indices)
    : Node(xla_update_slice, {input, source},
           [&]() { return NodeOutputShape(input, source, base_indices); },
           /*num_outputs=*/1, torch::lazy::Hash(base_indices)),
      base_indices_(base_indices.begin(), base_indices.end()) {}

NodePtr UpdateSlice::Clone(OpList operands) const {
  return MakeNode<UpdateSlice>(operands.at(0), operands.at(1), base_indices_);
}

XlaOpVector UpdateSlice::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp source = loctx->GetOutputOp(operand(1));
  xla::XlaOp output = BuildUpdateSlice(input, source, base_indices_);
  return ReturnOp(output, loctx);
}

std::string UpdateSlice::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", base_indices=("
     << absl::StrJoin(base_indices_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
