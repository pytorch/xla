#include "torch_xla/csrc/ops/update_slice.h"

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           const torch::lazy::Value& source,
                           absl::Span<const int64_t> base_indices) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildUpdateSlice(operands[0], operands[1], base_indices);
  };
  return InferOutputShape({GetXlaShape(input), GetXlaShape(source)},
                          lower_for_shape_fn);
}

}  // namespace

UpdateSlice::UpdateSlice(const torch::lazy::Value& input,
                         const torch::lazy::Value& source,
                         absl::Span<const int64_t> base_indices)
    : XlaNode(xla_update_slice, {input, source},
              [&]() { return NodeOutputShape(input, source, base_indices); },
              /*num_outputs=*/1, torch::lazy::Hash(base_indices)),
      base_indices_(base_indices.begin(), base_indices.end()) {}

torch::lazy::NodePtr UpdateSlice::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<UpdateSlice>(operands.at(0), operands.at(1),
                                            base_indices_);
}

XlaOpVector UpdateSlice::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp source = loctx->GetOutputOp(operand(1));
  xla::XlaOp output = BuildUpdateSlice(input, source, base_indices_);
  return ReturnOp(output, loctx);
}

std::string UpdateSlice::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", base_indices=("
     << absl::StrJoin(base_indices_, ", ") << ")";
  return ss.str();
}

}  // namespace torch_xla
