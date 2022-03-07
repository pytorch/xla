#include "torch_xla/csrc/ops/generic_slice.h"

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

xla::Shape NodeOutputShape(const Value& input,
                           absl::Span<const int64_t> base_indices,
                           absl::Span<const int64_t> sizes) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildSlice(operands[0], base_indices, sizes);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

GenericSlice::GenericSlice(const Value& input,
                           absl::Span<const int64_t> base_indices,
                           absl::Span<const int64_t> sizes)
    : Node(xla_generic_slice, {input},
           [&]() { return NodeOutputShape(input, base_indices, sizes); },
           /*num_outputs=*/1, torch::lazy::MHash(base_indices, sizes)),
      base_indices_(base_indices.begin(), base_indices.end()),
      sizes_(sizes.begin(), sizes.end()) {}

NodePtr GenericSlice::Clone(OpList operands) const {
  return MakeNode<GenericSlice>(operands.at(0), base_indices_, sizes_);
}

XlaOpVector GenericSlice::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildSlice(input, base_indices_, sizes_);
  return ReturnOp(output, loctx);
}

std::string GenericSlice::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", base_indices=("
     << absl::StrJoin(base_indices_, ", ") << "), sizes=("
     << absl::StrJoin(sizes_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
