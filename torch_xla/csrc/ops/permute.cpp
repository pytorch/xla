#include "torch_xla/csrc/ops/permute.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, absl::Span<const int64_t> dims) {
  auto lower_for_shape_fn =
      [dims](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1);
    return xla::Transpose(operands[0], dims);
  };
  return InferOutputShape({input.xla_shape()}, lower_for_shape_fn);
}

}  // namespace

Permute::Permute(const Value& input, std::vector<int64_t> dims)
    : Node(torch::lazy::OpKind(at::aten::permute), {input},
           [&]() { return NodeOutputShape(input, dims); },
           /*num_outputs=*/1, torch::lazy::MHash(dims)),
      dims_(std::move(dims)) {}

torch::lazy::NodePtr Permute::Clone(OpList operands) const {
  return ir::MakeNode<Permute>(operands.at(0), dims_);
}

XlaOpVector Permute::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = xla::Transpose(input, dims_);
  return ReturnOp(output, loctx);
}

std::string Permute::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dims=(" << absl::StrJoin(dims_, ", ") << ")";
  return ss.str();
}

xla::Shape Permute::MakePermuteShape(const xla::Shape& source_shape,
                                     absl::Span<const int64_t> permutation) {
  return XlaHelpers::GetDynamicReshape(
      source_shape,
      XlaHelpers::Permute(permutation, source_shape.dimensions()));
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
