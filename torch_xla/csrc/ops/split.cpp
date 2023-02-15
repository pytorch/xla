#include "torch_xla/csrc/ops/split.h"

#include "absl/strings/str_join.h"
#include "third_party/xla_client/debug_macros.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           const std::vector<int64_t>& split_sizes,
                           int64_t dim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::Tuple(operands[0].builder(),
                      BuildSplit(operands[0], split_sizes, dim));
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

}  // namespace

Split::Split(const torch::lazy::Value& input, std::vector<int64_t> split_sizes,
             int64_t dim)
    : XlaNode(
          torch::lazy::OpKind(at::aten::split), {input},
          [&]() { return NodeOutputShape(input, split_sizes, dim); },
          ComputeSplitCount(GetXlaShape(input).dimensions(dim), split_sizes),
          torch::lazy::MHash(split_sizes, dim)),
      split_sizes_(std::move(split_sizes)),
      dim_(dim) {}

torch::lazy::NodePtr Split::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Split>(operands.at(0), split_sizes_, dim_);
}

XlaOpVector Split::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  const auto outputs = BuildSplit(input, split_sizes_, dim_);
  return ReturnOps(outputs, loctx);
}

std::string Split::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", split_sizes=("
     << absl::StrJoin(split_sizes_, ", ") << "), dim=" << dim_;
  return ss.str();
}

}  // namespace torch_xla
