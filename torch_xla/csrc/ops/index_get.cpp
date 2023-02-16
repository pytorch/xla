#include "torch_xla/csrc/ops/index_get.h"

#include "third_party/xla_client/debug_macros.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& base,
                           const torch::lazy::Value& indices,
                           int64_t start_dim) {
  auto lower_for_shape_fn =
      [start_dim](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 2);
    return CreateIndex(operands[0], operands[1], start_dim);
  };
  return InferOutputShape({GetXlaShape(base), GetXlaShape(indices)},
                          lower_for_shape_fn);
}

}  // namespace

IndexGet::IndexGet(const torch::lazy::Value& base,
                   const torch::lazy::Value& indices, int64_t start_dim)
    : XlaNode(torch::lazy::OpKind(at::aten::index), {base, indices},
              [&]() { return NodeOutputShape(base, indices, start_dim); },
              /*num_outputs=*/1, torch::lazy::MHash(start_dim)),
      start_dim_(start_dim) {}

std::string IndexGet::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", start_dim=" << start_dim_;
  return ss.str();
}

torch::lazy::NodePtr IndexGet::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<IndexGet>(operands.at(0), operands.at(1),
                                         start_dim_);
}

XlaOpVector IndexGet::Lower(LoweringContext* loctx) const {
  xla::XlaOp base = loctx->GetOutputOp(operand(0));
  xla::XlaOp indices = loctx->GetOutputOp(operand(1));
  xla::XlaOp output = CreateIndex(base, indices, start_dim_);
  return ReturnOp(output, loctx);
}

}  // namespace torch_xla
