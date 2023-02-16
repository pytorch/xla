#include "torch_xla/csrc/ops/resize.h"

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           absl::Span<const int64_t> size) {
  return xla::ShapeUtil::MakeShape(GetXlaShape(input).element_type(), size);
}

}  // namespace

Resize::Resize(const torch::lazy::Value& input, std::vector<int64_t> size)
    : XlaNode(torch::lazy::OpKind(at::aten::resize), {input},
              [&]() { return NodeOutputShape(input, size); },
              /*num_outputs=*/1, torch::lazy::MHash(size)),
      size_(std::move(size)) {}

torch::lazy::NodePtr Resize::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Resize>(operands.at(0), size_);
}

XlaOpVector Resize::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildResize(input, size_);
  return ReturnOp(output, loctx);
}

std::string Resize::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", size=(" << absl::StrJoin(size_, ", ") << ")";
  return ss.str();
}

}  // namespace torch_xla
