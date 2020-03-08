#include "torch_xla/csrc/ops/resize.h"

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input,
                           absl::Span<const xla::int64> size) {
  return xla::ShapeUtil::MakeShape(input.shape().element_type(), size);
}

}  // namespace

Resize::Resize(const Value& input, std::vector<xla::int64> size)
    : Node(ir::OpKind(at::aten::resize), {input},
           [&]() { return NodeOutputShape(input, size); },
           /*num_outputs=*/1, xla::util::MHash(size)),
      size_(std::move(size)) {}

NodePtr Resize::Clone(OpList operands) const {
  return MakeNode<Resize>(operands.at(0), size_);
}

XlaOpVector Resize::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildResize(input, size_);
  return ReturnOp(output, loctx);
}

std::string Resize::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", size=(" << absl::StrJoin(size_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
