#include "torch_xla/csrc/ops/amin.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input,
                           std::vector<xla::int64>& dimensions, bool keepdim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMinInDims(operands[0], dimensions, keepdim);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

AMin::AMin(const Value& input, std::vector<xla::int64> dimensions, bool keepdim)
    : Node(ir::OpKind(at::aten::argmax), {input},
           [&]() { return NodeOutputShape(input, dimensions, keepdim); },
           /*num_outputs=*/1, xla::util::MHash(dimensions, keepdim)),
      dimensions_(std::move(dimensions)),
      keepdim_(keepdim) {}

NodePtr AMin::Clone(OpList operands) const {
  return MakeNode<AMin>(operands.at(0), dimensions_, keepdim_);
}

XlaOpVector AMin::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildMinInDims(input, dimensions_, keepdim_), loctx);
}

std::string AMin::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dimensions=" << absl::StrJoin(dimensions_, ", ")
     << ", keepdim=" << keepdim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
