#include "torch_xla/csrc/ops/index_select.h"

#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, const Value& index,
                           xla::int64 dim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::Shape input_shape;
    xla::XlaOp output =
        xla::TorchIndexSelect(XlaHelpers::MakeArray(operands[0], &input_shape),
                              XlaHelpers::MakeArray(operands[1]), dim);
    return XlaHelpers::MaybeReshapeToScalar(output, input_shape);
  };
  return InferOutputShape({input.shape(), index.shape()}, lower_for_shape_fn);
}

}  // namespace

IndexSelect::IndexSelect(const Value& input, xla::int64 dim, const Value& index)
    : Node(ir::OpKind(at::aten::index_select), {input, index},
           [&]() { return NodeOutputShape(input, index, dim); },
           /*num_outputs=*/1, xla::util::MHash(dim)),
      dim_(dim) {}

NodePtr IndexSelect::Clone(OpList operands) const {
  return MakeNode<IndexSelect>(operands.at(0), dim_, operands.at(1));
}

XlaOpVector IndexSelect::Lower(LoweringContext* loctx) const {
  xla::Shape input_shape;
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp index = loctx->GetOutputOp(operand(1));
  xla::XlaOp output =
      xla::TorchIndexSelect(XlaHelpers::MakeArray(input, &input_shape),
                            XlaHelpers::MakeArray(index), dim_);
  return ReturnOp(XlaHelpers::MaybeReshapeToScalar(output, input_shape), loctx);
}

std::string IndexSelect::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
