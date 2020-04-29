#include "torch_xla/csrc/ops/gather.h"

#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/data_ops.h"
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
    xla::XlaOp array_input = XlaHelpers::MakeArray(operands[0], &input_shape);
    xla::XlaOp array_index = XlaHelpers::MakeArray(operands[1]);
    xla::XlaOp output =
        xla::TorchGather(array_input, array_index, dim,
                         IsSparseGather(array_input, array_index, dim));
    return XlaHelpers::MaybeReshapeToScalar(output, input_shape);
  };
  return InferOutputShape({input.shape(), index.shape()}, lower_for_shape_fn);
}

}  // namespace

Gather::Gather(const Value& input, xla::int64 dim, const Value& index)
    : Node(ir::OpKind(at::aten::gather), {input, index},
           [&]() { return NodeOutputShape(input, index, dim); },
           /*num_outputs=*/1, xla::util::MHash(dim)),
      dim_(dim) {}

NodePtr Gather::Clone(OpList operands) const {
  return MakeNode<Gather>(operands.at(0), dim_, operands.at(1));
}

XlaOpVector Gather::Lower(LoweringContext* loctx) const {
  xla::Shape input_shape;
  xla::XlaOp array_input =
      XlaHelpers::MakeArray(loctx->GetOutputOp(operand(0)), &input_shape);
  xla::XlaOp array_index =
      XlaHelpers::MakeArray(loctx->GetOutputOp(operand(1)));
  xla::XlaOp output =
      xla::TorchGather(array_input, array_index, dim_,
                       IsSparseGather(array_input, array_index, dim_));
  return ReturnOp(XlaHelpers::MaybeReshapeToScalar(output, input_shape), loctx);
}

std::string Gather::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
