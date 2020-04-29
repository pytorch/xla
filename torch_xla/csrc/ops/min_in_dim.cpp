#include "torch_xla/csrc/ops/min_in_dim.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, xla::int64 dim, bool keepdim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::Shape input_shape;
    xla::XlaOp values = BuildMinInDim(
        XlaHelpers::MakeArray(operands[0], &input_shape), dim, keepdim);
    xla::XlaOp indices =
        BuildArgMin(XlaHelpers::MakeArray(operands[0]), dim, keepdim);
    return xla::Tuple(values.builder(),
                      {XlaHelpers::MaybeReshapeToScalar(values, input_shape),
                       XlaHelpers::MaybeReshapeToScalar(indices, input_shape)});
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

MinInDim::MinInDim(const Value& input, xla::int64 dim, bool keepdim)
    : Node(ir::OpKind(at::aten::min), {input},
           [&]() { return NodeOutputShape(input, dim, keepdim); },
           /*num_outputs=*/2, xla::util::MHash(dim, keepdim)),
      dim_(dim),
      keepdim_(keepdim) {}

NodePtr MinInDim::Clone(OpList operands) const {
  return MakeNode<MinInDim>(operands.at(0), dim_, keepdim_);
}

XlaOpVector MinInDim::Lower(LoweringContext* loctx) const {
  xla::Shape input_shape;
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp values =
      BuildMinInDim(XlaHelpers::MakeArray(input, &input_shape), dim_, keepdim_);
  xla::XlaOp indices =
      BuildArgMin(XlaHelpers::MakeArray(input), dim_, keepdim_);
  return ReturnOps({XlaHelpers::MaybeReshapeToScalar(values, input_shape),
                    XlaHelpers::MaybeReshapeToScalar(indices, input_shape)},
                   loctx);
}

std::string MinInDim::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_ << ", keepdim=" << keepdim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
