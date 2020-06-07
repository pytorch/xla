#include "torch_xla/csrc/ops/max_unpool_nd_backward.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/pooling.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& grad_output, const Value& input,
                           const Value& indices,
                           absl::Span<const xla::int64> output_size) {
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMaxUnpoolNdBackward(operands[0], operands[1], operands[2],
                                    output_size);
  };
  return InferOutputShape({grad_output.shape(), input.shape(), indices.shape()},
                          shape_fn);
}

c10::Symbol MaxUnpoolNdBackwardSymbol(xla::int64 spatial_dim_count) {
  switch (spatial_dim_count) {
    case 2:
      return at::aten::max_unpool2d_backward;
    case 3:
      return at::aten::max_unpool3d_backward;
    default:
      XLA_ERROR() << "Invalid number of spatial dimensions: "
                  << spatial_dim_count;
  }
}

}  // namespace

MaxUnpoolNdBackward::MaxUnpoolNdBackward(const Value& grad_output,
                                         const Value& input,
                                         const Value& indices,
                                         std::vector<xla::int64> output_size)
    : Node(ir::OpKind(MaxUnpoolNdBackwardSymbol(output_size.size())),
           {grad_output, input, indices},
           [&]() {
             return NodeOutputShape(grad_output, input, indices, output_size);
           },
           /*num_outputs=*/1, xla::util::MHash(output_size)),
      output_size_(std::move(output_size)) {}

NodePtr MaxUnpoolNdBackward::Clone(OpList operands) const {
  return MakeNode<MaxUnpoolNdBackward>(operands.at(0), operands.at(1),
                                       operands.at(2), output_size_);
}

XlaOpVector MaxUnpoolNdBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp indices = loctx->GetOutputOp(operand(2));
  xla::XlaOp output =
      BuildMaxUnpoolNdBackward(grad_output, input, indices, output_size_);
  return ReturnOp(output, loctx);
}

std::string MaxUnpoolNdBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", output_size=("
     << absl::StrJoin(output_size_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
