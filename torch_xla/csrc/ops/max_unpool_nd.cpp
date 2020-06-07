#include "torch_xla/csrc/ops/max_unpool_nd.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/pooling.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, const Value& indices,
                           absl::Span<const xla::int64> output_size) {
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMaxUnpoolNd(GetCurrentDevice(), operands[0], operands[1],
                            output_size);
  };
  return InferOutputShape({input.shape(), indices.shape()}, shape_fn);
}

c10::Symbol MaxUnpoolNdSymbol(xla::int64 spatial_dim_count) {
  switch (spatial_dim_count) {
    case 2:
      return at::aten::max_unpool2d;
    case 3:
      return at::aten::max_unpool3d;
    default:
      XLA_ERROR() << "Invalid number of spatial dimensions: "
                  << spatial_dim_count;
  }
}

}  // namespace

MaxUnpoolNd::MaxUnpoolNd(const Value& input, const Value& indices,
                         std::vector<xla::int64> output_size)
    : Node(ir::OpKind(MaxUnpoolNdSymbol(output_size.size())), {input, indices},
           [&]() { return NodeOutputShape(input, indices, output_size); },
           /*num_outputs=*/1, xla::util::MHash(output_size)),
      output_size_(std::move(output_size)) {}

NodePtr MaxUnpoolNd::Clone(OpList operands) const {
  return MakeNode<MaxUnpoolNd>(operands.at(0), operands.at(1), output_size_);
}

XlaOpVector MaxUnpoolNd::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp indices = loctx->GetOutputOp(operand(1));
  xla::XlaOp output =
      BuildMaxUnpoolNd(loctx->device(), input, indices, output_size_);
  return ReturnOp(output, loctx);
}

std::string MaxUnpoolNd::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", output_size=("
     << absl::StrJoin(output_size_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
