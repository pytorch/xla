#include "torch_xla/csrc/ops/gather.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaOp LowerGather(const xla::XlaOp& input, const xla::XlaOp& index,
                       xla::int64 dim) {
  // TODO: Use the new xla::TorchGather() once it shows up.
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::Shape index_shape = XlaHelpers::ShapeOfXlaOp(index);
  xla::ShapeUtil::AppendMajorDimension(1, &index_shape);
  std::vector<xla::XlaOp> to_concat;
  to_concat.reserve(input_shape.rank());
  for (xla::int64 i = 0; i < input_shape.rank(); ++i) {
    if (i == dim) {
      to_concat.push_back(xla::Reshape(index, index_shape.dimensions()));
    } else {
      to_concat.push_back(xla::Iota(input.builder(), index_shape, i));
    }
  }
  xla::XlaOp gather_indices =
      xla::ConcatInDim(input.builder(), to_concat, input_shape.rank());
  std::vector<xla::int64> slice_sizes(input_shape.rank(), 1);
  xla::GatherDimensionNumbers gather_dnums;
  gather_dnums.set_index_vector_dim(input_shape.rank());
  for (xla::int64 i = 0; i < input_shape.rank(); ++i) {
    gather_dnums.add_collapsed_slice_dims(i);
    gather_dnums.add_start_index_map(i);
  }
  return xla::Gather(input, gather_indices, gather_dnums, slice_sizes);
}

xla::Shape NodeOutputShape(const Value& input, const Value& index,
                           xla::int64 dim) {
  auto lower_for_shape_fn =
      [&](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp { return LowerGather(operands[0], operands[1], dim); };
  return InferOutputShape({input.shape(), index.shape()}, lower_for_shape_fn);
}

}  // namespace

Gather::Gather(const Value& input, xla::int64 dim, const Value& index)
    : Node(
          ir::OpKind(at::aten::gather), {input, index},
          [&]() { return NodeOutputShape(input, index, dim); },
          /*num_outputs=*/1, xla::util::MHash(dim)),
      dim_(dim) {}

XlaOpVector Gather::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp index = loctx->GetOutputOp(operand(1));
  return ReturnOp(LowerGather(input, index, dim_), loctx);
}

std::string Gather::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
