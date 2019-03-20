#include "torch_xla/csrc/ops/index_select.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaOp LowerIndexSelect(const xla::XlaOp& input, const xla::XlaOp& index,
                            xla::int64 dim) {
  // TODO: Use the new xla::TorchIndexSelect() once it shows up.
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::Shape index_shape = XlaHelpers::ShapeOfXlaOp(index);
  std::vector<xla::int64> slice_sizes = input_shape.dimensions();
  slice_sizes[dim] = 1;
  xla::GatherDimensionNumbers gather_dnums;
  for (xla::int64 i = 0; i < input_shape.rank(); ++i) {
    if (i != dim) {
      gather_dnums.add_offset_dims(i);
    }
  }
  gather_dnums.set_index_vector_dim(index_shape.rank());
  gather_dnums.add_collapsed_slice_dims(dim);
  gather_dnums.add_start_index_map(dim);
  return xla::Gather(input, index, gather_dnums, slice_sizes);
}

xla::Shape NodeOutputShape(const Value& input, const Value& index,
                           xla::int64 dim) {
  auto lower_for_shape_fn =
      [&](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp { return LowerIndexSelect(operands[0], operands[1], dim); };
  return InferOutputShape({input.shape(), index.shape()}, lower_for_shape_fn);
}

}  // namespace

IndexSelect::IndexSelect(const Value& input, xla::int64 dim, const Value& index)
    : Node(
          ir::OpKind(at::aten::index_select), {input, index},
          [&]() { return NodeOutputShape(input, index, dim); },
          /*num_outputs=*/1, xla::util::MHash(dim)),
      dim_(dim) {}

XlaOpVector IndexSelect::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp index = loctx->GetOutputOp(operand(1));
  return ReturnOp(LowerIndexSelect(input, index, dim_), loctx);
}

std::string IndexSelect::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
