#include "torch_xla/csrc/ops/scatter.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaComputation CreateCombiner(xla::PrimitiveType type) {
  xla::XlaBuilder builder("ScatterCombiner");
  xla::Shape shape = xla::ShapeUtil::MakeShape(type, {});
  xla::Parameter(&builder, 0, shape, "p0");
  xla::Parameter(&builder, 1, shape, "p1");
  return ConsumeValue(builder.Build());
}

xla::XlaOp LowerScatter(const xla::XlaOp& input, const xla::XlaOp& index,
                        const xla::XlaOp& src, xla::int64 dim) {
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::Shape index_shape = XlaHelpers::ShapeOfXlaOp(index);
  xla::Shape src_shape = XlaHelpers::ShapeOfXlaOp(src);
  XLA_CHECK_EQ(src_shape.rank(), index_shape.rank());
  xla::XlaOp src_op = src;
  if (src_shape.dimensions() != index_shape.dimensions()) {
    std::vector<xla::int64> base_indices(src_shape.rank(), 0);
    src_op = BuildSlice(src_op, base_indices, index_shape.dimensions());
  }
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
  xla::XlaOp scatter_indices =
      xla::ConcatInDim(input.builder(), to_concat, input_shape.rank());
  xla::ScatterDimensionNumbers scatter_dnums;
  scatter_dnums.set_index_vector_dim(input_shape.rank());
  for (xla::int64 i = 0; i < input_shape.rank(); ++i) {
    scatter_dnums.add_inserted_window_dims(i);
    scatter_dnums.add_scatter_dims_to_operand_dims(i);
  }
  return xla::Scatter(input, scatter_indices, src_op,
                      CreateCombiner(input_shape.element_type()),
                      scatter_dnums);
}

}  // namespace

Scatter::Scatter(const Value& input, xla::int64 dim, const Value& index,
                 const Value& src)
    : Node(ir::OpKind(at::aten::scatter), {input, index, src}, input.shape(),
           /*num_outputs=*/1, xla::util::MHash(dim)),
      dim_(dim) {}

XlaOpVector Scatter::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp index = loctx->GetOutputOp(operand(1));
  xla::XlaOp src = loctx->GetOutputOp(operand(2));
  return ReturnOp(LowerScatter(input, index, src, dim_), loctx);
}

std::string Scatter::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
