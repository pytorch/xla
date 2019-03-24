#include "torch_xla/csrc/ops/unselect.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaOp LowerUnselect(const xla::XlaOp& target, const xla::XlaOp& source,
                         xla::int64 dim, xla::int64 start, xla::int64 end,
                         xla::int64 stride) {
  xla::Shape target_shape = XlaHelpers::ShapeOfXlaOp(target);
  xla::Shape source_shape = XlaHelpers::ShapeOfXlaOp(source);
  if (target_shape.dimensions(dim) == source_shape.dimensions(dim)) {
    // Shortcut for unselects which are fully covering selects.
    XLA_CHECK_EQ(start, 0);
    XLA_CHECK_EQ(stride, 1);
    XLA_CHECK_EQ(end, target_shape.dimensions(dim));
    return source;
  }

  xla::PrimitiveType pred_type =
      GetDevicePrimitiveType(xla::PrimitiveType::PRED, /*device=*/nullptr);
  xla::XlaOp source_true = XlaHelpers::ScalarBroadcast(
      1, pred_type, source_shape.dimensions(), source.builder());
  xla::XlaOp pred_zero =
      XlaHelpers::ScalarValue(0, pred_type, target.builder());
  xla::XlaOp zero =
      XlaHelpers::ScalarValue(0, target_shape.element_type(), target.builder());
  xla::PaddingConfig padding_config;
  for (xla::int64 i = 0; i < target_shape.rank(); ++i) {
    auto* dims = padding_config.add_dimensions();
    if (i == dim) {
      dims->set_edge_padding_low(start);
      dims->set_interior_padding(stride - 1);

      xla::int64 size = start + source_shape.dimensions(i) +
                        (source_shape.dimensions(i) - 1) * (stride - 1);
      dims->set_edge_padding_high(target_shape.dimensions(i) - size);
    } else {
      XLA_CHECK_EQ(target_shape.dimensions(i), source_shape.dimensions(i))
          << target_shape << " vs. " << source_shape;
      dims->set_edge_padding_low(0);
      dims->set_interior_padding(0);
      dims->set_edge_padding_high(0);
    }
  }
  xla::XlaOp padded_source = xla::Pad(source, zero, padding_config);
  xla::XlaOp mask = xla::Pad(source_true, pred_zero, padding_config);
  return xla::Select(mask, padded_source, target);
}

}  // namespace

Unselect::Unselect(const Value& target, const Value& source, xla::int64 dim,
                   xla::int64 start, xla::int64 end, xla::int64 stride)
    : Node(xla_unselect, {target, source}, target.shape(),
           /*num_outputs=*/1, xla::util::MHash(dim, start, end, stride)),
      dim_(dim),
      start_(start),
      end_(end),
      stride_(stride) {}

XlaOpVector Unselect::Lower(LoweringContext* loctx) const {
  xla::XlaOp target = loctx->GetOutputOp(operand(0));
  xla::XlaOp source = loctx->GetOutputOp(operand(1));
  xla::XlaOp output =
      LowerUnselect(target, source, dim_, start_, end_, stride_);
  return ReturnOp(output, loctx);
}

std::string Unselect::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_ << ", start=" << start_
     << ", end=" << end_ << ", stride=" << stride_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
