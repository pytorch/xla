#include "torch_xla/csrc/nll_loss.h"

#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace {

// Build a iota tensor populated with values 0 through depth - 1, with the
// exception of ignore_index which is set to -1 (if between 0 and depth - 1).
// This allows the ignored index to be ignored by the one-hot conversion.
xla::XlaOp OneHotIota(xla::XlaBuilder* builder, xla::int64 depth, int axis,
                      const xla::Shape& indices_shape, int ignore_index) {
  int indices_dims = indices_shape.rank();
  int output_dims = indices_dims + 1;
  std::vector<xla::int64> linspace_dims(output_dims, 1);
  linspace_dims[axis] = depth;
  // Fast path, the ignored index is not a valid class. The one-hot conversion
  // will work correctly since it'll be different from all iota values.
  if (ignore_index < 0 || ignore_index >= depth) {
    xla::Shape linspace_xla_shape =
        xla::ShapeUtil::MakeShapeWithDescendingLayout(
            indices_shape.element_type(), linspace_dims);
    return xla::Iota(builder, linspace_xla_shape, axis);
  }
  // Slow path, need to materialize the iota values and poke a hole at the
  // ignored index position.
  auto iota_values = xla::util::Iota<xla::int64>(depth);
  iota_values[ignore_index] = -1;
  xla::XlaOp iota_rank1 = xla::ConstantR1(builder, xla::AsSlice(iota_values));
  iota_rank1 =
      xla::ConvertElementType(iota_rank1, indices_shape.element_type());
  return BuildExpand(iota_rank1, linspace_dims);
}

// Converts "indices" into a one-hot representation. "depth" is the size of the
// new axis to add. "axis" is the position at which to add the new axis.
// "on_value" and "off_value" represent the values to use for the on and off
// positions, respectively. If "ignore_index" is a valid class, it'll be
// considered off.
xla::XlaOp LabelsToOneHot(xla::XlaBuilder* builder, xla::int64 depth, int axis,
                          const xla::XlaOp& indices, const xla::XlaOp& on_value,
                          const xla::XlaOp& off_value, int ignore_index) {
  xla::Shape indices_shape = XlaHelpers::ShapeOfXlaOp(indices);

  // Expand the labels with a depth dimension for the classes.
  std::vector<xla::int64> output_dimensions(indices_shape.dimensions().begin(),
                                            indices_shape.dimensions().end());
  output_dimensions.insert(output_dimensions.begin() + axis, depth);

  xla::XlaOp iota = OneHotIota(/*builder=*/builder, /*depth=*/depth,
                               /*axis=*/axis, /*indices_shape=*/indices_shape,
                               /*ignore_index=*/ignore_index);

  // Now compare the labels in index form to the iota tensor to get the one hot
  // format.
  std::vector<xla::int64> broadcast_dims(indices_shape.rank());
  std::iota(broadcast_dims.begin(), broadcast_dims.begin() + axis, 0);
  std::iota(broadcast_dims.begin() + axis, broadcast_dims.end(), axis + 1);
  xla::XlaOp one_hot_bool = xla::Eq(indices, iota, broadcast_dims);

  // Selects the user-provided off_value and on_value values.
  return xla::Select(one_hot_bool, xla::Broadcast(on_value, output_dimensions),
                     xla::Broadcast(off_value, output_dimensions));
}

// Count the number of labels which aren't equal to "ignore_index".
xla::XlaOp ValidLabelsCount(const xla::XlaOp& labels, int ignore_index) {
  xla::Shape labels_shape = XlaHelpers::ShapeOfXlaOp(labels);
  xla::XlaOp valid_bitmap = xla::Ne(
      labels, XlaHelpers::ScalarValue<xla::int64>(
                  ignore_index, labels_shape.element_type(), labels.builder()));
  valid_bitmap = xla::ConvertElementType(valid_bitmap, xla::PrimitiveType::S32);
  xla::XlaOp zero = XlaHelpers::ScalarValue<xla::int32>(
      0, xla::PrimitiveType::S32, labels.builder());
  xla::XlaComputation add_func =
      XlaHelpers::CreateAddComputation(xla::PrimitiveType::S32);
  return xla::ReduceAll(valid_bitmap, zero, add_func);
}

}  // namespace

// Builds the NLLLoss for log-probabilities "logits" and class indices "labels".
xla::XlaOp BuildNllLoss(const xla::XlaOp& logits, const xla::XlaOp& labels,
                        int ignore_index) {
  xla::XlaBuilder* builder = logits.builder();
  xla::Shape logits_shape = XlaHelpers::ShapeOfXlaOp(logits);
  xla::XlaOp zero =
      XlaHelpers::ScalarValue<float>(0, logits_shape.element_type(), builder);
  xla::XlaOp one_hot_labels = LabelsToOneHot(
      /*builder=*/builder,
      /*depth=*/logits_shape.dimensions(1),
      /*axis=*/1,
      /*indices=*/labels,
      /*on_value=*/
      XlaHelpers::ScalarValue<float>(1, logits_shape.element_type(), builder),
      /*off_value=*/zero,
      /*ignore_index=*/ignore_index);
  // Compute sum(-one_hot_labels * logits) / batch.
  xla::XlaOp mul = xla::Mul(xla::Neg(one_hot_labels), logits);
  xla::XlaOp batch = xla::ConvertElementType(
      ValidLabelsCount(labels, ignore_index), logits_shape.element_type());
  xla::XlaComputation add_func =
      XlaHelpers::CreateAddComputation(logits_shape.element_type());
  return xla::Select(xla::Ne(batch, zero),
                     xla::ReduceAll(mul, zero, add_func) / batch,
                     xla::ReduceAll(mul, zero, add_func));
}

// Builds the NLLLoss gradient for log-probabilities "logits" and class indices
// "labels".
xla::XlaOp BuildNllLossBackward(const xla::XlaOp& logits,
                                const xla::XlaOp& labels, int ignore_index) {
  xla::XlaBuilder* builder = logits.builder();
  xla::Shape logits_shape = XlaHelpers::ShapeOfXlaOp(logits);
  xla::XlaOp one_hot_labels = LabelsToOneHot(
      /*builder=*/builder,
      /*depth=*/logits_shape.dimensions(1),
      /*axis=*/1,
      /*indices=*/labels,
      /*on_value=*/
      XlaHelpers::ScalarValue<float>(1, logits_shape.element_type(), builder),
      /*off_value=*/
      XlaHelpers::ScalarValue<float>(0, logits_shape.element_type(), builder),
      /*ignore_index=*/ignore_index);
  xla::XlaOp batch = xla::ConvertElementType(
      ValidLabelsCount(labels, ignore_index), logits_shape.element_type());
  // Compute -one_hot_labels / batch.
  xla::XlaOp zero =
      XlaHelpers::ScalarValue<float>(0, logits_shape.element_type(), builder);
  return xla::Select(xla::Ne(batch, zero),
                     xla::Neg(one_hot_labels) / batch,
                     xla::Neg(one_hot_labels));
}

}  // namespace torch_xla
