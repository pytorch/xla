#include "torch_xla/csrc/nll_loss.h"
#include "torch_xla/csrc/helpers.h"

namespace torch_xla {
namespace {

// Converts "indices" into a one-hot representation. "depth" is the size of the
// new axis to add. "axis" is the position at which to add the new axis.
// "on_value" and "off_value" represent the values to use for the on and off
// positions, respectively.
xla::XlaOp LabelsToOneHot(xla::XlaBuilder* builder, xla::int64 depth, int axis,
                          const xla::XlaOp indices, const xla::XlaOp on_value,
                          const xla::XlaOp off_value) {
  const auto indices_shape = XlaHelpers::ShapeOfXlaOp(indices);
  const int indices_dims = indices_shape.dimensions_size();
  const int output_dims = indices_dims + 1;

  // Expand the labels with a depth dimension for the classes.
  std::vector<xla::int64> output_dimensions(indices_shape.dimensions().begin(),
                                            indices_shape.dimensions().end());
  output_dimensions.insert(output_dimensions.begin() + axis, depth);

  // Build a iota tensor populated with values 0 through depth - 1.
  std::vector<int64_t> linspace_data(depth);
  std::iota(linspace_data.begin(), linspace_data.end(), 0);
  std::vector<xla::int64> linspace_dims(output_dims, 1);
  linspace_dims[axis] = depth;
  xla::Shape linspace_xla_shape = xla::ShapeUtil::MakeShapeWithDescendingLayout(
      xla::PrimitiveType::S64, linspace_dims);
  xla::BorrowingLiteral linspace_literal(
      reinterpret_cast<const char*>(linspace_data.data()), linspace_xla_shape);

  // Now compare the labels in index form to the iota tensor to get the one hot
  // format.
  std::vector<xla::int64> broadcast_dims(indices_shape.dimensions_size());
  std::iota(broadcast_dims.begin(), broadcast_dims.begin() + axis, 0);
  std::iota(broadcast_dims.begin() + axis, broadcast_dims.end(), axis + 1);
  xla::XlaOp linspace_xla;
  xla::XlaOp one_hot_bool = xla::Eq(
      indices, xla::ConstantLiteral(builder, linspace_literal), broadcast_dims);

  // Selects the user-provided off_value and on_value values.
  return xla::Select(one_hot_bool, xla::Broadcast(on_value, output_dimensions),
                     xla::Broadcast(off_value, output_dimensions));
}

}  // namespace

// Builds the NLLLoss for log-probabilities "logits" and class indices "labels".
xla::XlaOp BuildNllLoss(const xla::XlaOp& logits, const xla::XlaOp& labels) {
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
      /*off_value=*/zero);
  // Compute sum(-one_hot_labels * logits) / batch.
  xla::XlaOp mul = xla::Mul(xla::Neg(one_hot_labels), logits);
  xla::XlaComputation add_func =
      XlaHelpers::CreateAddComputation(logits_shape.element_type());
  xla::XlaOp batch = XlaHelpers::ScalarValue<float>(
      logits_shape.dimensions(0), logits_shape.element_type(), builder);
  return xla::ReduceAll(mul, zero, add_func) / batch;
}

// Builds the NLLLoss gradient for log-probabilities "logits" and class indices
// "labels".
xla::XlaOp BuildNllLossBackward(const xla::XlaOp& logits,
                                const xla::XlaOp& labels) {
  const int kBatchDim = 0;
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
      XlaHelpers::ScalarValue<float>(0, logits_shape.element_type(), builder));
  xla::XlaOp batch = XlaHelpers::ScalarValue<float>(
      logits_shape.dimensions(kBatchDim), logits_shape.element_type(), builder);
  // Compute -one_hot_labels / batch.
  return xla::Neg(one_hot_labels) / batch;
}

}  // namespace torch_xla
