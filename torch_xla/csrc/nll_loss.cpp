#include "torch_xla/csrc/nll_loss.h"

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace {

struct WeightScale {
  xla::XlaOp weight;
  xla::XlaOp scale;
};

// Build a iota tensor populated with values 0 through depth - 1, with the
// exception of ignore_index which is set to -1 (if between 0 and depth - 1).
// This allows the ignored index to be ignored by the one-hot conversion.
xla::XlaOp OneHotIota(xla::XlaBuilder* builder, xla::int64 depth, int axis,
                      const xla::Shape& indices_shape, int ignore_index) {
  int indices_dims = indices_shape.rank();
  std::vector<xla::int64> linspace_dims(indices_dims + 1, 1);
  linspace_dims[axis] = depth;
  xla::Shape linspace_xla_shape =
      xla::ShapeUtil::MakeShape(indices_shape.element_type(), linspace_dims);
  xla::XlaOp iota = xla::Iota(builder, linspace_xla_shape, axis);
  if (ignore_index >= 0 && ignore_index < depth) {
    xla::XlaOp ignore_index_op =
        xla::Broadcast(XlaHelpers::ScalarValue<xla::int64>(
                           ignore_index, indices_shape.element_type(), builder),
                       linspace_dims);
    xla::XlaOp invalid_index =
        xla::Broadcast(XlaHelpers::ScalarValue<xla::int64>(
                           -1, indices_shape.element_type(), builder),
                       linspace_dims);
    iota = xla::Select(xla::Eq(iota, ignore_index_op), invalid_index, iota);
  }
  return iota;
}

// Converts "indices" into a one-hot representation. "depth" is the size of the
// new axis to add. "axis" is the position at which to add the new axis.
// "on_value" and "off_value" represent the values to use for the on and off
// positions, respectively. If "ignore_index" is a valid class, it'll be
// considered off.
xla::XlaOp LabelsToOneHot(xla::XlaBuilder* builder, xla::int64 depth, int axis,
                          xla::XlaOp indices, xla::XlaOp on_value,
                          xla::XlaOp off_value, int ignore_index) {
  const xla::Shape& indices_shape = XlaHelpers::ShapeOfXlaOp(indices);

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

WeightScale GetMaskedWeight(const absl::optional<xla::XlaOp>& weight,
                            const xla::Shape& logits_shape, xla::XlaOp labels,
                            xla::XlaOp one_hot_labels, int ignore_index) {
  const xla::Shape& labels_shape = XlaHelpers::ShapeOfXlaOp(labels);
  xla::XlaOp valid_bitmap = xla::Ne(
      labels, XlaHelpers::ScalarValue<xla::int64>(
                  ignore_index, labels_shape.element_type(), labels.builder()));
  xla::XlaOp xweight;
  if (weight) {
    xweight = xla::BroadcastInDim(*weight, logits_shape.dimensions(), {1});
  } else {
    xweight =
        XlaHelpers::ScalarBroadcast<float>(1.0, logits_shape, labels.builder());
  }
  xla::XlaOp zeros =
      XlaHelpers::ScalarBroadcast<float>(0.0, logits_shape, labels.builder());
  xla::XlaOp xvalid_bitmap =
      xla::BroadcastInDim(valid_bitmap, logits_shape.dimensions(), {0});
  xla::XlaOp result_weight =
      xla::Select(xvalid_bitmap, xweight, zeros) * one_hot_labels;

  xla::XlaComputation add_func =
      XlaHelpers::CreateAddComputation(logits_shape.element_type());
  xla::XlaOp zero = xla::Zero(labels.builder(), logits_shape.element_type());
  xla::XlaOp one = xla::One(labels.builder(), logits_shape.element_type());
  xla::XlaOp scale = xla::ReduceAll(result_weight, zero, add_func);
  scale = xla::Select(xla::Ne(scale, zero), scale, one);
  return {result_weight, scale};
}

}  // namespace

// Builds the NLLLoss for log-probabilities "logits" and class indices "labels".
xla::XlaOp BuildNllLoss(xla::XlaOp logits, xla::XlaOp labels,
                        const absl::optional<xla::XlaOp>& weight,
                        int ignore_index, ReductionMode reduction_mode) {
  const int classes_axis = 1;
  const xla::Shape& logits_shape = XlaHelpers::ShapeOfXlaOp(logits);
  xla::XlaOp zero = xla::Zero(logits.builder(), logits_shape.element_type());
  xla::XlaOp one = xla::One(logits.builder(), logits_shape.element_type());
  xla::XlaOp one_hot_labels = LabelsToOneHot(
      /*builder=*/logits.builder(),
      /*depth=*/logits_shape.dimensions(1),
      /*axis=*/classes_axis,
      /*indices=*/labels,
      /*on_value=*/one,
      /*off_value=*/zero,
      /*ignore_index=*/ignore_index);
  xla::XlaOp mul = xla::Neg(one_hot_labels) * logits;
  WeightScale weight_scale = GetMaskedWeight(weight, logits_shape, labels,
                                             one_hot_labels, ignore_index);
  mul = mul * weight_scale.weight;
  xla::XlaComputation add_func =
      XlaHelpers::CreateAddComputation(logits_shape.element_type());
  if (reduction_mode == ReductionMode::kNone) {
    return xla::Reduce(mul, zero, add_func, {classes_axis});
  }
  xla::XlaOp sum = xla::ReduceAll(mul, zero, add_func);
  if (reduction_mode == ReductionMode::kSum) {
    return sum;
  }
  return sum / weight_scale.scale;
}

// Builds the NLLLoss gradient for log-probabilities "logits" and class indices
// "labels".
xla::XlaOp BuildNllLossBackward(xla::XlaOp grad_output, xla::XlaOp logits,
                                xla::XlaOp labels,
                                const absl::optional<xla::XlaOp>& weight,
                                const absl::optional<xla::XlaOp>& total_weight,
                                int ignore_index,
                                ReductionMode reduction_mode) {
  const int classes_axis = 1;
  const xla::Shape& logits_shape = XlaHelpers::ShapeOfXlaOp(logits);
  xla::XlaOp zero = xla::Zero(logits.builder(), logits_shape.element_type());
  xla::XlaOp one = xla::One(logits.builder(), logits_shape.element_type());
  xla::XlaOp one_hot_labels = LabelsToOneHot(
      /*builder=*/logits.builder(),
      /*depth=*/logits_shape.dimensions(1),
      /*axis=*/classes_axis,
      /*indices=*/labels,
      /*on_value=*/one,
      /*off_value=*/zero,
      /*ignore_index=*/ignore_index);

  const xla::Shape& grad_output_shape = XlaHelpers::ShapeOfXlaOp(grad_output);
  xla::XlaOp grad = grad_output;
  if (grad_output_shape.rank() == 1) {
    grad = xla::BroadcastInDim(grad, logits_shape.dimensions(), {0});
  }
  xla::XlaOp result = xla::Neg(one_hot_labels) * grad;
  WeightScale weight_scale = GetMaskedWeight(weight, logits_shape, labels,
                                             one_hot_labels, ignore_index);
  result = result * weight_scale.weight;
  if (reduction_mode != ReductionMode::kMean) {
    return result;
  }
  return result / weight_scale.scale;
}

}  // namespace torch_xla
