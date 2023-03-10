#include "torch_xla/csrc/nms_op.h"

#include <torch/csrc/lazy/core/util.h>

#include <limits>

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/loops.h"
#include "tensorflow/compiler/xla/client/lib/sorting.h"
#include "tensorflow/compiler/xla/util.h"
#include "third_party/xla_client/debug_macros.h"
#include "third_party/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"

// Code extracted from:
// https://github.com/tensorflow/tensorflow/blob/dc4c6d305ba3d2de4a795ec77b483b0fa695b9ee/tensorflow/compiler/tf2xla/kernels/image_ops.cc#L399

namespace torch_xla {
namespace {

struct WhileCondFn {
  WhileCondFn(int64_t num_boxes, int64_t output_size)
      : num_boxes(num_boxes), output_size(output_size) {}

  xla::StatusOr<xla::XlaOp> operator()(absl::Span<const xla::XlaOp> values,
                                       xla::XlaBuilder* builder) const {
    xla::XlaOp row_idx = values[0];
    xla::XlaOp row_in_bounds =
        xla::Lt(row_idx, xla::ConstantR0<int32_t>(builder, num_boxes));
    xla::XlaOp num_outputs_so_far = values[1];
    xla::XlaOp results_not_full = xla::Lt(
        num_outputs_so_far, xla::ConstantR0<int32_t>(builder, output_size));
    return xla::And(row_in_bounds, results_not_full);
  }

  int64_t num_boxes;
  int64_t output_size;
};

// Process the boxes one-by-one using the iou matrix mask.
// This implementation uses a correct, but greedy, sequential algorithm
// to ensure that suppressed boxes cannot themselves suppress other
// boxes.
struct SuppressBodyFn {
  explicit SuppressBodyFn(int64_t num_boxes) : num_boxes(num_boxes) {}

  xla::StatusOr<std::vector<xla::XlaOp>> operator()(
      absl::Span<const xla::XlaOp> values, xla::XlaBuilder* builder) const {
    xla::XlaOp row_idx = values[0];
    xla::XlaOp num_outputs_so_far = values[1];
    xla::XlaOp iou_mask = values[2];
    xla::XlaOp included_iou = values[3];
    xla::XlaOp zero = xla::Zero(builder, xla::PrimitiveType::S32);
    xla::XlaOp one = xla::One(builder, xla::PrimitiveType::S32);
    // Determine if current elem is active using a slice.
    // The only reason we need an explicit vector is because some old GCCs can't
    // deduce the right type for MakeConstSpan, and providing a single-value
    // initializer list directly uses the wrong overload. Delete this once the
    // deprecated overload is gone.
    std::vector<xla::XlaOp> row_idx_vector = {row_idx};
    xla::XlaOp active_elem =
        xla::DynamicSlice(included_iou, row_idx_vector, {1});
    active_elem = xla::Reshape(active_elem, {});
    // Increment output count iff current elem is not suppressed.
    num_outputs_so_far =
        xla::Select(active_elem, num_outputs_so_far + one, num_outputs_so_far);
    // Slice out the row_idx.
    xla::XlaOp row_iou =
        xla::DynamicSlice(iou_mask, {row_idx, zero}, {1, num_boxes});
    // Remove the diagonal from consideration. An elem cannot suppress
    // itself.
    row_iou = xla::DynamicUpdateSlice(
        row_iou, xla::ConstantR2FromArray2D<bool>(builder, {{false}}),
        {zero, row_idx});
    // Create a suppression by inverting polarity.
    row_iou = xla::Reshape(row_iou, {num_boxes});
    xla::XlaOp supp_mask = xla::Not(row_iou);
    // Update mask iff current elem is not suppressed.
    included_iou = xla::Select(xla::Broadcast(active_elem, {num_boxes}),
                               xla::And(included_iou, supp_mask), included_iou);
    return std::vector<xla::XlaOp>{row_idx + one, num_outputs_so_far, iou_mask,
                                   included_iou};
  }

  int64_t num_boxes;
};

xla::XlaOp NmsGather(xla::XlaOp input, absl::Span<const int64_t> input_sizes,
                     xla::XlaOp indices,
                     absl::Span<const int64_t> indices_sizes, int64_t axis) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  int64_t num_indices = xla::util::Multiply<int64_t>(indices_sizes);
  if (num_indices == 0) {
    std::vector<int64_t> output_sizes =
        torch::lazy::ToVector<int64_t>(input_sizes);
    output_sizes.erase(std::next(output_sizes.begin(), axis));
    return xla::Broadcast(
        xla::Zero(input.builder(), input_shape.element_type()), output_sizes);
  }

  // Example of a 1-D gather with axis=1, pulling two [3,1] tensors out of a
  // tensor of shape [3,3].
  //
  //  operand = s32[3,3] parameter(0)
  //  indices = s32[2] parameter(1)
  //  gather = s32[3,2] gather(operand, indices),
  //       offset_dims={0},
  //       collapsed_slice_dims={1},
  //       start_index_map={1},
  //       index_vector_dim=1,
  //       slice_sizes={3, 1}
  //
  //
  // Example of an N-D gather pulling out slices of shape [1,1,2] out of a
  // tensor of shape [3,3,2].
  //
  //  operand = s32[3,3,2] parameter(0)
  //  indices = s32[2,2] parameter(1)
  //  gather = s32[2,2] gather(operand, indices),
  //       offset_dims={1},
  //       collapsed_slice_dims={0,1},
  //       start_index_map={0,1},
  //       index_vector_dim=0,
  //       slice_sizes={1,1,2}
  xla::GatherDimensionNumbers dim_numbers;
  std::vector<int64_t> slice_sizes;
  for (int64_t i = 0; i < input_sizes.size(); ++i) {
    int64_t window_bound;
    if (i == axis) {
      dim_numbers.add_collapsed_slice_dims(i);
      window_bound = 1;
    } else {
      window_bound = input_sizes[i];
    }
    slice_sizes.push_back(window_bound);
    if (i < axis) {
      dim_numbers.add_offset_dims(i);
    } else if (i > axis) {
      dim_numbers.add_offset_dims(i + indices_sizes.size() - 1);
    }
  }
  dim_numbers.set_index_vector_dim(indices_sizes.size());
  dim_numbers.add_start_index_map(axis);
  return xla::Gather(input, indices, dim_numbers, slice_sizes);
}

}  // namespace

NmsResult BuildNms(xla::XlaOp boxes, xla::XlaOp scores,
                   xla::XlaOp score_threshold, xla::XlaOp iou_threshold,
                   int64_t output_size) {
  const xla::Shape& boxes_shape = XlaHelpers::ShapeOfXlaOp(boxes);
  int64_t num_boxes = boxes_shape.dimensions(0);
  const xla::Shape& scores_shape = XlaHelpers::ShapeOfXlaOp(scores);
  XLA_CHECK_EQ(boxes_shape.rank(), 2);
  XLA_CHECK_EQ(boxes_shape.dimensions(1), 4);
  XLA_CHECK_EQ(scores_shape.rank(), 1);
  XLA_CHECK_EQ(scores_shape.dimensions(0), num_boxes);
  XLA_CHECK_LT(num_boxes, std::numeric_limits<int32_t>::max());
  XLA_CHECK_GE(output_size, 0);
  XLA_CHECK_LT(output_size, std::numeric_limits<int32_t>::max());

  xla::XlaBuilder* builder = boxes.builder();
  // Choose a more convenient layout.
  xla::XlaOp boxes_transposed = xla::Transpose(boxes, {1, 0});
  xla::XlaOp boxes_sorted = xla::GetTupleElement(
      xla::Sort({xla::Broadcast(scores, {4}), boxes_transposed},
                xla::CreateScalarGtComputation(
                    {scores_shape.element_type(), boxes_shape.element_type()},
                    builder),
                /*dimension=*/1),
      1);
  // Track the mapping of indices into sorted domain.
  xla::XlaOp iota_indices =
      xla::Iota(builder, xla::PrimitiveType::S32, num_boxes);
  xla::XlaOp indices_sort = xla::Sort(
      {scores, iota_indices},
      xla::CreateScalarGtComputation(
          {scores_shape.element_type(), xla::PrimitiveType::S32}, builder));
  xla::XlaOp indices_sorted = xla::GetTupleElement(indices_sort, 1);
  xla::XlaOp scores_sorted = xla::GetTupleElement(indices_sort, 0);

  // Shapes are henceforth [1, num_boxes]. 'c_y0' denotes 'coordinate' y0.
  xla::XlaOp c_y0 = xla::Reshape(xla::SliceInDim(boxes_sorted,
                                                 /*start_index=*/0,
                                                 /*limit_index=*/1,
                                                 /*stride=*/1,
                                                 /*dimno=*/0),
                                 {num_boxes});
  xla::XlaOp c_x0 = xla::Reshape(xla::SliceInDim(boxes_sorted,
                                                 /*start_index=*/1,
                                                 /*limit_index=*/2,
                                                 /*stride=*/1,
                                                 /*dimno=*/0),
                                 {num_boxes});
  xla::XlaOp c_y1 = xla::Reshape(xla::SliceInDim(boxes_sorted,
                                                 /*start_index=*/2,
                                                 /*limit_index=*/3,
                                                 /*stride=*/1,
                                                 /*dimno=*/0),
                                 {num_boxes});
  xla::XlaOp c_x1 = xla::Reshape(xla::SliceInDim(boxes_sorted,
                                                 /*start_index=*/3,
                                                 /*limit_index=*/4,
                                                 /*stride=*/1,
                                                 /*dimno=*/0),
                                 {num_boxes});

  xla::XlaOp y1 = xla::Select(xla::Le(c_y0, c_y1), c_y0, c_y1);
  xla::XlaOp y2 = xla::Select(xla::Le(c_y0, c_y1), c_y1, c_y0);
  xla::XlaOp x1 = xla::Select(xla::Le(c_x0, c_x1), c_x0, c_x1);
  xla::XlaOp x2 = xla::Select(xla::Le(c_x0, c_x1), c_x1, c_x0);
  xla::XlaOp area = (y2 - y1) * (x2 - x1);

  // Shapes are henceforth [1, num_boxes].
  y1 = xla::Broadcast(y1, {1});
  y2 = xla::Broadcast(y2, {1});
  x1 = xla::Broadcast(x1, {1});
  x2 = xla::Broadcast(x2, {1});
  area = xla::Broadcast(area, {1});

  // Shapes are henceforth [num_boxes, num_boxes].
  xla::XlaOp i_xmin = xla::Max(x1, xla::Transpose(x1, {1, 0}));
  xla::XlaOp i_ymin = xla::Max(y1, xla::Transpose(y1, {1, 0}));
  xla::XlaOp i_xmax = xla::Min(x2, xla::Transpose(x2, {1, 0}));
  xla::XlaOp i_ymax = xla::Min(y2, xla::Transpose(y2, {1, 0}));
  auto square_zero = xla::ZerosLike(i_xmin);

  xla::XlaOp i_area = xla::Max(i_xmax - i_xmin, square_zero) *
                      xla::Max(i_ymax - i_ymin, square_zero);
  xla::XlaOp u_area = area + xla::Transpose(area, {1, 0}) - i_area;
  xla::XlaOp iou = i_area / u_area;

  xla::XlaOp iou_threshold_mask = xla::Gt(iou, iou_threshold + square_zero);
  xla::XlaOp included_iou =
      xla::Broadcast(xla::ConstantR0<bool>(builder, true), {num_boxes});
  if (boxes_shape.is_dynamic_dimension(0)) {
    // Update included_iou's size to match boxes actual size.
    included_iou = xla::SetDimensionSize(
        included_iou, XlaHelpers::GetDimensionsSize({boxes}, {0}).size, 0);
  }

  xla::XlaOp zero_s32 = xla::Zero(builder, xla::PrimitiveType::S32);
  xla::XlaOp one_s32 = xla::One(builder, xla::PrimitiveType::S32);
  std::vector<xla::XlaOp> init_values;
  init_values.reserve(4);
  init_values.push_back(zero_s32);  // col_idx
  init_values.push_back(zero_s32);  // num_outputs
  init_values.push_back(iou_threshold_mask);
  init_values.push_back(included_iou);

  auto suppress_loop_result = ConsumeValue(xla::WhileLoopHelper(
      WhileCondFn(num_boxes, output_size), SuppressBodyFn(num_boxes),
      init_values, "BoxSuppressLoop", builder));

  xla::XlaOp included_score =
      xla::Gt(scores_sorted, xla::Broadcast(score_threshold, {num_boxes}));
  xla::XlaOp included = xla::And(included_score, suppress_loop_result[3]);

  // Only consider boxes over which we have iterated. This allows for accurate
  // counting. DynamicSlice would require knowledge of the size of the output.
  xla::XlaOp valid_elem = xla::Lt(
      iota_indices, xla::Broadcast(suppress_loop_result[0], {num_boxes}));
  included = xla::And(included, valid_elem);

  xla::XlaOp neg_inf = xla::Broadcast(
      xla::MinValue(builder, scores_shape.element_type()), {num_boxes});
  xla::XlaOp scores_included = xla::Select(included, scores_sorted, neg_inf);
  xla::XlaOp output_tuple = xla::TopK(scores_included, output_size);
  xla::XlaOp selected_indices_sorted = xla::GetTupleElement(output_tuple, 1);
  // Calculate num_valid.
  // Note: num_valid cannot be taken from the loop outputs, because outputs
  // can be suppressed by score threshold.
  xla::XlaOp ones_included =
      xla::Select(included, xla::Broadcast(one_s32, {num_boxes}),
                  xla::Broadcast(zero_s32, {num_boxes}));
  // num_valid is scalar. torch::lazy::Value should be bound by output_size.
  xla::XlaOp num_valid_total = xla::Reduce(
      ones_included,
      /*init_value=*/zero_s32,
      /*computation=*/
      xla::CreateScalarAddComputation(xla::PrimitiveType::S32, builder),
      /*dimensions_to_reduce=*/{0});
  xla::XlaOp num_valid =
      xla::Min(num_valid_total, xla::ConstantR0<int32_t>(builder, output_size));

  // Re-index into the original scores input tensor, using a Gather.
  // Boxes were suppressed in the sorted domain.
  xla::XlaOp selected_indices =
      NmsGather(indices_sorted, scores_shape.dimensions(),
                selected_indices_sorted, {output_size},
                /*axis=*/0);
  return {selected_indices, num_valid};
}

}  // namespace torch_xla
