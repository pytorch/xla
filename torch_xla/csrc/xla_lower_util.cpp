#include "torch_xla/csrc/xla_lower_util.h"

#include <algorithm>
#include <vector>

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"

namespace torch_xla {
namespace {

std::pair<xla::XlaOp, xla::Shape> DotExpand(const xla::XlaOp& op,
                                            const xla::Shape& op_shape,
                                            const xla::Shape& to_shape) {
  xla::int64 rank_delta = to_shape.rank() - op_shape.rank();
  XLA_CHECK_GT(rank_delta, 0);

  std::vector<xla::int64> reshape_sizes(to_shape.rank(), 1);
  std::copy(op_shape.dimensions().begin(), op_shape.dimensions().end(),
            reshape_sizes.begin() + rank_delta);
  xla::XlaOp result = xla::Reshape(op, reshape_sizes);

  std::vector<xla::int64> broadcasted_sizes(
      to_shape.dimensions().begin(),
      to_shape.dimensions().begin() + rank_delta);
  broadcasted_sizes.insert(broadcasted_sizes.end(),
                           op_shape.dimensions().begin(),
                           op_shape.dimensions().end());
  return std::make_pair(
      xla::BroadcastInDim(result, broadcasted_sizes,
                          xla::util::Iota<xla::int64>(to_shape.rank())),
      xla::ShapeUtil::MakeShape(op_shape.element_type(), broadcasted_sizes));
}

std::pair<xla::XlaOp, xla::XlaOp> DotBroadcast(const xla::XlaOp& lhs,
                                               const xla::Shape& lhs_shape,
                                               const xla::XlaOp& rhs,
                                               const xla::Shape& rhs_shape) {
  auto lhs_dimensions = lhs_shape.dimensions();
  auto rhs_dimensions = rhs_shape.dimensions();
  XLA_CHECK_EQ(lhs_dimensions.size(), rhs_dimensions.size());
  for (xla::int64 i = 0; i < lhs_dimensions.size() - 2; ++i) {
    if (lhs_dimensions[i] == rhs_dimensions[i]) {
      continue;
    }
    if (lhs_dimensions[i] == 1) {
      lhs_dimensions[i] = rhs_dimensions[i];
    } else if (rhs_dimensions[i] == 1) {
      rhs_dimensions[i] = lhs_dimensions[i];
    } else {
      XLA_ERROR() << "Unsupported DotBroadcast: " << lhs_shape << " vs. "
                  << rhs_shape;
    }
  }

  xla::XlaOp broadcasted_lhs = lhs;
  xla::XlaOp broadcasted_rhs = rhs;
  if (lhs_dimensions != lhs_shape.dimensions()) {
    broadcasted_lhs =
        xla::BroadcastInDim(lhs, lhs_dimensions,
                            xla::util::Iota<xla::int64>(lhs_dimensions.size()));
  }
  if (rhs_dimensions != rhs_shape.dimensions()) {
    broadcasted_rhs =
        xla::BroadcastInDim(rhs, rhs_dimensions,
                            xla::util::Iota<xla::int64>(rhs_dimensions.size()));
  }
  return std::make_pair(broadcasted_lhs, broadcasted_rhs);
}

}  // namespace

xla::XlaOp CreateMatMul(const xla::XlaOp& lhs, const xla::XlaOp& rhs) {
  const auto precision_level = XlaHelpers::mat_mul_precision();
  xla::PrecisionConfig precision_config =
      XlaHelpers::BuildPrecisionConfig(precision_level);
  // Expand cases in https://pytorch.org/docs/stable/torch.html#torch.matmul
  xla::Shape lhs_shape = XlaHelpers::ShapeOfXlaOp(lhs);
  xla::Shape rhs_shape = XlaHelpers::ShapeOfXlaOp(rhs);
  if ((lhs_shape.rank() == 1 && rhs_shape.rank() == 1) ||
      (lhs_shape.rank() == 2 && rhs_shape.rank() == 2) ||
      (lhs_shape.rank() == 2 && rhs_shape.rank() == 1)) {
    return xla::Dot(lhs, rhs);
  }
  if (lhs_shape.rank() == 1 && rhs_shape.rank() == 2) {
    xla::XlaOp reshaped_lhs = xla::Reshape(lhs, {1, lhs_shape.dimensions(0)});
    return xla::Reshape(xla::Dot(reshaped_lhs, rhs), {rhs_shape.dimensions(1)});
  }
  if (lhs_shape.rank() >= 1 && rhs_shape.rank() >= 1 &&
      (lhs_shape.rank() >= 3 || rhs_shape.rank() >= 3)) {
    xla::XlaOp reshaped_lhs = lhs;
    xla::XlaOp reshaped_rhs = rhs;
    if (lhs_shape.rank() > rhs_shape.rank()) {
      std::tie(reshaped_rhs, rhs_shape) =
          DotExpand(reshaped_rhs, rhs_shape, lhs_shape);
    } else if (rhs_shape.rank() > lhs_shape.rank()) {
      std::tie(reshaped_lhs, lhs_shape) =
          DotExpand(reshaped_lhs, lhs_shape, rhs_shape);
    }
    std::tie(reshaped_lhs, reshaped_rhs) =
        DotBroadcast(reshaped_lhs, lhs_shape, reshaped_rhs, rhs_shape);

    // At this point lhs and rhs ranks are the same, use left rank in code
    // below.
    xla::DotDimensionNumbers dims;
    for (xla::int64 i = 0; i < lhs_shape.rank() - 2; ++i) {
      dims.add_lhs_batch_dimensions(i);
      dims.add_rhs_batch_dimensions(i);
    }
    dims.add_lhs_contracting_dimensions(lhs_shape.rank() - 1);
    dims.add_rhs_contracting_dimensions(lhs_shape.rank() - 2);

    return xla::DotGeneral(reshaped_lhs, reshaped_rhs, dims, &precision_config);
  }
  XLA_ERROR() << "Unsupported matmul operation: matmul(" << lhs_shape << ", "
              << rhs_shape << ")";
}

xla::XlaOp BuildDropout(const xla::XlaOp& input, float probability) {
  xla::Shape shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp zero =
      XlaHelpers::ScalarValue<float>(0, shape.element_type(), input.builder());
  xla::XlaOp one =
      XlaHelpers::ScalarValue<float>(1, shape.element_type(), input.builder());
  xla::XlaOp prob =
      XlaHelpers::ScalarBroadcast<float>(probability, shape, input.builder());
  xla::XlaOp noise = xla::RngUniform(zero, one, shape);
  xla::XlaOp mask =
      xla::ConvertElementType(xla::Lt(noise, prob), shape.element_type());
  if (probability > 0.0f) {
    mask = mask / prob;
  }
  return input * mask;
}

std::vector<xla::XlaOp> CreateBroadcastTensors(
    tensorflow::gtl::ArraySlice<const xla::XlaOp> operands) {
  xla::Shape result_shape = XlaHelpers::ShapeOfXlaOp(operands.front());
  std::vector<xla::Shape> operand_shapes;
  for (const xla::XlaOp operand : operands) {
    xla::Shape operand_shape = XlaHelpers::ShapeOfXlaOp(operand);
    operand_shapes.push_back(operand_shape);
    result_shape = XlaHelpers::GetPromotedShape(result_shape, operand_shape);
  }
  std::vector<xla::XlaOp> result;
  for (size_t i = 0; i < operands.size(); ++i) {
    result.push_back(XlaHelpers::ImplicitBroadcast(
        operands[i], operand_shapes[i], result_shape));
  }
  return result;
}

}  // namespace torch_xla
