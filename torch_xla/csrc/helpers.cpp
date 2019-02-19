#include "torch_xla/csrc/helpers.h"

#include <limits>

#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/tf_logging.h"
#include "torch_xla/csrc/convert_ops.h"

namespace torch_xla {

xla::PrecisionConfig::Precision XlaHelpers::s_mat_mul_precision =
    xla::PrecisionConfig::DEFAULT;

xla::PrecisionConfig XlaHelpers::BuildPrecisionConfig(
    const xla::PrecisionConfig::Precision conv_precision) {
  xla::PrecisionConfig precision_config;
  // Dot and convolution take two operators.
  precision_config.mutable_operand_precision()->Resize(
      /*new_size=*/2, conv_precision);
  return precision_config;
}

xla::XlaOp XlaHelpers::CreateReturnValue(
    xla::XlaBuilder* builder, const std::vector<xla::XlaOp>& outputs) {
  if (outputs.size() > 1) {
    return xla::Tuple(builder, outputs);
  } else if (!outputs.empty()) {
    return xla::GetTupleElement(xla::Tuple(builder, {outputs[0]}), 0);
  } else {
    return xla::Tuple(builder, {});
  }
}

std::vector<xla::int64> XlaHelpers::DropDimensions(
    tensorflow::gtl::ArraySlice<const xla::int64> sizes,
    tensorflow::gtl::ArraySlice<const xla::int64> drop_dims) {
  std::vector<xla::int64> new_dims;
  size_t drop_index = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (drop_index < drop_dims.size() && i == drop_dims[drop_index]) {
      ++drop_index;
    } else {
      new_dims.push_back(sizes[i]);
    }
  }
  XLA_CHECK_EQ(drop_index, drop_dims.size());
  return new_dims;
}

XlaHelpers::MinMax XlaHelpers::MinMaxValues(xla::PrimitiveType type) {
  switch (type) {
    case xla::PrimitiveType::S8:
      return {std::numeric_limits<xla::int8>::min(),
              std::numeric_limits<xla::int8>::max()};
    case xla::PrimitiveType::U8:
      return {std::numeric_limits<xla::uint8>::min(),
              std::numeric_limits<xla::uint8>::max()};
    case xla::PrimitiveType::S16:
      return {std::numeric_limits<xla::int16>::min(),
              std::numeric_limits<xla::int16>::max()};
    case xla::PrimitiveType::U16:
      return {std::numeric_limits<xla::uint16>::min(),
              std::numeric_limits<xla::uint16>::max()};
    case xla::PrimitiveType::S32:
      return {static_cast<int64_t>(std::numeric_limits<xla::int32>::min()),
              static_cast<int64_t>(std::numeric_limits<xla::int32>::max())};
    case xla::PrimitiveType::U32:
      return {static_cast<int64_t>(std::numeric_limits<xla::uint32>::min()),
              static_cast<int64_t>(std::numeric_limits<xla::uint32>::max())};
    case xla::PrimitiveType::S64:
      return {static_cast<int64_t>(std::numeric_limits<xla::int64>::min()),
              static_cast<int64_t>(std::numeric_limits<xla::int64>::max())};
    case xla::PrimitiveType::U64:
      return {static_cast<int64_t>(std::numeric_limits<xla::uint64>::min()),
              static_cast<int64_t>(std::numeric_limits<xla::uint64>::max())};
    case xla::PrimitiveType::BF16:
    case xla::PrimitiveType::F32:
      return {std::numeric_limits<float>::min(),
              std::numeric_limits<float>::max()};
    case xla::PrimitiveType::F64:
      return {std::numeric_limits<double>::min(),
              std::numeric_limits<double>::max()};
    case xla::PrimitiveType::PRED:
      return {0, 1};
    default:
      XLA_ERROR() << "Unsupported XLA type " << type;
  }
}

xla::PaddingConfig XlaHelpers::MakeXlaPaddingConfig(
    tensorflow::gtl::ArraySlice<const xla::int64> padding) {
  XLA_CHECK_EQ(padding.size(), 2) << "Only 2D padding supported";
  xla::PaddingConfig padding_config;
  for (int i = 0; i < 2; ++i) {
    padding_config.add_dimensions();
  }
  for (int i = 0; i < 2; ++i) {
    xla::PaddingConfig::PaddingConfigDimension* dims =
        padding_config.add_dimensions();
    dims->set_edge_padding_low(padding[i]);
    dims->set_edge_padding_high(padding[i]);
  }
  return padding_config;
}

xla::XlaComputation XlaHelpers::CreateAddComputation(xla::PrimitiveType type) {
  xla::XlaBuilder reduction_builder("xla_add_computation");
  xla::XlaOp x = xla::Parameter(&reduction_builder, 0,
                                xla::ShapeUtil::MakeShape(type, {}), "x");
  xla::XlaOp y = xla::Parameter(&reduction_builder, 1,
                                xla::ShapeUtil::MakeShape(type, {}), "y");
  xla::Add(x, y);
  return ConsumeValue(reduction_builder.Build());
}

xla::XlaComputation XlaHelpers::CreateMaxComputation(xla::PrimitiveType type) {
  xla::XlaBuilder builder("MaxComputation");
  xla::XlaOp x =
      xla::Parameter(&builder, 0, xla::ShapeUtil::MakeShape(type, {}), "x");
  xla::XlaOp y =
      xla::Parameter(&builder, 1, xla::ShapeUtil::MakeShape(type, {}), "y");
  xla::Max(x, y);
  return ConsumeValue(builder.Build());
}

xla::Shape XlaHelpers::ShapeOfXlaOp(const xla::XlaOp& op) {
  return ConsumeValue(op.builder()->GetShape(op));
}

std::vector<xla::int64> XlaHelpers::SizesOfXlaOp(const xla::XlaOp& op) {
  return ShapeSizes(ShapeOfXlaOp(op));
}

xla::PrimitiveType XlaHelpers::TypeOfXlaOp(const xla::XlaOp& op) {
  return ShapeOfXlaOp(op).element_type();
}

std::pair<xla::XlaOp, xla::XlaOp> XlaHelpers::PromoteValues(
    const xla::XlaOp& op1, const xla::XlaOp& op2) {
  xla::PrimitiveType type1 = TypeOfXlaOp(op1);
  xla::PrimitiveType type2 = TypeOfXlaOp(op2);
  if (type1 == type2) {
    return std::pair<xla::XlaOp, xla::XlaOp>(op1, op2);
  }
  xla::int64 size1 = xla::ShapeUtil::ByteSizeOfPrimitiveType(type1);
  xla::int64 size2 = xla::ShapeUtil::ByteSizeOfPrimitiveType(type2);
  if (xla::primitive_util::IsFloatingPointType(type1)) {
    if (!xla::primitive_util::IsFloatingPointType(type2) || size1 >= size2) {
      return std::pair<xla::XlaOp, xla::XlaOp>(
          op1, ConvertTo(op2, type2, type1, /*device=*/nullptr));
    }
    return std::pair<xla::XlaOp, xla::XlaOp>(
        ConvertTo(op1, type1, type2, /*device=*/nullptr), op2);
  }
  if (xla::primitive_util::IsFloatingPointType(type2) || size2 >= size1) {
    return std::pair<xla::XlaOp, xla::XlaOp>(
        ConvertTo(op1, type1, type2, /*device=*/nullptr), op2);
  }
  return std::pair<xla::XlaOp, xla::XlaOp>(
      op1, ConvertTo(op2, type2, type1, /*device=*/nullptr));
}

std::pair<xla::XlaOp, xla::XlaOp> XlaHelpers::PromoteSecondValue(
    const xla::XlaOp& op1, const xla::XlaOp& op2) {
  xla::PrimitiveType type1 = TypeOfXlaOp(op1);
  xla::PrimitiveType type2 = TypeOfXlaOp(op2);
  return type1 == type2
             ? std::pair<xla::XlaOp, xla::XlaOp>(op1, op2)
             : std::pair<xla::XlaOp, xla::XlaOp>(
                   op1, ConvertTo(op2, type2, type1, /*device=*/nullptr));
}

xla::Shape XlaHelpers::GetPromotedShape(const xla::Shape& shape1,
                                        const xla::Shape& shape2) {
  const auto& shape1_dims = shape1.dimensions();
  const auto& shape2_dims = shape2.dimensions();
  std::vector<xla::int64> dimensions;
  // If the rank of a shape is bigger than then other, fill up the first
  // dimensions with the ones of the bigger.
  // Example:
  //   shape1 = [9, 7, 6, 5, 2]
  //   shape2 =       [6, 1, 2]
  // Insert [9, 7] into the dimensions vector.
  if (shape1_dims.size() > shape2_dims.size()) {
    dimensions.insert(
        dimensions.end(), shape1_dims.begin(),
        shape1_dims.begin() + (shape1_dims.size() - shape2_dims.size()));
  } else if (shape2_dims.size() > shape1_dims.size()) {
    dimensions.insert(
        dimensions.end(), shape2_dims.begin(),
        shape2_dims.begin() + (shape2_dims.size() - shape1_dims.size()));
  }
  // For the common dimensions, they must match, or one of them be 1.
  size_t min_size = std::min(shape1_dims.size(), shape2_dims.size());
  for (xla::int64 i = 0; i < min_size; ++i) {
    xla::int64 dim1 = shape1_dims[shape1_dims.size() - min_size + i];
    xla::int64 dim2 = shape2_dims[shape2_dims.size() - min_size + i];
    XLA_CHECK(dim1 == dim2 || dim1 == 1 || dim2 == 1)
        << shape1 << " and " << shape2;
    dimensions.push_back(std::max<xla::int64>(dim1, dim2));
  }
  return xla::ShapeUtil::MakeShape(shape1.element_type(), dimensions);
}

std::pair<xla::XlaOp, xla::XlaOp> XlaHelpers::PromoteShapes(
    const xla::XlaOp& op1, const xla::XlaOp& op2) {
  xla::Shape shape1 = ShapeOfXlaOp(op1);
  xla::Shape shape2 = ShapeOfXlaOp(op2);
  if (xla::ShapeUtil::Compatible(shape1, shape2)) {
    // Fast path shortcut if the shapes already matches in dimensions.
    return std::pair<xla::XlaOp, xla::XlaOp>(op1, op2);
  }
  XLA_CHECK(xla::ShapeUtil::SameElementType(shape1, shape2))
      << shape1 << " and " << shape2;

  xla::Shape shape = GetPromotedShape(shape1, shape2);
  return std::pair<xla::XlaOp, xla::XlaOp>(
      ImplicitBroadcast(op1, shape1, shape),
      ImplicitBroadcast(op2, shape2, shape));
}

std::pair<xla::XlaOp, xla::XlaOp> XlaHelpers::Promote(const xla::XlaOp& op1,
                                                      const xla::XlaOp& op2) {
  std::pair<xla::XlaOp, xla::XlaOp> vops = PromoteValues(op1, op2);
  return PromoteShapes(vops.first, vops.second);
}

std::pair<xla::XlaOp, xla::XlaOp> XlaHelpers::PromoteSecond(
    const xla::XlaOp& op1, const xla::XlaOp& op2) {
  std::pair<xla::XlaOp, xla::XlaOp> vops = PromoteSecondValue(op1, op2);
  return PromoteShapes(vops.first, vops.second);
}

xla::XlaOp XlaHelpers::ImplicitBroadcast(const xla::XlaOp& op,
                                         const xla::Shape& op_shape,
                                         const xla::Shape& shape) {
  const auto& op_shape_dims = op_shape.dimensions();
  const auto& shape_dims = shape.dimensions();
  XLA_CHECK_GE(shape_dims.size(), op_shape_dims.size())
      << shape << " vs " << op_shape;
  xla::int64 size_delta = shape_dims.size() - op_shape_dims.size();
  xla::XlaOp new_op = op;
  if (!std::equal(op_shape_dims.begin(), op_shape_dims.end(),
                  shape_dims.begin() + size_delta)) {
    // If the base N dimensions do not match, broadcast the original op.
    // Example:
    //   op_shape =       [3, 1, 5]
    //   shape    = [6, 8, 3, 4, 5]
    // After this operation we will have:
    //   op_shape =       [3, 4, 5]
    std::vector<xla::int64> common_shape_dims(shape_dims.begin() + size_delta,
                                              shape_dims.end());
    std::vector<xla::int64> broadcast_dimensions(op_shape_dims.size());
    std::iota(broadcast_dimensions.begin(), broadcast_dimensions.end(), 0);
    new_op =
        xla::BroadcastInDim(new_op, common_shape_dims, broadcast_dimensions);
  }
  if (size_delta > 0) {
    // Add the major dimensions if necessary:
    // Example:
    //   op_shape =       [3, 4, 5]
    //   shape    = [6, 8, 3, 4, 5]
    // After this operation we will have (added [6, 8]):
    //   op_shape = [6, 8, 3, 4, 5]
    std::vector<xla::int64> broadcast_sizes(shape_dims.begin(),
                                            shape_dims.begin() + size_delta);
    new_op = xla::Broadcast(new_op, broadcast_sizes);
  }
  return new_op;
}

xla::XlaOp XlaHelpers::PromotedBinaryOp(
    const xla::XlaOp& op1, const xla::XlaOp& op2,
    const std::function<xla::XlaOp(const xla::XlaOp&, const xla::XlaOp&)>&
        bin_op) {
  std::pair<xla::XlaOp, xla::XlaOp> vops = PromoteSecond(op1, op2);
  return bin_op(vops.first, vops.second);
}

}  // namespace torch_xla
