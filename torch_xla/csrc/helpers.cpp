#include "helpers.h"

#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/tf_logging.h"

namespace torch_xla {
namespace {

bool ShouldUseBF16() {
  int use_fp16 = xla::sys_util::GetEnvInt("XLA_USE_BF16", 0);
  if (use_fp16 != 0) {
    TF_LOG(INFO) << "Using BF16 data type for floating point values";
  }
  return use_fp16 != 0;
}

}  // namespace

xla::PrecisionConfig XlaHelpers::BuildPrecisionConfig(
    const xla::PrecisionConfig::Precision conv_precision) {
  xla::PrecisionConfig precision_config;
  // Dot and convolution take two operators.
  precision_config.mutable_operand_precision()->Resize(
      /*new_size=*/2, conv_precision);
  return precision_config;
}

std::vector<xla::int64> XlaHelpers::I64List(const at::IntList& input) {
  std::vector<xla::int64> output(input.size());
  std::copy(input.begin(), input.end(), output.begin());
  return output;
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

xla::PaddingConfig XlaHelpers::MakeXlaPaddingConfig(
    tensorflow::gtl::ArraySlice<const xla::int64> padding) {
  XLA_CHECK_EQ(padding.size(), 2) << "Only 2D padding supported";
  xla::PaddingConfig padding_config;
  for (int i = 0; i < 2; ++i) {
    padding_config.add_dimensions();
  }
  for (int i = 0; i < 2; ++i) {
    auto* dims = padding_config.add_dimensions();
    dims->set_edge_padding_low(padding[i]);
    dims->set_edge_padding_high(padding[i]);
  }
  return padding_config;
}

xla::XlaComputation XlaHelpers::CreateAddComputation(xla::PrimitiveType type) {
  xla::XlaBuilder reduction_builder("xla_add_computation");
  const auto x = xla::Parameter(&reduction_builder, 0,
                                xla::ShapeUtil::MakeShape(type, {}), "x");
  const auto y = xla::Parameter(&reduction_builder, 1,
                                xla::ShapeUtil::MakeShape(type, {}), "y");
  Add(x, y);
  return reduction_builder.Build().ConsumeValueOrDie();
}

xla::PrimitiveType XlaHelpers::MakeXlaPrimitiveType(
    const at::ScalarType scalar_type) {
  // When PyTorch will support native BF16 type, the global configuration can be
  // replaced (or augmented) with the proper mapping.
  switch (scalar_type) {
    case at::ScalarType::Float:
      return UseBF16() ? xla::PrimitiveType::BF16 : xla::PrimitiveType::F32;
    case at::ScalarType::Long:
      return xla::PrimitiveType::S64;
    default:
      XLA_ERROR() << "Type not supported: " << scalar_type;
      return xla::PrimitiveType::PRIMITIVE_TYPE_INVALID;
  }
}

std::vector<xla::int64> XlaHelpers::ShapeSizes(const xla::Shape& shape) {
  std::vector<xla::int64> shape_sizes(shape.dimensions().begin(),
                                      shape.dimensions().end());
  return shape_sizes;
}

xla::Shape XlaHelpers::ShapeOfXlaOp(const xla::XlaOp& op) {
  return op.builder()->GetShape(op).ConsumeValueOrDie();
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
          op1, xla::ConvertElementType(op2, type1));
    }
    return std::pair<xla::XlaOp, xla::XlaOp>(
        xla::ConvertElementType(op1, type2), op2);
  }
  if (xla::primitive_util::IsFloatingPointType(type2) || size2 >= size1) {
    return std::pair<xla::XlaOp, xla::XlaOp>(
        xla::ConvertElementType(op1, type2), op2);
  }
  return std::pair<xla::XlaOp, xla::XlaOp>(op1,
                                           xla::ConvertElementType(op2, type1));
}

std::pair<xla::XlaOp, xla::XlaOp> XlaHelpers::PromoteSecondValue(
    const xla::XlaOp& op1, const xla::XlaOp& op2) {
  xla::PrimitiveType type1 = TypeOfXlaOp(op1);
  xla::PrimitiveType type2 = TypeOfXlaOp(op2);
  return type1 == type2 ? std::pair<xla::XlaOp, xla::XlaOp>(op1, op2)
                        : std::pair<xla::XlaOp, xla::XlaOp>(
                              op1, xla::ConvertElementType(op2, type1));
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

bool XlaHelpers::UseBF16() {
  static bool use_fp16 = ShouldUseBF16();
  return use_fp16;
}

}  // namespace torch_xla
