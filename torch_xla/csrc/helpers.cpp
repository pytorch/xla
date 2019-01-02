#include "helpers.h"

#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/tf_logging.h"

namespace torch {
namespace jit {
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
    const std::vector<int64_t>& padding) {
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

std::pair<xla::XlaOp, xla::XlaOp> XlaHelpers::PromoteShapes(
    const xla::XlaOp& op1, const xla::XlaOp& op2) {
  xla::Shape shape1 = ShapeOfXlaOp(op1);
  xla::Shape shape2 = ShapeOfXlaOp(op2);
  if (xla::ShapeUtil::CompatibleIgnoringFpPrecision(shape1, shape2)) {
    // Fast path shortcut if the shapes already matches in dimensions.
    return std::pair<xla::XlaOp, xla::XlaOp>(op1, op2);
  }
  XLA_CHECK(xla::ShapeUtil::SameElementTypeIgnoringFpPrecision(shape1, shape2))
      << shape1 << " and " << shape2;
  xla::int64 rank1 = xla::ShapeUtil::Rank(shape1);
  xla::int64 rank2 = xla::ShapeUtil::Rank(shape2);
  if (rank1 == rank2) {
    std::vector<xla::int64> dimensions;
    for (xla::int64 i = 0; i < rank1; ++i) {
      xla::int64 dim1 = xla::ShapeUtil::GetDimension(shape1, i);
      xla::int64 dim2 = xla::ShapeUtil::GetDimension(shape2, i);
      XLA_CHECK(dim1 == dim2 || dim1 == 1 || dim2 == 1)
          << shape1 << " and " << shape2;
      dimensions.push_back(std::max<xla::int64>(dim1, dim2));
    }
    xla::XlaOp new_op1 = op1;
    xla::XlaOp new_op2 = op2;
    xla::Shape new_shape1 =
        xla::ShapeUtil::MakeShape(shape1.element_type(), dimensions);
    xla::Shape new_shape2 =
        xla::ShapeUtil::MakeShape(shape2.element_type(), dimensions);
    std::vector<xla::int64> broadcast_sizes(rank1);
    std::iota(broadcast_sizes.begin(), broadcast_sizes.end(), 0);
    if (!xla::ShapeUtil::Compatible(shape1, new_shape1)) {
      new_op1 = xla::Broadcast(op1, broadcast_sizes);
    }
    if (!xla::ShapeUtil::Compatible(shape2, new_shape2)) {
      new_op2 = xla::Broadcast(op2, broadcast_sizes);
    }
    return std::pair<xla::XlaOp, xla::XlaOp>(new_op1, new_op2);
  }
  // The inner dimensions must match, as broadcast adds outer dimensions.
  xla::int64 min_rank = std::min<xla::int64>(rank1, rank2);
  for (xla::int64 i = 0; i < min_rank; ++i) {
    xla::int64 dim1 = xla::ShapeUtil::GetDimension(shape1, rank1 - 1 - i);
    xla::int64 dim2 = xla::ShapeUtil::GetDimension(shape2, rank2 - 1 - i);
    XLA_CHECK(dim1 == dim2) << shape1 << " and " << shape2;
  }
  if (rank1 < rank2) {
    std::vector<xla::int64> broadcast_sizes;
    for (xla::int64 i = 0; i < rank2 - rank1; ++i) {
      broadcast_sizes.push_back(xla::ShapeUtil::GetDimension(shape2, i));
    }
    return std::pair<xla::XlaOp, xla::XlaOp>(
        xla::Broadcast(op1, broadcast_sizes), op2);
  } else {
    std::vector<xla::int64> broadcast_sizes;
    for (xla::int64 i = 0; i < rank1 - rank2; ++i) {
      broadcast_sizes.push_back(xla::ShapeUtil::GetDimension(shape1, i));
    }
    return std::pair<xla::XlaOp, xla::XlaOp>(
        op1, xla::Broadcast(op2, broadcast_sizes));
  }
}

std::pair<xla::XlaOp, xla::XlaOp> XlaHelpers::Promote(const xla::XlaOp& op1,
                                                      const xla::XlaOp& op2) {
  std::pair<xla::XlaOp, xla::XlaOp> vops = PromoteValues(op1, op2);
  return PromoteShapes(vops.first, vops.second);
}

bool XlaHelpers::UseBF16() {
  static bool use_fp16 = ShouldUseBF16();
  return use_fp16;
}

}  // namespace jit
}  // namespace torch
