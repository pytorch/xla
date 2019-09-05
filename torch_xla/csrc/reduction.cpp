#include "torch_xla/csrc/reduction.h"

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace {

struct ReductionInfo {
  std::vector<xla::int64> new_dimensions;
  xla::int64 element_count = 1;
};

ReductionInfo GetReductionInfo(
    const xla::Shape& shape,
    tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
    bool keep_reduced_dimensions) {
  ReductionInfo rinfo;
  size_t idim = 0;
  for (xla::int64 i = 0; i < shape.rank(); ++i) {
    if (idim < dimensions.size() && dimensions[idim] == i) {
      rinfo.element_count *= shape.dimensions(i);
      ++idim;
      if (keep_reduced_dimensions) {
        rinfo.new_dimensions.push_back(1);
      }
    } else if (keep_reduced_dimensions) {
      rinfo.new_dimensions.push_back(shape.dimensions(i));
    }
  }
  return rinfo;
}

xla::XlaComputation CreateAllComputation(xla::PrimitiveType type) {
  xla::XlaBuilder builder("AllComputation");
  xla::XlaOp x =
      xla::Parameter(&builder, 0, xla::ShapeUtil::MakeShape(type, {}), "x");
  xla::XlaOp y =
      xla::Parameter(&builder, 1, xla::ShapeUtil::MakeShape(type, {}), "y");
  xla::XlaOp zero =
      xla::ConstantLiteral(&builder, xla::LiteralUtil::Zero(type));
  xla::XlaOp one = xla::ConstantLiteral(&builder, xla::LiteralUtil::One(type));
  xla::Select(xla::And(xla::Ne(x, zero), xla::Ne(y, zero)), one, zero);
  return ConsumeValue(builder.Build());
}

xla::XlaComputation CreateAnyComputation(xla::PrimitiveType type) {
  xla::XlaBuilder builder("AnyComputation");
  xla::XlaOp x =
      xla::Parameter(&builder, 0, xla::ShapeUtil::MakeShape(type, {}), "x");
  xla::XlaOp y =
      xla::Parameter(&builder, 1, xla::ShapeUtil::MakeShape(type, {}), "y");
  xla::XlaOp zero =
      xla::ConstantLiteral(&builder, xla::LiteralUtil::Zero(type));
  xla::XlaOp one = xla::ConstantLiteral(&builder, xla::LiteralUtil::One(type));
  xla::Select(xla::Or(xla::Ne(x, zero), xla::Ne(y, zero)), one, zero);
  return ConsumeValue(builder.Build());
}

xla::XlaOp CreateSummation(
    const xla::XlaOp& input,
    tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
    bool keep_reduced_dimensions, bool scale) {
  xla::Shape shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp init_value =
      XlaHelpers::ScalarValue<float>(0, shape.element_type(), input.builder());
  ReductionInfo rinfo =
      GetReductionInfo(shape, dimensions, keep_reduced_dimensions);
  xla::XlaOp result = xla::Reduce(
      input, init_value, XlaHelpers::CreateAddComputation(shape.element_type()),
      dimensions);
  if (scale) {
    xla::XlaOp scale = XlaHelpers::ScalarValue<float>(
        1.0f / static_cast<float>(rinfo.element_count), shape.element_type(),
        input.builder());
    result = xla::Mul(result, scale);
  }
  if (keep_reduced_dimensions) {
    result = xla::Reshape(result, rinfo.new_dimensions);
  }
  return result;
}

xla::XlaOp CreateProduct(
    const xla::XlaOp& input,
    tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
    bool keep_reduced_dimensions) {
  xla::Shape shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp init_value =
      XlaHelpers::ScalarValue<float>(1, shape.element_type(), input.builder());
  ReductionInfo rinfo =
      GetReductionInfo(shape, dimensions, keep_reduced_dimensions);
  xla::XlaOp result = xla::Reduce(
      input, init_value, XlaHelpers::CreateMulComputation(shape.element_type()),
      dimensions);
  if (keep_reduced_dimensions) {
    result = xla::Reshape(result, rinfo.new_dimensions);
  }
  return result;
}

}  // namespace

xla::XlaOp BuildCumulativeComputation(const xla::XlaOp& input, xla::int64 dim,
                                      const xla::XlaComputation& reducer,
                                      const xla::XlaOp& init) {
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  std::vector<xla::int64> window_strides(input_shape.rank(), 1);
  std::vector<xla::int64> window_dims(input_shape.rank(), 1);
  window_dims[dim] = input_shape.dimensions(dim);
  std::vector<std::pair<xla::int64, xla::int64>> padding(input_shape.rank());
  padding[dim].first = input_shape.dimensions(dim) - 1;
  return xla::ReduceWindowWithGeneralPadding(
      input, init, reducer, window_dims, window_strides,
      /*base_dilations=*/{}, /*window_dilations=*/{}, padding);
}

xla::XlaOp BuildMean(const xla::XlaOp& input,
                     tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
                     bool keep_reduced_dimensions) {
  return CreateSummation(input, dimensions, keep_reduced_dimensions,
                         /*scale=*/true);
}

xla::XlaOp BuildSum(const xla::XlaOp& input,
                    tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
                    bool keep_reduced_dimensions) {
  return CreateSummation(input, dimensions, keep_reduced_dimensions,
                         /*scale=*/false);
}

xla::XlaOp BuildProd(const xla::XlaOp& input,
                     tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
                     bool keep_reduced_dimensions) {
  return CreateProduct(input, dimensions, keep_reduced_dimensions);
}

xla::XlaOp BuildMaxInDim(const xla::XlaOp& input, xla::int64 dim,
                         bool keep_reduced_dimensions) {
  xla::Shape shape = XlaHelpers::ShapeOfXlaOp(input);
  XlaHelpers::MinMax min_max = XlaHelpers::MinMaxValues(shape.element_type());
  xla::XlaOp init_value = XlaHelpers::ScalarValue(
      min_max.min, shape.element_type(), input.builder());
  ReductionInfo rinfo = GetReductionInfo(shape, {dim}, keep_reduced_dimensions);
  XLA_CHECK_GT(rinfo.element_count, 0);
  xla::XlaOp result = xla::Reduce(
      input, init_value, XlaHelpers::CreateMaxComputation(shape.element_type()),
      {dim});
  if (keep_reduced_dimensions) {
    result = xla::Reshape(result, rinfo.new_dimensions);
  }
  return result;
}

xla::XlaOp BuildMinInDim(const xla::XlaOp& input, xla::int64 dim,
                         bool keep_reduced_dimensions) {
  xla::Shape shape = XlaHelpers::ShapeOfXlaOp(input);
  XlaHelpers::MinMax min_max = XlaHelpers::MinMaxValues(shape.element_type());
  xla::XlaOp init_value = XlaHelpers::ScalarValue(
      min_max.max, shape.element_type(), input.builder());
  ReductionInfo rinfo = GetReductionInfo(shape, {dim}, keep_reduced_dimensions);
  XLA_CHECK_GT(rinfo.element_count, 0);
  xla::XlaOp result = xla::Reduce(
      input, init_value, XlaHelpers::CreateMinComputation(shape.element_type()),
      {dim});
  if (keep_reduced_dimensions) {
    result = xla::Reshape(result, rinfo.new_dimensions);
  }
  return result;
}

xla::XlaOp BuildArgMax(const xla::XlaOp& input, xla::int64 dim, bool keepdim) {
  xla::Shape shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp operand = input;
  if (dim < 0) {
    dim = 0;
    operand = xla::Reshape(operand, {xla::ShapeUtil::ElementsIn(shape)});
    shape = XlaHelpers::ShapeOfXlaOp(operand);
  }
  xla::XlaOp result = xla::ArgMaxTwoPass(
      operand,
      GetDevicePrimitiveType(xla::PrimitiveType::S64, /*device=*/nullptr), dim);
  if (keepdim) {
    auto dimensions = xla::util::ToVector<xla::int64>(shape.dimensions());
    dimensions[dim] = 1;
    result = xla::Reshape(result, dimensions);
  }
  return result;
}

xla::XlaOp BuildArgMin(const xla::XlaOp& input, xla::int64 dim, bool keepdim) {
  xla::Shape shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp operand = input;
  if (dim < 0) {
    dim = 0;
    operand = xla::Reshape(operand, {xla::ShapeUtil::ElementsIn(shape)});
    shape = XlaHelpers::ShapeOfXlaOp(operand);
  }
  xla::XlaOp result = xla::ArgMinTwoPass(
      operand,
      GetDevicePrimitiveType(xla::PrimitiveType::S64, /*device=*/nullptr), dim);
  if (keepdim) {
    auto dimensions = xla::util::ToVector<xla::int64>(shape.dimensions());
    dimensions[dim] = 1;
    result = xla::Reshape(result, dimensions);
  }
  return result;
}

xla::XlaOp BuildAll(const xla::XlaOp& input,
                    tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
                    bool keep_reduced_dimensions) {
  xla::Shape shape = XlaHelpers::ShapeOfXlaOp(input);
  ReductionInfo rinfo =
      GetReductionInfo(shape, dimensions, keep_reduced_dimensions);
  xla::XlaOp init_value = xla::ConstantLiteral(
      input.builder(), xla::LiteralUtil::One(shape.element_type()));
  xla::XlaOp result =
      xla::Reduce(input, init_value, CreateAllComputation(shape.element_type()),
                  dimensions);
  if (keep_reduced_dimensions) {
    result = xla::Reshape(result, rinfo.new_dimensions);
  }
  return result;
}

xla::XlaOp BuildAny(const xla::XlaOp& input,
                    tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
                    bool keep_reduced_dimensions) {
  xla::Shape shape = XlaHelpers::ShapeOfXlaOp(input);
  ReductionInfo rinfo =
      GetReductionInfo(shape, dimensions, keep_reduced_dimensions);
  xla::XlaOp init_value = xla::ConstantLiteral(
      input.builder(), xla::LiteralUtil::Zero(shape.element_type()));
  xla::XlaOp result =
      xla::Reduce(input, init_value, CreateAnyComputation(shape.element_type()),
                  dimensions);
  if (keep_reduced_dimensions) {
    result = xla::Reshape(result, rinfo.new_dimensions);
  }
  return result;
}

}  // namespace torch_xla
