#include "torch_xla/csrc/xla_op_builder.h"

#include <map>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/client/lib/logdet.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/computation.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace op_builder {
namespace {

typedef xla::XlaOp (*XlaOpFunction)(const BuilderPtr&,
                                    const std::vector<OpPtr>&, py::dict);
using XlaOpFunctionMap = std::map<std::string, XlaOpFunction>;

#define XLA_PBSET(msg, name, type, args) \
  msg.set_##name(args[#name].cast<type>())

#define XLA_PBSET_REP(msg, name, type, args)          \
  for (auto& value : args[#name].cast<py::tuple>()) { \
    msg.add_##name(value.cast<type>());               \
  }

#define XLA_UNARY_OP(name)                                               \
  xla::XlaOp name(const BuilderPtr&, const std::vector<OpPtr>& operands, \
                  py::dict /* args */) {                                 \
    return xla::name(operands.at(0)->op);                                \
  }

#define XLA_BINARY_OP(name)                                              \
  xla::XlaOp name(const BuilderPtr&, const std::vector<OpPtr>& operands, \
                  py::dict /* args */) {                                 \
    return xla::name(operands.at(0)->op, operands.at(1)->op);            \
  }

XLA_UNARY_OP(Abs);
XLA_UNARY_OP(Acos);
XLA_UNARY_OP(Asin);
XLA_UNARY_OP(Atan);
XLA_UNARY_OP(Ceil);
XLA_UNARY_OP(Cos);
XLA_UNARY_OP(Cosh);
XLA_UNARY_OP(Erf);
XLA_UNARY_OP(Erfc);
XLA_UNARY_OP(ErfInv);
XLA_UNARY_OP(Exp);
XLA_UNARY_OP(Expm1);
XLA_UNARY_OP(Floor);
XLA_UNARY_OP(Log);
XLA_UNARY_OP(Log1p);
XLA_UNARY_OP(Neg);
XLA_UNARY_OP(Not);
XLA_UNARY_OP(Sqrt);
XLA_UNARY_OP(Rsqrt);
XLA_UNARY_OP(Sin);
XLA_UNARY_OP(Sinh);
XLA_UNARY_OP(Tan);
XLA_UNARY_OP(Tanh);

XLA_BINARY_OP(Add);
XLA_BINARY_OP(And);
XLA_BINARY_OP(Atan2);
XLA_BINARY_OP(Div);
XLA_BINARY_OP(Eq);
XLA_BINARY_OP(Ge);
XLA_BINARY_OP(Gt);
XLA_BINARY_OP(Le);
XLA_BINARY_OP(Lt);
XLA_BINARY_OP(Max);
XLA_BINARY_OP(Min);
XLA_BINARY_OP(Mul);
XLA_BINARY_OP(Ne);
XLA_BINARY_OP(Or);
XLA_BINARY_OP(Pow);
XLA_BINARY_OP(Rem);
XLA_BINARY_OP(Sub);
XLA_BINARY_OP(Xor);

template <typename T>
std::vector<T> GetTupleVector(py::tuple tuple) {
  std::vector<T> values;
  values.reserve(tuple.size());
  for (auto& v : tuple) {
    values.push_back(v.cast<T>());
  }
  return values;
}

template <typename T>
absl::optional<T> ArgOptional(py::dict args, const char* name) {
  if (!args.contains(name)) {
    return absl::nullopt;
  }
  auto value = args[name];
  if (value.is_none()) {
    return absl::nullopt;
  }
  return value.cast<T>();
}

template <typename T>
T ArgOrDefault(py::dict args, const char* name, T defval) {
  absl::optional<T> value = ArgOptional<T>(args, name);
  return value.value_or(defval);
}

std::vector<xla::XlaOp> ExtractXlaOps(const std::vector<OpPtr>& operands) {
  std::vector<xla::XlaOp> ops;
  for (auto& operand : operands) {
    ops.push_back(operand->op);
  }
  return ops;
}

std::vector<xla::XlaOp> GetOpVector(py::tuple tuple) {
  std::vector<xla::XlaOp> ops;
  for (auto& op : tuple) {
    ops.push_back(op.cast<OpPtr>()->op);
  }
  return ops;
}

xla::XlaOp Reshape(const BuilderPtr& builder,
                   const std::vector<OpPtr>& operands, py::dict args) {
  std::vector<xla::int64> sizes = GetTupleVector<xla::int64>(args["sizes"]);
  absl::optional<py::tuple> arg_dimensions =
      ArgOptional<py::tuple>(args, "dimensions");
  if (arg_dimensions) {
    std::vector<xla::int64> dimensions =
        GetTupleVector<xla::int64>(*arg_dimensions);
    return xla::Reshape(operands.at(0)->op, dimensions, sizes);
  }
  xla::int64 inferred_dimension =
      ArgOrDefault<xla::int64>(args, "inferred_dimension", -1);
  if (inferred_dimension >= 0) {
    return xla::ReshapeWithInferredDimension(operands.at(0)->op, sizes,
                                             inferred_dimension);
  }
  return xla::Reshape(operands.at(0)->op, sizes);
}

xla::XlaOp DynamicReshape(const BuilderPtr& builder,
                          const std::vector<OpPtr>& operands, py::dict args) {
  std::vector<xla::int64> sizes = GetTupleVector<xla::int64>(args["sizes"]);
  return XlaHelpers::DynamicReshape(operands.at(0)->op, sizes);
}

xla::XlaOp Broadcast(const BuilderPtr& builder,
                     const std::vector<OpPtr>& operands, py::dict args) {
  std::vector<xla::int64> sizes = GetTupleVector<xla::int64>(args["sizes"]);
  return xla::Broadcast(operands.at(0)->op, sizes);
}

xla::XlaOp BroadcastInDim(const BuilderPtr& builder,
                          const std::vector<OpPtr>& operands, py::dict args) {
  std::vector<xla::int64> sizes = GetTupleVector<xla::int64>(args["sizes"]);
  std::vector<xla::int64> dimensions =
      GetTupleVector<xla::int64>(args["dimensions"]);
  return xla::BroadcastInDim(operands.at(0)->op, sizes, dimensions);
}

xla::XlaOp Tuple(const BuilderPtr& builder, const std::vector<OpPtr>& operands,
                 py::dict args) {
  std::vector<xla::XlaOp> ops = ExtractXlaOps(operands);
  return xla::Tuple(builder.get(), ops);
}

xla::PrecisionConfig DotPrecisonConfig(py::dict args) {
  xla::PrecisionConfig::Precision precision = XlaHelpers::mat_mul_precision();
  absl::optional<std::string> arg_precision_config =
      ArgOptional<std::string>(args, "precision_config");
  if (arg_precision_config) {
    if (*arg_precision_config == "default") {
      precision = xla::PrecisionConfig::DEFAULT;
    } else if (*arg_precision_config == "high") {
      precision = xla::PrecisionConfig::HIGH;
    } else if (*arg_precision_config == "highest") {
      precision = xla::PrecisionConfig::HIGHEST;
    }
  }
  return XlaHelpers::BuildPrecisionConfig(precision);
}

xla::XlaOp Dot(const BuilderPtr& builder, const std::vector<OpPtr>& operands,
               py::dict args) {
  xla::PrecisionConfig precision_config = DotPrecisonConfig(args);
  return xla::Dot(operands.at(0)->op, operands.at(1)->op, &precision_config);
}

xla::XlaOp Constant(const BuilderPtr& builder,
                    const std::vector<OpPtr>& operands, py::dict args) {
  at::Tensor tensor = args["value"].cast<at::Tensor>();
  xla::Literal literal =
      GetTensorLiteral(tensor, /*shape=*/nullptr, /*device=*/nullptr);
  return xla::ConstantLiteral(builder.get(), literal);
}

xla::PaddingConfig ParsePaddingConfig(py::tuple cfg) {
  xla::PaddingConfig pad_config;
  for (auto& dimp : cfg) {
    py::tuple dims = dimp.cast<py::tuple>();
    XLA_CHECK_EQ(dims.size(), 3);
    auto dim = pad_config.add_dimensions();
    dim->set_edge_padding_low(dims[0].cast<xla::int64>());
    dim->set_edge_padding_high(dims[1].cast<xla::int64>());
    dim->set_interior_padding(dims[2].cast<xla::int64>());
  }
  return pad_config;
}

template <typename T>
std::vector<std::pair<T, T>> ParsePairList(py::tuple plist) {
  std::vector<std::pair<T, T>> pairs;
  for (auto& p : plist) {
    py::tuple pt = p.cast<py::tuple>();
    XLA_CHECK_EQ(pt.size(), 2);
    pairs.emplace_back(pt[0].cast<T>(), pt[1].cast<T>());
  }
  return pairs;
}

xla::XlaOp Pad(const BuilderPtr& builder, const std::vector<OpPtr>& operands,
               py::dict args) {
  xla::PaddingConfig pad_config = ParsePaddingConfig(args["config"]);
  return xla::Pad(operands.at(0)->op, operands.at(1)->op, pad_config);
}

xla::XlaOp Transpose(const BuilderPtr& builder,
                     const std::vector<OpPtr>& operands, py::dict args) {
  std::vector<xla::int64> permutation =
      GetTupleVector<xla::int64>(args["permutation"]);
  return xla::Transpose(operands.at(0)->op, permutation);
}

xla::Padding ParseConvPadding(const std::string& padding_str) {
  if (padding_str == "same") {
    return xla::Padding::kSame;
  }
  if (padding_str == "valid") {
    return xla::Padding::kValid;
  }
  XLA_ERROR() << "Invalid padding: " << padding_str;
}

xla::ConvolutionDimensionNumbers ParseConvolutionDimensionNumbers(
    py::dict args) {
  xla::ConvolutionDimensionNumbers dimension_numbers;
  XLA_PBSET(dimension_numbers, input_batch_dimension, xla::int64, args);
  XLA_PBSET(dimension_numbers, input_feature_dimension, xla::int64, args);
  XLA_PBSET(dimension_numbers, kernel_input_feature_dimension, xla::int64,
            args);
  XLA_PBSET(dimension_numbers, kernel_output_feature_dimension, xla::int64,
            args);
  XLA_PBSET(dimension_numbers, output_batch_dimension, xla::int64, args);
  XLA_PBSET(dimension_numbers, output_feature_dimension, xla::int64, args);
  XLA_PBSET_REP(dimension_numbers, input_spatial_dimensions, xla::int64, args);
  XLA_PBSET_REP(dimension_numbers, kernel_spatial_dimensions, xla::int64, args);
  XLA_PBSET_REP(dimension_numbers, output_spatial_dimensions, xla::int64, args);
  return dimension_numbers;
}

xla::XlaOp ConvWithGeneralPadding(const BuilderPtr& builder,
                                  const std::vector<OpPtr>& operands,
                                  py::dict args) {
  std::vector<xla::int64> window_strides =
      GetTupleVector<xla::int64>(args["window_strides"]);
  xla::int64 feature_group_count =
      ArgOrDefault<xla::int64>(args, "feature_group_count", 1);
  xla::int64 batch_group_count =
      ArgOrDefault<xla::int64>(args, "batch_group_count", 1);
  auto padding = ParsePairList<xla::int64>(args["padding"]);
  xla::PrecisionConfig precision_config = DotPrecisonConfig(args);
  return xla::ConvWithGeneralPadding(
      operands.at(0)->op, operands.at(1)->op, window_strides, padding,
      feature_group_count, batch_group_count, &precision_config);
}

xla::XlaOp ConvWithGeneralDimensions(const BuilderPtr& builder,
                                     const std::vector<OpPtr>& operands,
                                     py::dict args) {
  std::vector<xla::int64> window_strides =
      GetTupleVector<xla::int64>(args["window_strides"]);
  xla::int64 feature_group_count =
      ArgOrDefault<xla::int64>(args, "feature_group_count", 1);
  xla::int64 batch_group_count =
      ArgOrDefault<xla::int64>(args, "batch_group_count", 1);
  xla::Padding padding = ParseConvPadding(args["padding"].cast<std::string>());
  xla::ConvolutionDimensionNumbers dimension_numbers =
      ParseConvolutionDimensionNumbers(args);
  xla::PrecisionConfig precision_config = DotPrecisonConfig(args);
  return xla::ConvWithGeneralDimensions(operands.at(0)->op, operands.at(1)->op,
                                        window_strides, padding,
                                        dimension_numbers, feature_group_count,
                                        batch_group_count, &precision_config);
}

xla::XlaOp ConvGeneral(const BuilderPtr& builder,
                       const std::vector<OpPtr>& operands, py::dict args) {
  std::vector<xla::int64> window_strides =
      GetTupleVector<xla::int64>(args["window_strides"]);
  xla::int64 feature_group_count =
      ArgOrDefault<xla::int64>(args, "feature_group_count", 1);
  xla::int64 batch_group_count =
      ArgOrDefault<xla::int64>(args, "batch_group_count", 1);
  auto padding = ParsePairList<xla::int64>(args["padding"]);
  xla::ConvolutionDimensionNumbers dimension_numbers =
      ParseConvolutionDimensionNumbers(args);
  xla::PrecisionConfig precision_config = DotPrecisonConfig(args);
  return xla::ConvGeneral(operands.at(0)->op, operands.at(1)->op,
                          window_strides, padding, dimension_numbers,
                          feature_group_count, batch_group_count,
                          &precision_config);
}

xla::XlaOp ConvGeneralDilated(const BuilderPtr& builder,
                              const std::vector<OpPtr>& operands,
                              py::dict args) {
  std::vector<xla::int64> window_strides =
      GetTupleVector<xla::int64>(args["window_strides"]);
  std::vector<xla::int64> lhs_dilation =
      GetTupleVector<xla::int64>(args["lhs_dilation"]);
  std::vector<xla::int64> rhs_dilation =
      GetTupleVector<xla::int64>(args["rhs_dilation"]);
  xla::int64 feature_group_count =
      ArgOrDefault<xla::int64>(args, "feature_group_count", 1);
  xla::int64 batch_group_count =
      ArgOrDefault<xla::int64>(args, "batch_group_count", 1);
  auto padding = ParsePairList<xla::int64>(args["padding"]);
  xla::ConvolutionDimensionNumbers dimension_numbers =
      ParseConvolutionDimensionNumbers(args);
  xla::PrecisionConfig precision_config = DotPrecisonConfig(args);
  return xla::ConvGeneralDilated(
      operands.at(0)->op, operands.at(1)->op, window_strides, padding,
      lhs_dilation, rhs_dilation, dimension_numbers, feature_group_count,
      batch_group_count, &precision_config);
}

xla::XlaOp Conv(const BuilderPtr& builder, const std::vector<OpPtr>& operands,
                py::dict args) {
  std::vector<xla::int64> window_strides =
      GetTupleVector<xla::int64>(args["window_strides"]);
  xla::int64 feature_group_count =
      ArgOrDefault<xla::int64>(args, "feature_group_count", 1);
  xla::int64 batch_group_count =
      ArgOrDefault<xla::int64>(args, "batch_group_count", 1);
  xla::Padding padding = ParseConvPadding(args["padding"].cast<std::string>());
  xla::PrecisionConfig precision_config = DotPrecisonConfig(args);
  return xla::Conv(operands.at(0)->op, operands.at(1)->op, window_strides,
                   padding, feature_group_count, batch_group_count,
                   &precision_config);
}

xla::XlaOp Slice(const BuilderPtr& builder, const std::vector<OpPtr>& operands,
                 py::dict args) {
  std::vector<xla::int64> start_indices =
      GetTupleVector<xla::int64>(args["start_indices"]);
  std::vector<xla::int64> limit_indices =
      GetTupleVector<xla::int64>(args["limit_indices"]);
  std::vector<xla::int64> strides = GetTupleVector<xla::int64>(args["strides"]);
  return xla::Slice(operands.at(0)->op, start_indices, limit_indices, strides);
}

xla::XlaOp SliceInDim(const BuilderPtr& builder,
                      const std::vector<OpPtr>& operands, py::dict args) {
  xla::int64 start_index = args["start_index"].cast<xla::int64>();
  xla::int64 limit_index = args["limit_index"].cast<xla::int64>();
  xla::int64 dimno = args["dimno"].cast<xla::int64>();
  xla::int64 stride = ArgOrDefault<xla::int64>(args, "stride", 1);
  return xla::SliceInDim(operands.at(0)->op, start_index, limit_index, stride,
                         dimno);
}

xla::XlaOp DynamicSlice(const BuilderPtr& builder,
                        const std::vector<OpPtr>& operands, py::dict args) {
  std::vector<xla::int64> slice_sizes =
      GetTupleVector<xla::int64>(args["slice_sizes"]);
  std::vector<xla::XlaOp> start_indices =
      GetOpVector(args["start_indices"].cast<py::tuple>());
  return xla::DynamicSlice(operands.at(0)->op, start_indices, slice_sizes);
}

xla::XlaOp DynamicUpdateSlice(const BuilderPtr& builder,
                              const std::vector<OpPtr>& operands,
                              py::dict args) {
  std::vector<xla::XlaOp> start_indices =
      GetOpVector(args["start_indices"].cast<py::tuple>());
  return xla::DynamicUpdateSlice(operands.at(0)->op, operands.at(1)->op,
                                 start_indices);
}

xla::XlaOp Reduce(const BuilderPtr& builder, const std::vector<OpPtr>& operands,
                  py::dict args) {
  std::vector<xla::int64> dimensions =
      GetTupleVector<xla::int64>(args["dimensions"]);
  ComputationPtr computation = args["computation"].cast<ComputationPtr>();
  return xla::Reduce(operands.at(0)->op, operands.at(1)->op,
                     computation->computation(), dimensions);
}

xla::XlaOp Call(const BuilderPtr& builder, const std::vector<OpPtr>& operands,
                py::dict args) {
  ComputationPtr computation = args["computation"].cast<ComputationPtr>();
  std::vector<xla::XlaOp> ops = ExtractXlaOps(operands);
  return xla::Call(builder.get(), computation->computation(), ops);
}

xla::XlaOp Conditional(const BuilderPtr& builder,
                       const std::vector<OpPtr>& operands, py::dict args) {
  ComputationPtr true_computation =
      args["true_computation"].cast<ComputationPtr>();
  ComputationPtr false_computation =
      args["false_computation"].cast<ComputationPtr>();
  return xla::Conditional(operands.at(0)->op, operands.at(1)->op,
                          true_computation->computation(), operands.at(2)->op,
                          false_computation->computation());
}

xla::XlaOp Select(const BuilderPtr& builder, const std::vector<OpPtr>& operands,
                  py::dict args) {
  return xla::Select(operands.at(0)->op, operands.at(1)->op,
                     operands.at(2)->op);
}

xla::XlaOp ShiftLeft(const BuilderPtr& builder,
                     const std::vector<OpPtr>& operands, py::dict args) {
  return xla::ShiftLeft(operands.at(0)->op, operands.at(1)->op);
}

xla::XlaOp ShifRight(const BuilderPtr& builder,
                     const std::vector<OpPtr>& operands, py::dict args) {
  return operands.at(0)->op >> operands.at(1)->op;
}

xla::GatherDimensionNumbers ParseGatherDimensionNumbers(py::dict args) {
  xla::GatherDimensionNumbers dimension_numbers;
  absl::optional<py::tuple> arg_offset_dims =
      ArgOptional<py::tuple>(args, "offset_dims");
  if (arg_offset_dims) {
    for (auto& dim : *arg_offset_dims) {
      dimension_numbers.add_offset_dims(dim.cast<xla::int64>());
    }
  }
  absl::optional<py::tuple> arg_collapsed_slice_dims =
      ArgOptional<py::tuple>(args, "collapsed_slice_dims");
  if (arg_collapsed_slice_dims) {
    for (auto& dim : *arg_collapsed_slice_dims) {
      dimension_numbers.add_collapsed_slice_dims(dim.cast<xla::int64>());
    }
  }
  absl::optional<py::tuple> arg_start_index_map =
      ArgOptional<py::tuple>(args, "start_index_map");
  if (arg_start_index_map) {
    for (auto& dim : *arg_start_index_map) {
      dimension_numbers.add_start_index_map(dim.cast<xla::int64>());
    }
  }
  absl::optional<xla::int64> arg_index_vector_dim =
      ArgOptional<xla::int64>(args, "index_vector_dim");
  if (arg_index_vector_dim) {
    dimension_numbers.set_index_vector_dim(*arg_index_vector_dim);
  }
  return dimension_numbers;
}

xla::XlaOp Gather(const BuilderPtr& builder, const std::vector<OpPtr>& operands,
                  py::dict args) {
  std::vector<xla::int64> slice_sizes =
      GetTupleVector<xla::int64>(args["slice_sizes"]);
  bool indices_are_sorted =
      ArgOrDefault<bool>(args, "indices_are_sorted", false);
  xla::GatherDimensionNumbers dimension_numbers =
      ParseGatherDimensionNumbers(args);
  return xla::Gather(operands.at(0)->op, operands.at(1)->op, dimension_numbers,
                     slice_sizes, indices_are_sorted);
}

xla::ScatterDimensionNumbers ParseScatterDimensionNumbers(py::dict args) {
  xla::ScatterDimensionNumbers dimension_numbers;
  absl::optional<py::tuple> arg_update_window_dims =
      ArgOptional<py::tuple>(args, "update_window_dims");
  if (arg_update_window_dims) {
    for (auto& dim : *arg_update_window_dims) {
      dimension_numbers.add_update_window_dims(dim.cast<xla::int64>());
    }
  }
  absl::optional<py::tuple> arg_inserted_window_dims =
      ArgOptional<py::tuple>(args, "inserted_window_dims");
  if (arg_inserted_window_dims) {
    for (auto& dim : *arg_inserted_window_dims) {
      dimension_numbers.add_inserted_window_dims(dim.cast<xla::int64>());
    }
  }
  absl::optional<xla::int64> arg_index_vector_dim =
      ArgOptional<xla::int64>(args, "index_vector_dim");
  if (arg_index_vector_dim) {
    dimension_numbers.set_index_vector_dim(*arg_index_vector_dim);
  }
  return dimension_numbers;
}

xla::XlaOp Scatter(const BuilderPtr& builder,
                   const std::vector<OpPtr>& operands, py::dict args) {
  bool indices_are_sorted =
      ArgOrDefault<bool>(args, "indices_are_sorted", false);
  bool unique_indices = ArgOrDefault<bool>(args, "unique_indices", false);
  ComputationPtr computation = args["computation"].cast<ComputationPtr>();
  xla::ScatterDimensionNumbers dimension_numbers =
      ParseScatterDimensionNumbers(args);
  return xla::Scatter(operands.at(0)->op, operands.at(1)->op,
                      operands.at(2)->op, computation->computation(),
                      dimension_numbers, indices_are_sorted, unique_indices);
}

xla::XlaOp Sort(const BuilderPtr& builder, const std::vector<OpPtr>& operands,
                py::dict args) {
  bool is_stable = ArgOrDefault<bool>(args, "is_stable", false);
  xla::int64 dimension = ArgOrDefault<xla::int64>(args, "dimension", -1);
  ComputationPtr comparator = args["comparator"].cast<ComputationPtr>();
  std::vector<xla::XlaOp> ops = ExtractXlaOps(operands);
  return xla::Sort(ops, comparator->computation(), dimension, is_stable);
}

xla::XlaOp Iota(const BuilderPtr& builder, const std::vector<OpPtr>& operands,
                py::dict args) {
  xla::Shape shape = PyShapeToShape(args["shape"]);
  xla::int64 iota_dimension = args["iota_dimension"].cast<xla::int64>();
  return xla::Iota(builder.get(), shape, iota_dimension);
}

xla::XlaOp Clamp(const BuilderPtr& builder, const std::vector<OpPtr>& operands,
                 py::dict args) {
  return xla::Clamp(operands.at(1)->op, operands.at(0)->op, operands.at(2)->op);
}

xla::XlaOp Rev(const BuilderPtr& builder, const std::vector<OpPtr>& operands,
               py::dict args) {
  std::vector<xla::int64> dimensions =
      GetTupleVector<xla::int64>(args["dimensions"]);
  return xla::Rev(operands.at(0)->op, dimensions);
}

xla::XlaOp ConcatInDim(const BuilderPtr& builder,
                       const std::vector<OpPtr>& operands, py::dict args) {
  xla::int64 dimension = args["dimension"].cast<xla::int64>();
  std::vector<xla::XlaOp> ops = ExtractXlaOps(operands);
  return xla::ConcatInDim(builder.get(), ops, dimension);
}

xla::XlaOp Convert(const BuilderPtr& builder,
                   const std::vector<OpPtr>& operands, py::dict args) {
  std::string type = args["to_type"].cast<std::string>();
  xla::PrimitiveType xla_type =
      ConsumeValue(xla::primitive_util::StringToPrimitiveType(type));
  return MaybeConvertTo(operands.at(0)->op, xla_type);
}

xla::XlaOp GetTupleElement(const BuilderPtr& builder,
                           const std::vector<OpPtr>& operands, py::dict args) {
  xla::int64 index = args["index"].cast<xla::int64>();
  return xla::GetTupleElement(operands.at(0)->op, index);
}

xla::XlaOp BitcastConvert(const BuilderPtr& builder,
                          const std::vector<OpPtr>& operands, py::dict args) {
  std::string type = args["to_type"].cast<std::string>();
  xla::PrimitiveType xla_type =
      ConsumeValue(xla::primitive_util::StringToPrimitiveType(type));
  return xla::BitcastConvertType(operands.at(0)->op, xla_type);
}

const XlaOpFunctionMap* CreateXlaOpFunctionMap() {
  XlaOpFunctionMap* fn_map = new XlaOpFunctionMap();

#define XLA_OPADD(name) fn_map->emplace(#name, name)

  XLA_OPADD(Abs);
  XLA_OPADD(Add);
  XLA_OPADD(And);
  XLA_OPADD(Acos);
  XLA_OPADD(Asin);
  XLA_OPADD(Atan2);
  XLA_OPADD(Atan);
  XLA_OPADD(BitcastConvert);
  XLA_OPADD(Broadcast);
  XLA_OPADD(BroadcastInDim);
  XLA_OPADD(Call);
  XLA_OPADD(Ceil);
  XLA_OPADD(Clamp);
  XLA_OPADD(ConcatInDim);
  XLA_OPADD(Conditional);
  XLA_OPADD(Constant);
  XLA_OPADD(Conv);
  XLA_OPADD(Convert);
  XLA_OPADD(ConvGeneral);
  XLA_OPADD(ConvGeneralDilated);
  XLA_OPADD(ConvWithGeneralDimensions);
  XLA_OPADD(ConvWithGeneralPadding);
  XLA_OPADD(Cos);
  XLA_OPADD(Cosh);
  XLA_OPADD(Div);
  XLA_OPADD(Dot);
  XLA_OPADD(DynamicReshape);
  XLA_OPADD(DynamicSlice);
  XLA_OPADD(DynamicUpdateSlice);
  XLA_OPADD(Eq);
  XLA_OPADD(Erf);
  XLA_OPADD(Erfc);
  XLA_OPADD(ErfInv);
  XLA_OPADD(Exp);
  XLA_OPADD(Expm1);
  XLA_OPADD(Floor);
  XLA_OPADD(Gather);
  XLA_OPADD(Ge);
  XLA_OPADD(GetTupleElement);
  XLA_OPADD(Gt);
  XLA_OPADD(Iota);
  XLA_OPADD(Le);
  XLA_OPADD(Log);
  XLA_OPADD(Log1p);
  XLA_OPADD(Lt);
  XLA_OPADD(Max);
  XLA_OPADD(Min);
  XLA_OPADD(Mul);
  XLA_OPADD(Ne);
  XLA_OPADD(Neg);
  XLA_OPADD(Not);
  XLA_OPADD(Or);
  XLA_OPADD(Pad);
  XLA_OPADD(Pow);
  XLA_OPADD(Reduce);
  XLA_OPADD(Rem);
  XLA_OPADD(Reshape);
  XLA_OPADD(Rev);
  XLA_OPADD(Rsqrt);
  XLA_OPADD(Scatter);
  XLA_OPADD(Select);
  XLA_OPADD(ShiftLeft);
  XLA_OPADD(ShifRight);
  XLA_OPADD(Sin);
  XLA_OPADD(Sinh);
  XLA_OPADD(Slice);
  XLA_OPADD(SliceInDim);
  XLA_OPADD(Sort);
  XLA_OPADD(Sqrt);
  XLA_OPADD(Sub);
  XLA_OPADD(Tan);
  XLA_OPADD(Tanh);
  XLA_OPADD(Transpose);
  XLA_OPADD(Tuple);
  XLA_OPADD(Xor);

#undef XLA_OPADD

  return fn_map;
}

const XlaOpFunctionMap* GetXlaOpFunctionMap() {
  static const XlaOpFunctionMap* fn_map = CreateXlaOpFunctionMap();
  return fn_map;
}

}  // namespace

py::object ShapeToPyShape(const xla::Shape& shape) {
  if (shape.IsTuple()) {
    py::tuple py_shapes(shape.tuple_shapes_size());
    for (size_t i = 0; i < shape.tuple_shapes_size(); ++i) {
      py_shapes[i] = ShapeToPyShape(shape.tuple_shapes(i));
    }
    return py_shapes;
  }
  py::dict py_shape;
  py_shape["type"] = py::cast(
      xla::primitive_util::LowercasePrimitiveTypeName(shape.element_type()));
  auto sizes = py::tuple(shape.rank());
  for (xla::int64 i = 0; i < shape.rank(); ++i) {
    sizes[i] = py::cast(shape.dimensions(i));
  }
  py_shape["sizes"] = sizes;
  return py_shape;
}

xla::Shape PyShapeToShape(py::object shape) {
  if (py::isinstance<py::tuple>(shape)) {
    py::tuple py_shape = shape.cast<py::tuple>();
    std::vector<xla::Shape> shapes;
    for (auto& tshape : py_shape) {
      shapes.emplace_back(PyShapeToShape(tshape.cast<py::object>()));
    }
    return xla::ShapeUtil::MakeTupleShape(shapes);
  }
  py::dict py_shape = shape.cast<py::dict>();
  std::string type = py_shape["type"].cast<std::string>();
  std::vector<xla::int64> dimensions =
      GetTupleVector<xla::int64>(py_shape["sizes"].cast<py::tuple>());
  xla::PrimitiveType xla_type =
      ConsumeValue(xla::primitive_util::StringToPrimitiveType(type));
  return xla::ShapeUtil::MakeShape(xla_type, dimensions);
}

OpPtr CreateOp(BuilderPtr builder, const std::string& opname,
               const std::vector<OpPtr>& operands, py::dict args) {
  const XlaOpFunctionMap* fn_map = GetXlaOpFunctionMap();
  auto it = fn_map->find(opname);
  if (it == fn_map->end()) {
    XLA_ERROR() << "Unknown XLA op name: " << opname;
  }
  xla::XlaOp result = (*it->second)(builder, operands, args);
  return std::make_shared<Op>(std::move(builder), std::move(result));
}

}  // namespace op_builder
}  // namespace torch_xla
