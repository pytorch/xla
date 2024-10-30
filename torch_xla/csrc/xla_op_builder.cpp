#include "torch_xla/csrc/xla_op_builder.h"

#include <map>

#include "absl/types/optional.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/tensor_util.h"
#include "xla/client/lib/math.h"
#include "xla/client/lib/matrix.h"
#include "xla/client/lib/pooling.h"
#include "xla/hlo/builder/lib/logdet.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"

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
XLA_UNARY_OP(Clz);
XLA_UNARY_OP(Conj);
XLA_UNARY_OP(Cos);
XLA_UNARY_OP(Cosh);
XLA_UNARY_OP(Erf);
XLA_UNARY_OP(Erfc);
XLA_UNARY_OP(ErfInv);
XLA_UNARY_OP(Exp);
XLA_UNARY_OP(Expm1);
XLA_UNARY_OP(Floor);
XLA_UNARY_OP(Imag);
XLA_UNARY_OP(Log);
XLA_UNARY_OP(Log1p);
XLA_UNARY_OP(Neg);
XLA_UNARY_OP(Not);
XLA_UNARY_OP(Sqrt);
XLA_UNARY_OP(Real);
XLA_UNARY_OP(Rsqrt);
XLA_UNARY_OP(Sin);
XLA_UNARY_OP(Sinh);
XLA_UNARY_OP(Square);
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
py::tuple MakeTuple(const T& container) {
  py::tuple py_tuple(container.size());
  for (size_t i = 0; i < container.size(); ++i) {
    py_tuple[i] = py::cast(container[i]);
  }
  return py_tuple;
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

xla::Padding ParsePadding(const std::string& padding_str) {
  if (padding_str == "same") {
    return xla::Padding::kSame;
  }
  if (padding_str == "valid") {
    return xla::Padding::kValid;
  }
  XLA_ERROR() << "Invalid padding: " << padding_str;
}

xla::XlaOp Reshape(const BuilderPtr& builder,
                   const std::vector<OpPtr>& operands, py::dict args) {
  std::vector<int64_t> sizes = GetTupleVector<int64_t>(args["sizes"]);
  absl::optional<py::tuple> arg_dimensions =
      ArgOptional<py::tuple>(args, "dimensions");
  if (arg_dimensions) {
    std::vector<int64_t> dimensions = GetTupleVector<int64_t>(*arg_dimensions);
    return xla::Reshape(operands.at(0)->op, dimensions, sizes);
  }
  int64_t inferred_dimension =
      ArgOrDefault<int64_t>(args, "inferred_dimension", -1);
  if (inferred_dimension >= 0) {
    return xla::ReshapeWithInferredDimension(operands.at(0)->op, sizes,
                                             inferred_dimension);
  }
  return xla::Reshape(operands.at(0)->op, sizes);
}

xla::XlaOp DynamicReshape(const BuilderPtr& builder,
                          const std::vector<OpPtr>& operands, py::dict args) {
  std::vector<int64_t> sizes = GetTupleVector<int64_t>(args["sizes"]);
  return XlaHelpers::DynamicReshape(operands.at(0)->op, sizes);
}

xla::XlaOp Broadcast(const BuilderPtr& builder,
                     const std::vector<OpPtr>& operands, py::dict args) {
  std::vector<int64_t> sizes = GetTupleVector<int64_t>(args["sizes"]);
  return xla::Broadcast(operands.at(0)->op, sizes);
}

xla::XlaOp BroadcastInDim(const BuilderPtr& builder,
                          const std::vector<OpPtr>& operands, py::dict args) {
  std::vector<int64_t> sizes = GetTupleVector<int64_t>(args["sizes"]);
  std::vector<int64_t> dimensions = GetTupleVector<int64_t>(args["dimensions"]);
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
    dim->set_edge_padding_low(dims[0].cast<int64_t>());
    dim->set_edge_padding_high(dims[1].cast<int64_t>());
    dim->set_interior_padding(dims[2].cast<int64_t>());
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
  std::vector<int64_t> permutation =
      GetTupleVector<int64_t>(args["permutation"]);
  return xla::Transpose(operands.at(0)->op, permutation);
}

xla::ConvolutionDimensionNumbers ParseConvolutionDimensionNumbers(
    py::dict args) {
  xla::ConvolutionDimensionNumbers dimension_numbers;
  XLA_PBSET(dimension_numbers, input_batch_dimension, int64_t, args);
  XLA_PBSET(dimension_numbers, input_feature_dimension, int64_t, args);
  XLA_PBSET(dimension_numbers, kernel_input_feature_dimension, int64_t, args);
  XLA_PBSET(dimension_numbers, kernel_output_feature_dimension, int64_t, args);
  XLA_PBSET(dimension_numbers, output_batch_dimension, int64_t, args);
  XLA_PBSET(dimension_numbers, output_feature_dimension, int64_t, args);
  XLA_PBSET_REP(dimension_numbers, input_spatial_dimensions, int64_t, args);
  XLA_PBSET_REP(dimension_numbers, kernel_spatial_dimensions, int64_t, args);
  XLA_PBSET_REP(dimension_numbers, output_spatial_dimensions, int64_t, args);
  return dimension_numbers;
}

xla::XlaOp ConvWithGeneralPadding(const BuilderPtr& builder,
                                  const std::vector<OpPtr>& operands,
                                  py::dict args) {
  std::vector<int64_t> window_strides =
      GetTupleVector<int64_t>(args["window_strides"]);
  int64_t feature_group_count =
      ArgOrDefault<int64_t>(args, "feature_group_count", 1);
  int64_t batch_group_count =
      ArgOrDefault<int64_t>(args, "batch_group_count", 1);
  auto padding = ParsePairList<int64_t>(args["padding"]);
  xla::PrecisionConfig precision_config = DotPrecisonConfig(args);
  return xla::ConvWithGeneralPadding(
      operands.at(0)->op, operands.at(1)->op, window_strides, padding,
      feature_group_count, batch_group_count, &precision_config);
}

xla::XlaOp ConvWithGeneralDimensions(const BuilderPtr& builder,
                                     const std::vector<OpPtr>& operands,
                                     py::dict args) {
  std::vector<int64_t> window_strides =
      GetTupleVector<int64_t>(args["window_strides"]);
  int64_t feature_group_count =
      ArgOrDefault<int64_t>(args, "feature_group_count", 1);
  int64_t batch_group_count =
      ArgOrDefault<int64_t>(args, "batch_group_count", 1);
  xla::Padding padding = ParsePadding(args["padding"].cast<std::string>());
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
  std::vector<int64_t> window_strides =
      GetTupleVector<int64_t>(args["window_strides"]);
  int64_t feature_group_count =
      ArgOrDefault<int64_t>(args, "feature_group_count", 1);
  int64_t batch_group_count =
      ArgOrDefault<int64_t>(args, "batch_group_count", 1);
  auto padding = ParsePairList<int64_t>(args["padding"]);
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
  std::vector<int64_t> window_strides =
      GetTupleVector<int64_t>(args["window_strides"]);
  std::vector<int64_t> lhs_dilation =
      GetTupleVector<int64_t>(args["lhs_dilation"]);
  std::vector<int64_t> rhs_dilation =
      GetTupleVector<int64_t>(args["rhs_dilation"]);
  int64_t feature_group_count =
      ArgOrDefault<int64_t>(args, "feature_group_count", 1);
  int64_t batch_group_count =
      ArgOrDefault<int64_t>(args, "batch_group_count", 1);
  auto padding = ParsePairList<int64_t>(args["padding"]);
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
  std::vector<int64_t> window_strides =
      GetTupleVector<int64_t>(args["window_strides"]);
  int64_t feature_group_count =
      ArgOrDefault<int64_t>(args, "feature_group_count", 1);
  int64_t batch_group_count =
      ArgOrDefault<int64_t>(args, "batch_group_count", 1);
  xla::Padding padding = ParsePadding(args["padding"].cast<std::string>());
  xla::PrecisionConfig precision_config = DotPrecisonConfig(args);
  return xla::Conv(operands.at(0)->op, operands.at(1)->op, window_strides,
                   padding, feature_group_count, batch_group_count,
                   &precision_config);
}

xla::XlaOp Slice(const BuilderPtr& builder, const std::vector<OpPtr>& operands,
                 py::dict args) {
  std::vector<int64_t> start_indices =
      GetTupleVector<int64_t>(args["start_indices"]);
  std::vector<int64_t> limit_indices =
      GetTupleVector<int64_t>(args["limit_indices"]);
  std::vector<int64_t> strides = GetTupleVector<int64_t>(args["strides"]);
  return xla::Slice(operands.at(0)->op, start_indices, limit_indices, strides);
}

xla::XlaOp SliceInDim(const BuilderPtr& builder,
                      const std::vector<OpPtr>& operands, py::dict args) {
  int64_t start_index = args["start_index"].cast<int64_t>();
  int64_t limit_index = args["limit_index"].cast<int64_t>();
  int64_t dimno = args["dimno"].cast<int64_t>();
  int64_t stride = ArgOrDefault<int64_t>(args, "stride", 1);
  return xla::SliceInDim(operands.at(0)->op, start_index, limit_index, stride,
                         dimno);
}

xla::XlaOp DynamicSlice(const BuilderPtr& builder,
                        const std::vector<OpPtr>& operands, py::dict args) {
  std::vector<int64_t> slice_sizes =
      GetTupleVector<int64_t>(args["slice_sizes"]);
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
  std::vector<int64_t> dimensions = GetTupleVector<int64_t>(args["dimensions"]);
  runtime::ComputationClient::ComputationPtr computation =
      args["computation"].cast<runtime::ComputationClient::ComputationPtr>();
  return xla::Reduce(operands.at(0)->op, operands.at(1)->op,
                     computation->computation(), dimensions);
}

xla::XlaOp ReduceAll(const BuilderPtr& builder,
                     const std::vector<OpPtr>& operands, py::dict args) {
  runtime::ComputationClient::ComputationPtr computation =
      args["computation"].cast<runtime::ComputationClient::ComputationPtr>();
  return xla::ReduceAll(operands.at(0)->op, operands.at(1)->op,
                        computation->computation());
}

xla::XlaOp ReduceWindow(const BuilderPtr& builder,
                        const std::vector<OpPtr>& operands, py::dict args) {
  std::vector<int64_t> window_dimensions =
      GetTupleVector<int64_t>(args["window_dimensions"]);
  std::vector<int64_t> window_strides =
      GetTupleVector<int64_t>(args["window_strides"]);
  runtime::ComputationClient::ComputationPtr computation =
      args["computation"].cast<runtime::ComputationClient::ComputationPtr>();
  xla::Padding padding = ParsePadding(args["padding"].cast<std::string>());
  return xla::ReduceWindow(operands.at(0)->op, operands.at(1)->op,
                           computation->computation(), window_dimensions,
                           window_strides, padding);
}

xla::XlaOp Map(const BuilderPtr& builder, const std::vector<OpPtr>& operands,
               py::dict args) {
  std::vector<int64_t> dimensions = GetTupleVector<int64_t>(args["dimensions"]);
  runtime::ComputationClient::ComputationPtr computation =
      args["computation"].cast<runtime::ComputationClient::ComputationPtr>();
  std::vector<xla::XlaOp> static_operands =
      GetOpVector(args["static_operands"].cast<py::tuple>());
  std::vector<xla::XlaOp> ops = ExtractXlaOps(operands);
  return xla::Map(builder.get(), ops, computation->computation(), dimensions,
                  static_operands);
}

xla::XlaOp Call(const BuilderPtr& builder, const std::vector<OpPtr>& operands,
                py::dict args) {
  runtime::ComputationClient::ComputationPtr computation =
      args["computation"].cast<runtime::ComputationClient::ComputationPtr>();
  std::vector<xla::XlaOp> ops = ExtractXlaOps(operands);
  return xla::Call(builder.get(), computation->computation(), ops);
}

xla::XlaOp Conditional(const BuilderPtr& builder,
                       const std::vector<OpPtr>& operands, py::dict args) {
  runtime::ComputationClient::ComputationPtr true_computation =
      args["true_computation"]
          .cast<runtime::ComputationClient::ComputationPtr>();
  runtime::ComputationClient::ComputationPtr false_computation =
      args["false_computation"]
          .cast<runtime::ComputationClient::ComputationPtr>();
  return xla::Conditional(operands.at(0)->op, operands.at(1)->op,
                          true_computation->computation(), operands.at(2)->op,
                          false_computation->computation());
}

xla::XlaOp While(const BuilderPtr& builder, const std::vector<OpPtr>& operands,
                 py::dict args) {
  runtime::ComputationClient::ComputationPtr condition_computation =
      args["condition_computation"]
          .cast<runtime::ComputationClient::ComputationPtr>();
  runtime::ComputationClient::ComputationPtr body_computation =
      args["body_computation"]
          .cast<runtime::ComputationClient::ComputationPtr>();
  return xla::While(condition_computation->computation(),
                    body_computation->computation(), operands.at(0)->op);
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
      dimension_numbers.add_offset_dims(dim.cast<int64_t>());
    }
  }
  absl::optional<py::tuple> arg_collapsed_slice_dims =
      ArgOptional<py::tuple>(args, "collapsed_slice_dims");
  if (arg_collapsed_slice_dims) {
    for (auto& dim : *arg_collapsed_slice_dims) {
      dimension_numbers.add_collapsed_slice_dims(dim.cast<int64_t>());
    }
  }
  absl::optional<py::tuple> arg_start_index_map =
      ArgOptional<py::tuple>(args, "start_index_map");
  if (arg_start_index_map) {
    for (auto& dim : *arg_start_index_map) {
      dimension_numbers.add_start_index_map(dim.cast<int64_t>());
    }
  }
  absl::optional<int64_t> arg_index_vector_dim =
      ArgOptional<int64_t>(args, "index_vector_dim");
  if (arg_index_vector_dim) {
    dimension_numbers.set_index_vector_dim(*arg_index_vector_dim);
  }
  return dimension_numbers;
}

xla::XlaOp Gather(const BuilderPtr& builder, const std::vector<OpPtr>& operands,
                  py::dict args) {
  std::vector<int64_t> slice_sizes =
      GetTupleVector<int64_t>(args["slice_sizes"]);
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
      dimension_numbers.add_update_window_dims(dim.cast<int64_t>());
    }
  }
  absl::optional<py::tuple> arg_inserted_window_dims =
      ArgOptional<py::tuple>(args, "inserted_window_dims");
  if (arg_inserted_window_dims) {
    for (auto& dim : *arg_inserted_window_dims) {
      dimension_numbers.add_inserted_window_dims(dim.cast<int64_t>());
    }
  }
  absl::optional<py::tuple> arg_scatter_dims_to_operand_dims =
      ArgOptional<py::tuple>(args, "scatter_dims_to_operand_dims");
  if (arg_scatter_dims_to_operand_dims) {
    for (auto& dim : *arg_scatter_dims_to_operand_dims) {
      dimension_numbers.add_scatter_dims_to_operand_dims(dim.cast<int64_t>());
    }
  }
  absl::optional<int64_t> arg_index_vector_dim =
      ArgOptional<int64_t>(args, "index_vector_dim");
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
  runtime::ComputationClient::ComputationPtr computation =
      args["computation"].cast<runtime::ComputationClient::ComputationPtr>();
  xla::ScatterDimensionNumbers dimension_numbers =
      ParseScatterDimensionNumbers(args);
  return xla::Scatter(operands.at(0)->op, operands.at(1)->op,
                      operands.at(2)->op, computation->computation(),
                      dimension_numbers, indices_are_sorted, unique_indices);
}

xla::XlaOp SelectAndScatter(const BuilderPtr& builder,
                            const std::vector<OpPtr>& operands, py::dict args) {
  runtime::ComputationClient::ComputationPtr select_computation =
      args["select_computation"]
          .cast<runtime::ComputationClient::ComputationPtr>();
  runtime::ComputationClient::ComputationPtr scatter_computation =
      args["scatter_computation"]
          .cast<runtime::ComputationClient::ComputationPtr>();
  std::vector<int64_t> window_dimensions =
      GetTupleVector<int64_t>(args["window_dimensions"]);
  std::vector<int64_t> window_strides =
      GetTupleVector<int64_t>(args["window_strides"]);
  xla::Padding padding = ParsePadding(args["padding"].cast<std::string>());
  return xla::SelectAndScatter(
      operands.at(0)->op, select_computation->computation(), window_dimensions,
      window_strides, padding, operands.at(1)->op, operands.at(2)->op,
      scatter_computation->computation());
}

xla::XlaOp SelectAndScatterWithGeneralPadding(
    const BuilderPtr& builder, const std::vector<OpPtr>& operands,
    py::dict args) {
  runtime::ComputationClient::ComputationPtr select_computation =
      args["select_computation"]
          .cast<runtime::ComputationClient::ComputationPtr>();
  runtime::ComputationClient::ComputationPtr scatter_computation =
      args["scatter_computation"]
          .cast<runtime::ComputationClient::ComputationPtr>();
  std::vector<int64_t> window_dimensions =
      GetTupleVector<int64_t>(args["window_dimensions"]);
  std::vector<int64_t> window_strides =
      GetTupleVector<int64_t>(args["window_strides"]);
  auto padding = ParsePairList<int64_t>(args["padding"]);
  return xla::SelectAndScatterWithGeneralPadding(
      operands.at(0)->op, select_computation->computation(), window_dimensions,
      window_strides, padding, operands.at(1)->op, operands.at(2)->op,
      scatter_computation->computation());
}

xla::TensorFormat ParseTensorFormat(py::dict args) {
  int64_t batch_dimension = args["batch_dimension"].cast<int64_t>();
  int64_t feature_dimension = args["feature_dimension"].cast<int64_t>();
  std::vector<int64_t> spatial_dimensions =
      GetTupleVector<int64_t>(args["spatial_dimensions"]);
  return xla::TensorFormat(batch_dimension, feature_dimension,
                           spatial_dimensions);
}

xla::XlaOp MaxPool(const BuilderPtr& builder,
                   const std::vector<OpPtr>& operands, py::dict args) {
  std::vector<int64_t> kernel_size =
      GetTupleVector<int64_t>(args["kernel_size"]);
  std::vector<int64_t> stride = GetTupleVector<int64_t>(args["stride"]);
  xla::Padding padding = ParsePadding(args["padding"].cast<std::string>());
  xla::TensorFormat data_format = ParseTensorFormat(args);
  return xla::MaxPool(operands.at(0)->op, kernel_size, stride, padding,
                      data_format);
}

xla::XlaOp Sort(const BuilderPtr& builder, const std::vector<OpPtr>& operands,
                py::dict args) {
  bool is_stable = ArgOrDefault<bool>(args, "is_stable", false);
  int64_t dimension = ArgOrDefault<int64_t>(args, "dimension", -1);
  runtime::ComputationClient::ComputationPtr comparator =
      args["comparator"].cast<runtime::ComputationClient::ComputationPtr>();
  std::vector<xla::XlaOp> ops = ExtractXlaOps(operands);
  return xla::Sort(ops, comparator->computation(), dimension, is_stable);
}

xla::XlaOp Iota(const BuilderPtr& builder, const std::vector<OpPtr>& operands,
                py::dict args) {
  xla::Shape shape = PyShapeToShape(args["shape"]);
  int64_t iota_dimension = args["iota_dimension"].cast<int64_t>();
  return xla::Iota(builder.get(), shape, iota_dimension);
}

xla::XlaOp Clamp(const BuilderPtr& builder, const std::vector<OpPtr>& operands,
                 py::dict args) {
  return xla::Clamp(operands.at(1)->op, operands.at(0)->op, operands.at(2)->op);
}

xla::XlaOp Rev(const BuilderPtr& builder, const std::vector<OpPtr>& operands,
               py::dict args) {
  std::vector<int64_t> dimensions = GetTupleVector<int64_t>(args["dimensions"]);
  return xla::Rev(operands.at(0)->op, dimensions);
}

xla::XlaOp ConcatInDim(const BuilderPtr& builder,
                       const std::vector<OpPtr>& operands, py::dict args) {
  int64_t dimension = args["dimension"].cast<int64_t>();
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
  int64_t index = args["index"].cast<int64_t>();
  return xla::GetTupleElement(operands.at(0)->op, index);
}

xla::XlaOp GetDimensionSize(const BuilderPtr& builder,
                            const std::vector<OpPtr>& operands, py::dict args) {
  int64_t dimension = args["dimension"].cast<int64_t>();
  return xla::GetDimensionSize(operands.at(0)->op, dimension);
}

xla::XlaOp SetDimensionSize(const BuilderPtr& builder,
                            const std::vector<OpPtr>& operands, py::dict args) {
  int64_t dimension = args["dimension"].cast<int64_t>();
  return xla::SetDimensionSize(operands.at(0)->op, operands.at(1)->op,
                               dimension);
}

xla::XlaOp BitcastConvert(const BuilderPtr& builder,
                          const std::vector<OpPtr>& operands, py::dict args) {
  std::string type = args["to_type"].cast<std::string>();
  xla::PrimitiveType xla_type =
      ConsumeValue(xla::primitive_util::StringToPrimitiveType(type));
  return xla::BitcastConvertType(operands.at(0)->op, xla_type);
}

xla::XlaOp TriangularSolve(const BuilderPtr& builder,
                           const std::vector<OpPtr>& operands, py::dict args) {
  bool left_side = ArgOrDefault<bool>(args, "left_side", true);
  bool lower = ArgOrDefault<bool>(args, "lower", false);
  bool unit_diagonal = ArgOrDefault<bool>(args, "unit_diagonal", false);
  bool transpose_a = ArgOrDefault<bool>(args, "transpose_a", false);
  return xla::TriangularSolve(
      operands.at(0)->op, operands.at(1)->op, left_side, lower, unit_diagonal,
      transpose_a ? xla::TriangularSolveOptions::TRANSPOSE
                  : xla::TriangularSolveOptions::NO_TRANSPOSE);
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
  XLA_OPADD(Clz);
  XLA_OPADD(ConcatInDim);
  XLA_OPADD(Conj);
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
  XLA_OPADD(GetDimensionSize);
  XLA_OPADD(GetTupleElement);
  XLA_OPADD(Gt);
  XLA_OPADD(Imag);
  XLA_OPADD(Iota);
  XLA_OPADD(Le);
  XLA_OPADD(Log);
  XLA_OPADD(Log1p);
  XLA_OPADD(Lt);
  XLA_OPADD(Map);
  XLA_OPADD(Max);
  XLA_OPADD(MaxPool);
  XLA_OPADD(Min);
  XLA_OPADD(Mul);
  XLA_OPADD(Ne);
  XLA_OPADD(Neg);
  XLA_OPADD(Not);
  XLA_OPADD(Or);
  XLA_OPADD(Pad);
  XLA_OPADD(Pow);
  XLA_OPADD(Real);
  XLA_OPADD(Reduce);
  XLA_OPADD(ReduceAll);
  XLA_OPADD(ReduceWindow);
  XLA_OPADD(Rem);
  XLA_OPADD(Reshape);
  XLA_OPADD(Rev);
  XLA_OPADD(Rsqrt);
  XLA_OPADD(Scatter);
  XLA_OPADD(Select);
  XLA_OPADD(SelectAndScatter);
  XLA_OPADD(SelectAndScatterWithGeneralPadding);
  XLA_OPADD(SetDimensionSize);
  XLA_OPADD(ShiftLeft);
  XLA_OPADD(ShifRight);
  XLA_OPADD(Sin);
  XLA_OPADD(Sinh);
  XLA_OPADD(Slice);
  XLA_OPADD(SliceInDim);
  XLA_OPADD(Sort);
  XLA_OPADD(Sqrt);
  XLA_OPADD(Square);
  XLA_OPADD(Sub);
  XLA_OPADD(Tan);
  XLA_OPADD(Tanh);
  XLA_OPADD(Transpose);
  XLA_OPADD(TriangularSolve);
  XLA_OPADD(Tuple);
  XLA_OPADD(While);
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
  py_shape["sizes"] = MakeTuple(shape.dimensions());
  if (shape.is_dynamic()) {
    py_shape["dynamic_dimensions"] = MakeTuple(shape.dynamic_dimensions());
  }
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
  std::vector<int64_t> dimensions =
      GetTupleVector<int64_t>(py_shape["sizes"].cast<py::tuple>());
  xla::PrimitiveType xla_type =
      ConsumeValue(xla::primitive_util::StringToPrimitiveType(type));
  if (py_shape.contains("dynamic_dimensions")) {
    std::vector<bool> dynamic_dimensions =
        GetTupleVector<bool>(py_shape["dynamic_dimensions"]);
    return xla::ShapeUtil::MakeShape(xla_type, dimensions, dynamic_dimensions);
  }
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
