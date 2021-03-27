#include "tensorflow/compiler/xla/client/xla_builder.h"

#include "lazy_xla/csrc/compiler/debug_macros.h"
#include "lazy_xla/csrc/compiler/helpers.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/permutation_util.h"

using namespace torch::jit::tensorexpr;

namespace xla {
namespace {

ExprHandle call(Placeholder* buf, const std::vector<ExprHandle>& indices) {
  return buf->load(indices);
}

ExprHandle call(Tensor* tensor, const std::vector<ExprHandle>& indices) {
  return tensor->call(indices);
}

std::vector<DimArg> dims(Placeholder* buf) {
  std::vector<DimArg> dimensions;
  for (const auto dim : buf->dims()) {
    dimensions.emplace_back(ExprHandle(dim));
  }
  return dimensions;
}

std::vector<DimArg> dims(Tensor* tensor) {
  std::vector<DimArg> dimensions;
  for (const auto dim : tensor->buf()->dims()) {
    dimensions.emplace_back(ExprHandle(dim));
  }
  return dimensions;
}

std::vector<ExprHandle> VarsToExprs(const std::vector<VarHandle>& vars) {
  std::vector<ExprHandle> exprs;
  for (const auto var : vars) {
    exprs.push_back(var);
  }
  return exprs;
}

XlaOp BinaryOp(XlaOp lhs, XlaOp rhs,
               const std::function<ExprHandle(const ExprHandle&,
                                              const ExprHandle&)>& bin_op,
               const std::string& name,
               absl::Span<const int64> broadcast_dimensions,
               PrimitiveType output_element_type) {
  if (!broadcast_dimensions.empty()) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
  std::tie(lhs, rhs) =
      torch_lazy_tensors::compiler::XlaHelpers::PromoteShapes(lhs, rhs);
  const auto& shape =
      torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(lhs);
  return XlaOp(Compute(name, lhs.dims(),
                       [&](const std::vector<VarHandle>& indices) {
                         const auto expr_indices = VarsToExprs(indices);
                         return bin_op(lhs.call(expr_indices),
                                       rhs.call(expr_indices));
                       }),
               std::make_unique<Shape>(output_element_type, shape.dimensions()),
               lhs.builder());
}

XlaOp BinaryOp(XlaOp lhs, XlaOp rhs,
               const std::function<ExprHandle(const ExprHandle&,
                                              const ExprHandle&)>& bin_op,
               const std::string& name,
               absl::Span<const int64> broadcast_dimensions) {
  std::tie(lhs, rhs) =
      torch_lazy_tensors::compiler::XlaHelpers::PromoteShapes(lhs, rhs);
  const auto& shape =
      torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(lhs);
  return BinaryOp(lhs, rhs, bin_op, name, broadcast_dimensions,
                  shape.element_type());
}

XlaOp UnaryOp(XlaOp input,
              const std::function<ExprHandle(const ExprHandle&)>& unary_op,
              const std::string& name, std::unique_ptr<Shape> shape) {
  return XlaOp(Compute(name, input.dims(),
                       [&](const std::vector<VarHandle>& indices) {
                         const auto expr_indices = VarsToExprs(indices);
                         return unary_op(input.call(expr_indices));
                       }),
               std::move(shape), input.builder());
}

std::tuple<XlaOp, XlaOp, PrimitiveType> PromoteToInteger(XlaOp x, XlaOp y) {
  const auto element_type =
      torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(x).element_type();
  XLA_CHECK(element_type == PrimitiveType::S32 ||
            element_type == PrimitiveType::PRED)
      << "Element type not supported for bitwise " << element_type;
  if (element_type == PrimitiveType::PRED) {
    return {ConvertElementType(x, PrimitiveType::S32),
            ConvertElementType(y, PrimitiveType::S32), element_type};
  }
  return {x, y, element_type};
}

ExprHandle IndexIsInsidePadding(
    const PaddingConfig::PaddingConfigDimension& padding_dimension,
    const ExprHandle& index, int64 dim_size) {
  ExprHandle low_padding_end(
      static_cast<int>(padding_dimension.edge_padding_low()));
  int interior_padding = padding_dimension.interior_padding();
  ExprHandle high_padding_begin(
      static_cast<int>(padding_dimension.edge_padding_low() +
                       (dim_size - 1) * (interior_padding + 1) + 1));
  auto is_low_padding = cast<int>(index < low_padding_end);
  auto is_high_padding = cast<int>(index >= high_padding_begin);
  auto is_interior_padding =
      cast<int>((index - low_padding_end) % ExprHandle(interior_padding + 1) !=
                ExprHandle(0));
  return is_low_padding | is_high_padding | is_interior_padding;
}

std::vector<ExprHandle> LinearToMultiDimIndex(const ExprHandle& index,
                                              absl::Span<const int64> sizes) {
  std::vector<ExprHandle> indices;
  ExprHandle linear_index = index;
  for (ssize_t i = sizes.size() - 1; i >= 0; --i) {
    ExprHandle dim_index = linear_index;
    int output_dim_size = sizes[i];
    if (i > 0) {
      dim_index = Mod::make(linear_index, ExprHandle(output_dim_size));
    }
    indices.push_back(dim_index);
    linear_index = linear_index / ExprHandle(output_dim_size);
  }
  std::reverse(indices.begin(), indices.end());
  return indices;
}

std::vector<ExprHandle> MultiDimToLinearIndex(
    const std::vector<ExprHandle>& indices,
    const std::vector<ExprHandle>& dims) {
  std::vector<ExprHandle> linearized_index;
  linearized_index.emplace_back(
      flatten_index(ExprHandleVectorToExprVector(dims),
                    ExprHandleVectorToExprVector(indices)));
  return linearized_index;
}

std::vector<DimArg> LinearizedSize(absl::Span<const int64> sizes) {
  int element_count = lazy_tensors::util::Multiply<int>(sizes);
  return std::vector<DimArg>{ExprHandle(element_count)};
}

std::vector<ExprHandle> DimsAsExprHandles(absl::Span<const int64> dims) {
  std::vector<ExprHandle> expr_dims;
  expr_dims.reserve(dims.size());
  for (int size : dims) {
    expr_dims.emplace_back(ExprHandle(size));
  }
  return expr_dims;
}

}  // namespace

XlaOp::XlaOp(std::shared_ptr<Placeholder> parameter,
             std::unique_ptr<Shape> shape, int64 parameter_number,
             XlaBuilder* builder)
    : outputs_{{nullptr, std::move(parameter), parameter_number}},
      builder_(builder) {
  id_ = builder_->AddParameter(*this, std::move(shape));
}

XlaOp::XlaOp(Tensor* op, std::unique_ptr<Shape> shape, XlaBuilder* builder)
    : outputs_{{op, nullptr}}, builder_(builder) {
  id_ = builder_->AddOperator(*this, std::move(shape));
}

XlaOp::XlaOp(absl::Span<const XlaOp> ops, XlaBuilder* builder)
    : builder_(builder) {
  outputs_.reserve(ops.size());
  for (auto op : ops) {
    const auto& outputs = op.outputs();
    XLA_CHECK_EQ(outputs.size(), size_t(1));
    outputs_.push_back(outputs.front());
  }
  id_ = builder_->AddTuple(*this, ops);
}

XlaBuilder* XlaOp::builder() const { return builder_; }

std::string XlaOp::ToString() const {
  std::ostringstream oss;
  XLA_CHECK(!outputs_.empty());
  const auto& output = outputs_[0];
  if (output.arg) {
    oss << *output.arg->data();
  } else {
    XLA_CHECK(output.expr);
    oss << *output.expr;
  }
  return oss.str();
}

XlaOp XlaOp::Add(XlaOp lhs, XlaOp rhs,
                 absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(
      lhs, rhs,
      [&](const ExprHandle& lhs, const ExprHandle& rhs) { return lhs + rhs; },
      "add", broadcast_dimensions);
}

XlaOp XlaOp::Sub(XlaOp lhs, XlaOp rhs,
                 absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(
      lhs, rhs,
      [&](const ExprHandle& lhs, const ExprHandle& rhs) { return lhs - rhs; },
      "sub", broadcast_dimensions);
}

XlaOp XlaOp::Mul(XlaOp lhs, XlaOp rhs,
                 absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(
      lhs, rhs,
      [&](const ExprHandle& lhs, const ExprHandle& rhs) { return lhs * rhs; },
      "mul", broadcast_dimensions);
}

XlaOp XlaOp::Div(XlaOp lhs, XlaOp rhs,
                 absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(
      lhs, rhs,
      [&](const ExprHandle& lhs, const ExprHandle& rhs) { return lhs / rhs; },
      "div", broadcast_dimensions);
}

XlaOp XlaOp::Reshape(XlaOp operand, absl::Span<const int64> new_sizes) {
  const auto& source_shape =
      torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(operand);
  auto new_shape =
      std::make_unique<Shape>(source_shape.element_type(), new_sizes);
  if (operand.outputs_.size() == 1 && operand.outputs_.front().expr) {
    // Fast path, nothing to do except to associate with new shape.
    return XlaOp(operand.outputs_.front().expr, std::move(new_shape),
                 operand.builder());
  }
  return XlaOp(Compute("reshape", LinearizedSize(new_sizes),
                       [&](const VarHandle& index) {
                         std::vector<ExprHandle> linear_index;
                         linear_index.push_back(index);
                         return operand.call(linear_index);
                       }),
               std::move(new_shape), operand.builder());
}

ExprHandle XlaOp::call(const std::vector<ExprHandle>& indices) const {
  XLA_CHECK_EQ(outputs_.size(), size_t(1));
  const auto& output = outputs_.front();
  if (output.arg) {
    return ::xla::call(output.arg.get(), indices);
  }
  XLA_CHECK(output.expr);
  return ::xla::call(output.expr, indices);
}

std::vector<DimArg> XlaOp::dims() const {
  XLA_CHECK_EQ(outputs_.size(), size_t(1));
  const auto& output = outputs_.front();
  if (output.arg) {
    return ::xla::dims(output.arg.get());
  }
  XLA_CHECK(output.expr);
  return ::xla::dims(output.expr);
}

XlaBuilder::XlaBuilder(const std::string& computation_name)
    : kernel_arena_(std::make_shared<KernelArena>()),
      kernel_scope_(std::make_shared<KernelScope>(kernel_arena_.get())) {}

XlaBuilder::~XlaBuilder() {}

StatusOr<XlaComputation> XlaBuilder::Build(XlaOp root,
                                           bool remove_dynamic_dimensions) {
  return XlaComputation(root, this);
}

Status XlaBuilder::first_error() const { return Status::OK(); }

StatusOr<const Shape*> XlaBuilder::GetShapePtr(XlaOp op) const {
  XLA_CHECK_GE(op.id(), 0);
  XLA_CHECK_LT(static_cast<size_t>(op.id()), shapes_.size());
  return shapes_[op.id()].get();
}

void XlaBuilder::SetUpAlias(const ShapeIndex& output_index, int64 param_number,
                            const ShapeIndex& param_index) {
  XLA_CHECK_EQ(output_index.size(), 1);
  const auto it_ok =
      output_to_input_aliases_.emplace(output_index[0], param_number);
  XLA_CHECK(it_ok.second);
}

const std::unordered_map<size_t, size_t>& XlaBuilder::GetOutputToInputAliases()
    const {
  return output_to_input_aliases_;
}

size_t XlaBuilder::AddParameter(XlaOp op, std::unique_ptr<Shape> shape) {
  size_t id = shapes_.size();
  shapes_.push_back(std::move(shape));
  op.set_id(id);
  parameters_.push_back(op);
  return id;
}

size_t XlaBuilder::AddOperator(XlaOp op, std::unique_ptr<Shape> shape) {
  size_t id = shapes_.size();
  shapes_.push_back(std::move(shape));
  op.set_id(id);
  operators_.push_back(op);
  return id;
}

size_t XlaBuilder::AddTuple(XlaOp tuple, absl::Span<const XlaOp> elements) {
  size_t id = shapes_.size();
  std::vector<Shape> element_shapes;
  for (const XlaOp& element : elements) {
    element_shapes.push_back(
        torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(element));
  }
  shapes_.emplace_back(new Shape(absl::MakeSpan(element_shapes)));
  tuple.set_id(id);
  operators_.push_back(tuple);
  return id;
}

const std::vector<XlaOp>& XlaBuilder::GetParameters() const {
  return parameters_;
}

const std::vector<XlaOp>& XlaBuilder::GetOperators() const {
  return operators_;
}

std::shared_ptr<KernelArena> XlaBuilder::kernel_arena() const {
  return kernel_arena_;
}

XlaOp Parameter(XlaBuilder* builder, int64 parameter_number, const Shape& shape,
                const std::string& name) {
  int numel = lazy_tensors::util::Multiply<int>(shape.dimensions());
  std::vector<ExprHandle> sizes{ExprHandle(numel)};
  Dtype param_type(PrimitiveToScalarType(shape.element_type()));
  return XlaOp(
      /*parameter=*/std::make_shared<Placeholder>(
          BufHandle(name, sizes, param_type)),
      /*shape=*/std::make_unique<Shape>(shape),
      /*parameter_number=*/parameter_number,
      /*builder=*/builder);
}

XlaOp ConstantLiteral(XlaBuilder* builder, const LiteralSlice& literal) {
  const auto& shape = literal.literal()->shape();
  XLA_CHECK(shape.dimensions().empty()) << "Only scalar literals supported";
  std::vector<DimArg> dimensions;
  dimensions.emplace_back(ExprHandle(1));
  ExprHandle handle;
  switch (shape.element_type()) {
    case PrimitiveType::PRED: {
      handle = BoolImm::make(literal.literal()->data<bool>()[0]);
      break;
    }
    case PrimitiveType::S8: {
      handle = CharImm::make(literal.literal()->data<int8_t>()[0]);
      break;
    }
    case PrimitiveType::S16: {
      handle = ShortImm::make(literal.literal()->data<int16_t>()[0]);
      break;
    }
    case PrimitiveType::S32: {
      handle = IntImm::make(literal.literal()->data<int32_t>()[0]);
      break;
    }
    case PrimitiveType::S64: {
      handle = LongImm::make(literal.literal()->data<int64_t>()[0]);
      break;
    }
    case PrimitiveType::U8: {
      handle = ByteImm::make(literal.literal()->data<uint8_t>()[0]);
      break;
    }
    case PrimitiveType::F32: {
      handle = FloatImm::make(literal.literal()->data<float>()[0]);
      break;
    }
    case PrimitiveType::F64: {
      handle = DoubleImm::make(literal.literal()->data<double>()[0]);
      break;
    }
    default: {
      TF_LOG(FATAL) << "Not implemented yet: " << shape.element_type();
    }
  }
  return XlaOp(
      Compute("constant_literal", dimensions,
              [&](const std::vector<VarHandle>& indices) { return handle; }),
      std::make_unique<Shape>(shape), builder);
}

XlaOp Broadcast(XlaOp operand, absl::Span<const int64> broadcast_sizes) {
  std::vector<int64> output_sizes(broadcast_sizes.begin(),
                                  broadcast_sizes.end());
  const auto& operand_shape =
      torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(operand);
  auto operand_sizes = operand_shape.dimensions();
  output_sizes.insert(output_sizes.end(), operand_sizes.begin(),
                      operand_sizes.end());
  return XlaOp(
      Compute("broadcast", operand, output_sizes,
              [&](const std::vector<ExprHandle>& indices) {
                std::vector<ExprHandle> adjusted_indices(
                    indices.begin() + broadcast_sizes.size(), indices.end());
                // If the operand was a scalar, it's a 1D buffer with one
                // element.
                if (adjusted_indices.empty()) {
                  adjusted_indices.emplace_back(0);
                }
                return adjusted_indices;
              }),
      std::make_unique<Shape>(operand_shape.element_type(), output_sizes),
      operand.builder());
}

XlaOp BroadcastInDim(XlaOp operand, const absl::Span<const int64> out_dim_size,
                     const absl::Span<const int64> broadcast_dimensions) {
  const auto& input_shape =
      torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(operand);
  XLA_CHECK_EQ(broadcast_dimensions.size(), input_shape.rank());
  XLA_CHECK_EQ(input_shape.rank(), out_dim_size.size());
  return XlaOp(
      Compute("broadcast_in_dim", operand, out_dim_size,
              [&](const std::vector<ExprHandle>& indices) {
                std::vector<ExprHandle> adjusted_indices;
                for (size_t i = 0; i < broadcast_dimensions.size(); ++i) {
                  if (input_shape.dimensions(i) ==
                      out_dim_size[broadcast_dimensions[i]]) {
                    adjusted_indices.push_back(indices[i]);
                  } else {
                    adjusted_indices.push_back(ExprHandle(0));
                  }
                }
                return adjusted_indices;
              }),
      std::make_unique<Shape>(input_shape.element_type(), out_dim_size),
      operand.builder());
}

XlaOp Pad(XlaOp operand, XlaOp padding_value,
          const PaddingConfig& padding_config) {
  const auto& shape =
      torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(operand);
  const auto& padding_dimensions = padding_config.dimensions();
  XLA_CHECK_EQ(padding_dimensions.size(), shape.rank());
  std::vector<int64> output_sizes;
  output_sizes.reserve(shape.rank());
  for (size_t dim_idx = 0; dim_idx < shape.rank(); ++dim_idx) {
    const auto& padding_dimension = padding_dimensions[dim_idx];
    int interior_padding = padding_dimension->interior_padding();
    output_sizes.push_back(
        padding_dimensions[dim_idx]->edge_padding_low() +
        (shape.dimensions(dim_idx) - 1) * (interior_padding + 1) + 1 +
        padding_dimensions[dim_idx]->edge_padding_high());
  }
  XLA_CHECK_EQ(
      torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(padding_value)
          .rank(),
      0);
  return XlaOp(
      Compute("pad", LinearizedSize(output_sizes),
              [&](const VarHandle& index) {
                const auto indices = LinearToMultiDimIndex(
                    /*index=*/index, /*sizes=*/output_sizes);
                XLA_CHECK_EQ(padding_dimensions.size(), indices.size());
                std::vector<ExprHandle> expr_indices;
                ExprHandle is_padding(0);
                for (size_t i = 0; i < indices.size(); ++i) {
                  const auto& padding_dimension = padding_dimensions[i];
                  ExprHandle low_padding_end(
                      static_cast<int>(padding_dimension->edge_padding_low()));
                  int interior_padding = padding_dimension->interior_padding();
                  expr_indices.push_back((indices[i] - low_padding_end) /
                                         ExprHandle(interior_padding + 1));
                  is_padding =
                      is_padding | IndexIsInsidePadding(
                                       /*padding_dimension=*/*padding_dimension,
                                       /*index=*/indices[i],
                                       /*dim_size=*/shape.dimensions(i));
                }
                is_padding = cast<bool>(is_padding);
                return ifThenElse(
                    is_padding, padding_value.call({ExprHandle(0)}),
                    operand.call(MultiDimToLinearIndex(
                        /*indices=*/expr_indices,
                        /*dims=*/DimsAsExprHandles(shape.dimensions()))));
              }),
      std::make_unique<Shape>(shape.element_type(), output_sizes),
      operand.builder());
}

XlaOp Reshape(XlaOp operand, absl::Span<const int64> new_sizes) {
  return XlaOp::Reshape(operand, new_sizes);
}

XlaOp SliceInDim(XlaOp operand, int64 start_index, int64 limit_index,
                 int64 stride, int64 dimno) {
  XLA_CHECK_EQ(stride, 1) << "Only stride 1 supported for now";
  const auto& shape =
      torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(operand);
  std::vector<int64> output_sizes;
  output_sizes.reserve(shape.rank());
  for (size_t dim_idx = 0; dim_idx < shape.rank(); ++dim_idx) {
    if (dim_idx == dimno) {
      output_sizes.push_back(limit_index - start_index);
    } else {
      output_sizes.push_back(shape.dimensions(dim_idx));
    }
  }
  return XlaOp(
      Compute("slice_in_dim", operand, output_sizes,
              [&](const std::vector<ExprHandle>& indices) {
                std::vector<ExprHandle> expr_indices;
                for (size_t dim_idx = 0; dim_idx < shape.rank(); ++dim_idx) {
                  auto expr_index = static_cast<ExprHandle>(indices[dim_idx]);
                  if (dim_idx == dimno) {
                    expr_indices.push_back(
                        expr_index + ExprHandle(static_cast<int>(start_index)));
                  } else {
                    expr_indices.push_back(expr_index);
                  }
                }
                return expr_indices;
              }),
      std::make_unique<Shape>(shape.element_type(), output_sizes),
      operand.builder());
}

XlaOp DynamicUpdateSlice(XlaOp operand, XlaOp update,
                         absl::Span<const XlaOp> start_indices) {
  const auto& operand_shape =
      torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(operand);
  const auto output_sizes = operand_shape.dimensions();
  const auto update_shape =
      torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(update);
  return XlaOp(
      Compute("dynamic_update_slice", LinearizedSize(output_sizes),
              [&](const VarHandle& index) {
                const auto indices = LinearToMultiDimIndex(
                    /*index=*/index, /*sizes=*/output_sizes);
                ExprHandle is_update(1);
                std::vector<ExprHandle> update_indices;
                update_indices.reserve(update_shape.rank());
                for (size_t i = 0; i < indices.size(); ++i) {
                  XLA_CHECK_EQ(
                      torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(
                          start_indices[i])
                          .rank(),
                      0);
                  const auto start_index =
                      cast<int>(start_indices[i].call({ExprHandle(0)}));
                  is_update = is_update & cast<int>(indices[i] >= start_index);
                  const auto end_index =
                      start_index +
                      ExprHandle(static_cast<int>(update_shape.dimensions(i)));
                  is_update = is_update & cast<int>(indices[i] < end_index);
                  update_indices.push_back(indices[i] - start_index);
                }
                return ifThenElse(
                    is_update,
                    update.call(MultiDimToLinearIndex(
                        /*indices=*/update_indices,
                        /*dims=*/DimsAsExprHandles(update_shape.dimensions()))),
                    operand.call({index}));
              }),
      std::make_unique<Shape>(operand_shape), operand.builder());
}

XlaOp Slice(XlaOp operand, absl::Span<const int64> start_indices,
            absl::Span<const int64> limit_indices,
            absl::Span<const int64> strides) {
  auto result = operand;
  for (size_t dimno = 0; dimno < limit_indices.size(); ++dimno) {
    result = SliceInDim(result, start_indices[dimno], limit_indices[dimno],
                        strides[dimno], dimno);
  }
  return result;
}

XlaOp Select(XlaOp pred, XlaOp on_true, XlaOp on_false) {
  std::tie(on_true, on_false) =
      torch_lazy_tensors::compiler::XlaHelpers::PromoteShapes(on_true,
                                                              on_false);
  const auto& on_true_shape =
      torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(on_true);
  const auto out_dims = on_true.dims();
  return XlaOp(Compute("select", out_dims,
                       [&](const std::vector<VarHandle>& indices) {
                         const auto expr_indices = VarsToExprs(indices);
                         return ifThenElse(pred.call(expr_indices),
                                           on_true.call(expr_indices),
                                           on_false.call(expr_indices));
                       }),
               std::make_unique<Shape>(on_true_shape.element_type(),
                                       on_true_shape.dimensions()),
               on_true.builder());
}

XlaOp Tuple(XlaBuilder* builder, absl::Span<const XlaOp> elements) {
  return XlaOp(elements, builder);
}

XlaOp Eq(XlaOp lhs, XlaOp rhs, absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(lhs, rhs,
                  [&](const ExprHandle& lhs, const ExprHandle& rhs) {
                    return cast<bool>(lhs == rhs);
                  },
                  "eq", broadcast_dimensions, PrimitiveType::PRED);
}

XlaOp Ne(XlaOp lhs, XlaOp rhs, absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(lhs, rhs,
                  [&](const ExprHandle& lhs, const ExprHandle& rhs) {
                    return cast<bool>(lhs != rhs);
                  },
                  "ne", broadcast_dimensions, PrimitiveType::PRED);
}

XlaOp Ge(XlaOp lhs, XlaOp rhs, absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(lhs, rhs,
                  [&](const ExprHandle& lhs, const ExprHandle& rhs) {
                    return cast<bool>(lhs >= rhs);
                  },
                  "ge", broadcast_dimensions, PrimitiveType::PRED);
}

XlaOp Gt(XlaOp lhs, XlaOp rhs, absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(lhs, rhs,
                  [&](const ExprHandle& lhs, const ExprHandle& rhs) {
                    return cast<bool>(lhs > rhs);
                  },
                  "gt", broadcast_dimensions, PrimitiveType::PRED);
}

XlaOp Lt(XlaOp lhs, XlaOp rhs, absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(lhs, rhs,
                  [&](const ExprHandle& lhs, const ExprHandle& rhs) {
                    return cast<bool>(lhs < rhs);
                  },
                  "lt", broadcast_dimensions, PrimitiveType::PRED);
}

XlaOp Le(XlaOp lhs, XlaOp rhs, absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(lhs, rhs,
                  [&](const ExprHandle& lhs, const ExprHandle& rhs) {
                    return cast<bool>(lhs <= rhs);
                  },
                  "le", broadcast_dimensions, PrimitiveType::PRED);
}

XlaOp Div(XlaOp lhs, XlaOp rhs, absl::Span<const int64> broadcast_dimensions) {
  return XlaOp::Div(lhs, rhs, broadcast_dimensions);
}

XlaOp Rem(XlaOp lhs, XlaOp rhs, absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(lhs, rhs,
                  [&](const ExprHandle& lhs, const ExprHandle& rhs) {
                    XLA_CHECK_EQ(lhs.dtype(), rhs.dtype());
                    if (lhs.dtype().is_integral()) {
                      return lhs % rhs;
                    }
                    return fmod(lhs, rhs);
                  },
                  "rem", broadcast_dimensions);
}

XlaOp Max(XlaOp lhs, XlaOp rhs, absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(lhs, rhs,
                  [&](const ExprHandle& lhs, const ExprHandle& rhs) {
                    return Max::make(lhs, rhs,
                                     /*propagate_nans=*/true);
                  },
                  "max", broadcast_dimensions);
}

XlaOp Min(XlaOp lhs, XlaOp rhs, absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(lhs, rhs,
                  [&](const ExprHandle& lhs, const ExprHandle& rhs) {
                    return Min::make(lhs, rhs,
                                     /*propagate_nans=*/true);
                  },
                  "min", broadcast_dimensions);
}

XlaOp And(XlaOp lhs, XlaOp rhs, absl::Span<const int64> broadcast_dimensions) {
  return lhs & rhs;
}

XlaOp Or(XlaOp lhs, XlaOp rhs, absl::Span<const int64> broadcast_dimensions) {
  return lhs | rhs;
}

XlaOp Abs(XlaOp operand) {
  return UnaryOp(
      operand, [&](const ExprHandle& operand) { return abs(operand); }, "abs");
}

XlaOp Atan2(XlaOp y, XlaOp x, absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(
      y, x,
      [&](const ExprHandle& y, const ExprHandle& x) { return atan2(y, x); },
      "atan2", broadcast_dimensions);
}

XlaOp Exp(XlaOp operand) {
  return UnaryOp(
      operand, [&](const ExprHandle& operand) { return exp(operand); }, "exp");
}

XlaOp Expm1(XlaOp operand) {
  return UnaryOp(operand,
                 [&](const ExprHandle& operand) { return expm1(operand); },
                 "expm1");
}

XlaOp Floor(XlaOp operand) {
  return UnaryOp(operand,
                 [&](const ExprHandle& operand) { return floor(operand); },
                 "floor");
}

XlaOp Ceil(XlaOp operand) {
  return UnaryOp(operand,
                 [&](const ExprHandle& operand) { return ceil(operand); },
                 "ceil");
}

XlaOp Log(XlaOp operand) {
  return UnaryOp(
      operand, [&](const ExprHandle& operand) { return log(operand); }, "log");
}

XlaOp Log1p(XlaOp operand) {
  return UnaryOp(operand,
                 [&](const ExprHandle& operand) { return log1p(operand); },
                 "log1p");
}

XlaOp Sign(XlaOp operand) {
  const auto& operand_shape =
      torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(operand);
  const auto zero =
      Broadcast(Zero(operand.builder(), operand_shape.element_type()),
                operand_shape.dimensions());
  const auto one =
      Broadcast(One(operand.builder(), operand_shape.element_type()),
                operand_shape.dimensions());
  const auto minus_one = Neg(one);
  return Select(Lt(operand, zero), minus_one,
                Select(Gt(operand, zero), one, zero));
}

XlaOp Cos(XlaOp operand) {
  return UnaryOp(
      operand, [&](const ExprHandle& operand) { return cos(operand); }, "cos");
}

XlaOp Sin(XlaOp operand) {
  return UnaryOp(
      operand, [&](const ExprHandle& operand) { return sin(operand); }, "sin");
}

XlaOp Tanh(XlaOp operand) {
  return UnaryOp(operand,
                 [&](const ExprHandle& operand) { return tanh(operand); },
                 "tanh");
}

XlaOp Sqrt(XlaOp operand) {
  return UnaryOp(operand,
                 [&](const ExprHandle& operand) { return sqrt(operand); },
                 "sqrt");
}

XlaOp Rsqrt(XlaOp operand) {
  return UnaryOp(operand,
                 [&](const ExprHandle& operand) { return rsqrt(operand); },
                 "rsqrt");
}

XlaOp Pow(XlaOp lhs, XlaOp rhs, absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(lhs, rhs,
                  [&](const ExprHandle& lhs, const ExprHandle& rhs) {
                    return pow(lhs, rhs);
                  },
                  "pow", broadcast_dimensions);
}

XlaOp ConvertElementType(XlaOp operand, PrimitiveType new_element_type) {
  const auto& operand_shape =
      torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(operand);
  return UnaryOp(
      operand,
      [&](const ExprHandle& operand) {
        auto dtype = ToDtype(PrimitiveToScalarType(new_element_type));
        return Cast::make(dtype, operand);
      },
      "cast",
      std::make_unique<Shape>(new_element_type, operand_shape.dimensions()));
}

XlaOp Neg(XlaOp operand) {
  const auto& operand_shape =
      torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(operand);
  return Zero(operand.builder(), operand_shape.element_type()) - operand;
}

XlaOp Transpose(XlaOp operand, absl::Span<const int64> permutation) {
  const auto& operand_shape =
      torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(operand);
  const auto operand_sizes = operand_shape.dimensions();
  std::vector<int64> output_sizes;
  output_sizes.reserve(operand_sizes.size());
  for (size_t i = 0; i < operand_sizes.size(); ++i) {
    output_sizes.push_back(operand_sizes[permutation[i]]);
  }
  return XlaOp(
      Compute("transpose", operand, output_sizes,
              [&](const std::vector<ExprHandle>& indices) {
                // PermuteInverse applies the permutation to the output, which
                // makes it the inverse of the provided permutation. This is
                // precisely what we need to access the input.
                return PermuteInverse(indices, permutation);
              }),
      std::make_unique<Shape>(operand_shape.element_type(), output_sizes),
      operand.builder());
}

XlaOp Clamp(XlaOp min, XlaOp operand, XlaOp max) {
  const auto out_dims = operand.dims();
  const auto& operand_shape =
      torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(operand);
  return XlaOp(Compute("clamp", out_dims,
                       [&](const std::vector<VarHandle>& indices) {
                         const auto expr_indices = VarsToExprs(indices);
                         const auto operand_expr = operand.call(expr_indices);
                         std::vector<ExprHandle> zero_index;
                         zero_index.emplace_back(0);
                         const auto min_expr = min.call(zero_index);
                         const auto max_expr = max.call(zero_index);
                         return CompareSelect::make(
                             operand_expr, min_expr, min_expr,
                             CompareSelect::make(operand_expr, max_expr,
                                                 max_expr, operand_expr, kGT),
                             kLT);
                       }),
               std::make_unique<Shape>(operand_shape), operand.builder());
}

XlaOp operator+(XlaOp x, XlaOp y) { return XlaOp::Add(x, y); }

XlaOp operator-(XlaOp x, XlaOp y) { return XlaOp::Sub(x, y); }

XlaOp operator*(XlaOp x, XlaOp y) { return XlaOp::Mul(x, y); }

XlaOp operator/(XlaOp x, XlaOp y) { return XlaOp::Div(x, y); }

XlaOp operator&(XlaOp x, XlaOp y) {
  PrimitiveType original_type;
  std::tie(x, y, original_type) = PromoteToInteger(x, y);
  return ConvertElementType(
      BinaryOp(x, y,
               [&](const ExprHandle& lhs, const ExprHandle& rhs) {
                 return lhs & rhs;
               },
               "bitwise_and", {}),
      original_type);
}

XlaOp operator|(XlaOp x, XlaOp y) {
  PrimitiveType original_type;
  std::tie(x, y, original_type) = PromoteToInteger(x, y);
  return ConvertElementType(
      BinaryOp(x, y,
               [&](const ExprHandle& lhs, const ExprHandle& rhs) {
                 return lhs | rhs;
               },
               "bitwise_or", {}),
      original_type);
}

XlaOp operator^(XlaOp x, XlaOp y) {
  PrimitiveType original_type;
  std::tie(x, y, original_type) = PromoteToInteger(x, y);
  return ConvertElementType(
      BinaryOp(x, y,
               [&](const ExprHandle& lhs, const ExprHandle& rhs) {
                 return lhs ^ rhs;
               },
               "bitwise_xor", {}),
      original_type);
}

XlaOp UnaryOp(XlaOp input,
              const std::function<ExprHandle(const ExprHandle&)>& unary_op,
              const std::string& name) {
  const auto& input_shape =
      torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(input);
  return UnaryOp(input, unary_op, name, std::make_unique<Shape>(input_shape));
}

Tensor* Compute(const std::string& func_name, XlaOp operand,
                absl::Span<const int64> output_sizes,
                const std::function<std::vector<ExprHandle>(
                    const std::vector<ExprHandle>&)>& to_input_indices) {
  const auto operand_shape =
      torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(operand);
  return Compute(func_name, LinearizedSize(/*sizes=*/output_sizes),
                 [&](const VarHandle& index) {
                   auto output_indices = LinearToMultiDimIndex(
                       /*index=*/index, /*sizes=*/output_sizes);
                   return operand.call(MultiDimToLinearIndex(
                       /*indices=*/to_input_indices(output_indices),
                       /*dims=*/DimsAsExprHandles(operand_shape.dimensions())));
                 });
}

}  // namespace xla
