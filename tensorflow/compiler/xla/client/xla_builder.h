#pragma once

#include <string>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "torch/csrc/jit/tensorexpr/loopnest.h"
#include "torch/csrc/jit/tensorexpr/mem_arena.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

namespace xla {

class XlaOp {
 public:
  XlaOp() : builder_(nullptr) {}

  XlaOp(std::shared_ptr<torch::jit::tensorexpr::Placeholder> parameter,
        std::unique_ptr<Shape> shape, int64 parameter_number,
        XlaBuilder* builder);

  XlaOp(torch::jit::tensorexpr::Tensor* op, std::unique_ptr<Shape> shape,
        XlaBuilder* builder);

  XlaOp(absl::Span<const XlaOp> ops, XlaBuilder* builder);

  XlaBuilder* builder() const;

  bool valid() const { return id_.has_value(); }

  int id() const {
    LTC_CHECK(id_);
    return *id_;
  }

  void set_id(int id) {
    LTC_CHECK(!id_);
    id_ = id;
  }

  bool IsUninitialized() const { LTC_LOG(FATAL) << "Not implemented yet."; }

  std::string ToString() const;

  static XlaOp Add(XlaOp lhs, XlaOp rhs,
                   absl::Span<const int64> broadcast_dimensions = {});

  static XlaOp Sub(XlaOp lhs, XlaOp rhs,
                   absl::Span<const int64> broadcast_dimensions = {});

  static XlaOp Mul(XlaOp lhs, XlaOp rhs,
                   absl::Span<const int64> broadcast_dimensions = {});

  static XlaOp Div(XlaOp lhs, XlaOp rhs,
                   absl::Span<const int64> broadcast_dimensions = {});

  static XlaOp Reshape(XlaOp operand, absl::Span<const int64> new_sizes);

  struct Handle {
    torch::jit::tensorexpr::Tensor* expr;
    std::shared_ptr<torch::jit::tensorexpr::Placeholder> arg;
    absl::optional<size_t> arg_idx;
  };

  const std::vector<Handle>& outputs() const { return outputs_; }

  torch::jit::tensorexpr::ExprHandle call(
      const std::vector<torch::jit::tensorexpr::ExprHandle>& indices) const;

  std::vector<torch::jit::tensorexpr::DimArg> dims() const;

 private:
  std::vector<Handle> outputs_;
  XlaBuilder* builder_;
  absl::optional<int> id_;
};

class XlaBuilder {
 public:
  XlaBuilder(const std::string& computation_name);

  virtual ~XlaBuilder();

  void SetOpMetadata(OpMetadata metadata) {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  void ClearOpMetadata() { LTC_LOG(FATAL) << "Not implemented yet."; }

  static ConvolutionDimensionNumbers CreateDefaultConvDimensionNumbers(
      int num_spatial_dims = 2) {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  StatusOr<XlaComputation> Build(bool remove_dynamic_dimensions = false) {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  StatusOr<XlaComputation> Build(XlaOp root,
                                 bool remove_dynamic_dimensions = false);

  Status first_error() const;

  virtual StatusOr<const Shape*> GetShapePtr(XlaOp op) const;

  Status GetCurrentStatus() const { LTC_LOG(FATAL) << "Not implemented yet."; }

  XlaOp ReportErrorOrReturn(
      const std::function<StatusOr<XlaOp>()>& op_creator) {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  void SetUpAlias(const ShapeIndex& output_index, int64 param_number,
                  const ShapeIndex& param_index);

  const std::unordered_map<size_t, size_t>& GetOutputToInputAliases() const;

  size_t AddParameter(XlaOp op, std::unique_ptr<Shape> shape);

  size_t AddOperator(XlaOp op, std::unique_ptr<Shape> shape);

  size_t AddTuple(XlaOp tuple, absl::Span<const XlaOp> elements);

  const std::vector<XlaOp>& GetParameters() const;

  const std::vector<XlaOp>& GetOperators() const;

  std::shared_ptr<torch::jit::tensorexpr::KernelArena> kernel_arena() const;

 private:
  std::shared_ptr<torch::jit::tensorexpr::KernelArena> kernel_arena_;
  std::shared_ptr<torch::jit::tensorexpr::KernelScope> kernel_scope_;
  std::vector<XlaOp> parameters_;
  std::vector<XlaOp> operators_;
  std::vector<std::unique_ptr<Shape>> shapes_;
  std::unordered_map<size_t, size_t> output_to_input_aliases_;
};

XlaOp Parameter(XlaBuilder* builder, int64 parameter_number, const Shape& shape,
                const std::string& name);

XlaOp ConstantLiteral(XlaBuilder* builder, const LiteralSlice& literal);

template <typename NativeT>
inline XlaOp ConstantR0(XlaBuilder* builder, NativeT value) {
  return ConstantLiteral(builder, LiteralUtil::CreateR0<NativeT>(value));
}

XlaOp Broadcast(XlaOp operand, absl::Span<const int64> broadcast_sizes);

XlaOp BroadcastInDim(XlaOp operand, const absl::Span<const int64> out_dim_size,
                     const absl::Span<const int64> broadcast_dimensions);

XlaOp Pad(XlaOp operand, XlaOp padding_value,
          const PaddingConfig& padding_config);

XlaOp Reshape(XlaOp operand, absl::Span<const int64> new_sizes);

inline XlaOp ReshapeWithInferredDimension(XlaOp operand,
                                          absl::Span<const int64> new_sizes,
                                          int64 inferred_dimension) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

XlaOp Slice(XlaOp operand, absl::Span<const int64> start_indices,
            absl::Span<const int64> limit_indices,
            absl::Span<const int64> strides);

XlaOp SliceInDim(XlaOp operand, int64 start_index, int64 limit_index,
                 int64 stride, int64 dimno);

inline XlaOp DynamicSlice(XlaOp operand, absl::Span<const XlaOp> start_indices,
                          absl::Span<const int64> slice_sizes) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

XlaOp DynamicUpdateSlice(XlaOp operand, XlaOp update,
                         absl::Span<const XlaOp> start_indices);

inline XlaOp ConcatInDim(XlaBuilder* builder, absl::Span<const XlaOp> operands,
                         int64 dimension) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

XlaOp Select(XlaOp pred, XlaOp on_true, XlaOp on_false);

XlaOp Tuple(XlaBuilder* builder, absl::Span<const XlaOp> elements);

inline XlaOp GetTupleElement(XlaOp tuple_data, int64 index) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

XlaOp Eq(XlaOp lhs, XlaOp rhs,
         absl::Span<const int64> broadcast_dimensions = {});

XlaOp Ne(XlaOp lhs, XlaOp rhs,
         absl::Span<const int64> broadcast_dimensions = {});

XlaOp Ge(XlaOp lhs, XlaOp rhs,
         absl::Span<const int64> broadcast_dimensions = {});

XlaOp Gt(XlaOp lhs, XlaOp rhs,
         absl::Span<const int64> broadcast_dimensions = {});

XlaOp Lt(XlaOp lhs, XlaOp rhs,
         absl::Span<const int64> broadcast_dimensions = {});

XlaOp Le(XlaOp lhs, XlaOp rhs,
         absl::Span<const int64> broadcast_dimensions = {});

inline XlaOp Dot(XlaOp lhs, XlaOp rhs,
                 const PrecisionConfig* precision_config = nullptr) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp DotGeneral(XlaOp lhs, XlaOp rhs,
                        const DotDimensionNumbers& dimension_numbers,
                        const PrecisionConfig* precision_config = nullptr) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp ConvGeneralDilated(
    XlaOp lhs, XlaOp rhs, absl::Span<const int64> window_strides,
    absl::Span<const std::pair<int64, int64>> padding,
    absl::Span<const int64> lhs_dilation, absl::Span<const int64> rhs_dilation,
    const ConvolutionDimensionNumbers& dimension_numbers,
    int64 feature_group_count = 1, int64 batch_group_count = 1,
    const PrecisionConfig* precision_config = nullptr) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp TriangularSolve(XlaOp a, XlaOp b, bool left_side, bool lower,
                             bool unit_diagonal,
                             TriangularSolveOptions::Transpose transpose_a) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp Cholesky(XlaOp a, bool lower) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp Call(XlaBuilder* builder, const XlaComputation& computation,
                  absl::Span<const XlaOp> operands) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp CustomCall(XlaBuilder* builder,
                        const std::string& call_target_name,
                        absl::Span<const XlaOp> operands, const Shape& shape,
                        const std::string& opaque = "") {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp Complex(XlaOp real, XlaOp imag,
                     absl::Span<const int64> broadcast_dimensions = {}) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp Conj(XlaOp operand) { LTC_LOG(FATAL) << "Not implemented yet."; }

inline XlaOp Add(XlaOp lhs, XlaOp rhs,
                 absl::Span<const int64> broadcast_dimensions = {}) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp Sub(XlaOp lhs, XlaOp rhs,
                 absl::Span<const int64> broadcast_dimensions = {}) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp Mul(XlaOp lhs, XlaOp rhs,
                 absl::Span<const int64> broadcast_dimensions = {}) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

XlaOp Div(XlaOp lhs, XlaOp rhs,
          absl::Span<const int64> broadcast_dimensions = {});

XlaOp Rem(XlaOp lhs, XlaOp rhs,
          absl::Span<const int64> broadcast_dimensions = {});

XlaOp Max(XlaOp lhs, XlaOp rhs,
          absl::Span<const int64> broadcast_dimensions = {});

XlaOp Min(XlaOp lhs, XlaOp rhs,
          absl::Span<const int64> broadcast_dimensions = {});

XlaOp And(XlaOp lhs, XlaOp rhs,
          absl::Span<const int64> broadcast_dimensions = {});

XlaOp Or(XlaOp lhs, XlaOp rhs,
         absl::Span<const int64> broadcast_dimensions = {});

inline XlaOp Xor(XlaOp lhs, XlaOp rhs,
                 absl::Span<const int64> broadcast_dimensions = {}) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp Not(XlaOp operand) { LTC_LOG(FATAL) << "Not implemented yet."; }

inline XlaOp Reduce(XlaOp operand, XlaOp init_value,
                    const XlaComputation& computation,
                    absl::Span<const int64> dimensions_to_reduce) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp ReduceAll(XlaOp operand, XlaOp init_value,
                       const XlaComputation& computation) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp ReduceWindow(XlaOp operand, XlaOp init_value,
                          const XlaComputation& computation,
                          absl::Span<const int64> window_dimensions,
                          absl::Span<const int64> window_strides,
                          Padding padding) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp ReduceWindowWithGeneralPadding(
    XlaOp operand, XlaOp init_value, const XlaComputation& computation,
    absl::Span<const int64> window_dimensions,
    absl::Span<const int64> window_strides,
    absl::Span<const int64> base_dilations,
    absl::Span<const int64> window_dilations,
    absl::Span<const std::pair<int64, int64>> padding) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp AllReduce(
    XlaOp operand, const XlaComputation& computation,
    absl::Span<const ReplicaGroup> replica_groups = {},
    const absl::optional<ChannelHandle>& channel_id = absl::nullopt,
    const absl::optional<Shape>& shape_with_layout = absl::nullopt) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp AllToAll(XlaOp operand, int64 split_dimension,
                      int64 concat_dimension, int64 split_count,
                      const std::vector<ReplicaGroup>& replica_groups = {},
                      const absl::optional<Layout>& layout = absl::nullopt) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp CollectivePermute(
    XlaOp operand,
    const std::vector<std::pair<int64, int64>>& source_target_pairs) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp SelectAndScatter(XlaOp operand, const XlaComputation& select,
                              absl::Span<const int64> window_dimensions,
                              absl::Span<const int64> window_strides,
                              Padding padding, XlaOp source, XlaOp init_value,
                              const XlaComputation& scatter) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp SelectAndScatterWithGeneralPadding(
    XlaOp operand, const XlaComputation& select,
    absl::Span<const int64> window_dimensions,
    absl::Span<const int64> window_strides,
    absl::Span<const std::pair<int64, int64>> padding, XlaOp source,
    XlaOp init_value, const XlaComputation& scatter) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

XlaOp Abs(XlaOp operand);

XlaOp Atan2(XlaOp y, XlaOp x,
            absl::Span<const int64> broadcast_dimensions = {});

XlaOp Exp(XlaOp operand);

XlaOp Expm1(XlaOp operand);

XlaOp Floor(XlaOp operand);

XlaOp Ceil(XlaOp operand);

XlaOp Log(XlaOp operand);

XlaOp Log1p(XlaOp operand);

XlaOp Sign(XlaOp operand);

inline XlaOp Clz(XlaOp operand) { LTC_LOG(FATAL) << "Not implemented yet."; }

XlaOp Cos(XlaOp operand);

XlaOp Sin(XlaOp operand);

XlaOp Tanh(XlaOp operand);

inline XlaOp Real(XlaOp operand) { LTC_LOG(FATAL) << "Not implemented yet."; }

inline XlaOp Imag(XlaOp operand) { LTC_LOG(FATAL) << "Not implemented yet."; }

XlaOp Sqrt(XlaOp operand);

XlaOp Rsqrt(XlaOp operand);

XlaOp Pow(XlaOp lhs, XlaOp rhs,
          absl::Span<const int64> broadcast_dimensions = {});

inline XlaOp IsFinite(XlaOp operand) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp Iota(XlaBuilder* builder, const Shape& shape,
                  int64 iota_dimension) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp Iota(XlaBuilder* builder, PrimitiveType type, int64 size) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

XlaOp ConvertElementType(XlaOp operand, PrimitiveType new_element_type);

XlaOp Neg(XlaOp operand);

XlaOp Transpose(XlaOp operand, absl::Span<const int64> permutation);

inline XlaOp Rev(XlaOp operand, absl::Span<const int64> dimensions) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp Sort(absl::Span<const XlaOp> operands,
                  const XlaComputation& comparator, int64 dimension = -1,
                  bool is_stable = false) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

XlaOp Clamp(XlaOp min, XlaOp operand, XlaOp max);

inline XlaOp RngBitGenerator(RandomAlgorithm algorithm, XlaOp initial_state,
                             const Shape& shape) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp Gather(XlaOp input, XlaOp start_indices,
                    const GatherDimensionNumbers& dimension_numbers,
                    absl::Span<const int64> slice_sizes,
                    bool indices_are_sorted = false) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp Scatter(XlaOp input, XlaOp scatter_indices, XlaOp updates,
                     const XlaComputation& update_computation,
                     const ScatterDimensionNumbers& dimension_numbers,
                     bool indices_are_sorted = false,
                     bool unique_indices = false) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp BatchNormTraining(XlaOp operand, XlaOp scale, XlaOp offset,
                               float epsilon, int64 feature_index) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp BatchNormInference(XlaOp operand, XlaOp scale, XlaOp offset,
                                XlaOp mean, XlaOp variance, float epsilon,
                                int64 feature_index) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp BatchNormGrad(XlaOp operand, XlaOp scale, XlaOp batch_mean,
                           XlaOp batch_var, XlaOp grad_output, float epsilon,
                           int64 feature_index) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp GetDimensionSize(XlaOp operand, int64 dimension) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp SetDimensionSize(XlaOp operand, XlaOp val, int64 dimension) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp operator-(XlaOp x) { LTC_LOG(FATAL) << "Not implemented yet."; }
XlaOp operator+(XlaOp x, XlaOp y);
XlaOp operator-(XlaOp x, XlaOp y);
XlaOp operator*(XlaOp x, XlaOp y);
XlaOp operator/(XlaOp x, XlaOp y);
inline XlaOp operator%(XlaOp x, XlaOp y) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

XlaOp operator&(XlaOp x, XlaOp y);
XlaOp operator|(XlaOp x, XlaOp y);
XlaOp operator^(XlaOp x, XlaOp y);
inline XlaOp operator<<(XlaOp x, XlaOp y) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}
inline XlaOp operator>>(XlaOp x, XlaOp y) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

template <typename NativeT>
using Array2D = std::vector<std::vector<NativeT>>;

template <typename NativeT>
inline XlaOp ConstantR2FromArray2D(XlaBuilder* builder,
                                   const Array2D<NativeT>& values) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

XlaOp UnaryOp(XlaOp input,
              const std::function<torch::jit::tensorexpr::ExprHandle(
                  const torch::jit::tensorexpr::ExprHandle&)>& unary_op,
              const std::string& name);

torch::jit::tensorexpr::Tensor* Compute(
    const std::string& func_name, XlaOp operand,
    absl::Span<const int64> output_sizes,
    const std::function<std::vector<torch::jit::tensorexpr::ExprHandle>(
        const std::vector<torch::jit::tensorexpr::ExprHandle>&)>&
        to_input_indices);

}  // namespace xla
