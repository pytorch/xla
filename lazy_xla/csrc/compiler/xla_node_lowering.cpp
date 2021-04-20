#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/data_ops.h"
#include "lazy_tensor_core/csrc/ops/adaptive_avg_pool2d.h"
#include "lazy_tensor_core/csrc/ops/adaptive_avg_pool3d.h"
#include "lazy_tensor_core/csrc/ops/all.h"
#include "lazy_tensor_core/csrc/ops/amp_foreach_non_finite_check_and_unscale.h"
#include "lazy_tensor_core/csrc/ops/amp_update_scale.h"
#include "lazy_tensor_core/csrc/ops/any.h"
#include "lazy_tensor_core/csrc/ops/arg_max.h"
#include "lazy_tensor_core/csrc/ops/arg_min.h"
#include "lazy_tensor_core/csrc/ops/as_strided.h"
#include "lazy_tensor_core/csrc/ops/as_strided_view_update.h"
#include "lazy_tensor_core/csrc/ops/avg_pool_nd.h"
#include "lazy_tensor_core/csrc/ops/avg_pool_nd_backward.h"
#include "lazy_tensor_core/csrc/ops/binary_cross_entropy.h"
#include "lazy_tensor_core/csrc/ops/binary_cross_entropy_backward.h"
#include "lazy_tensor_core/csrc/ops/bitwise_ir_ops.h"
#include "lazy_tensor_core/csrc/ops/cast.h"
#include "lazy_tensor_core/csrc/ops/constant.h"
#include "lazy_tensor_core/csrc/ops/constant_pad_nd.h"
#include "lazy_tensor_core/csrc/ops/device_data.h"
#include "lazy_tensor_core/csrc/ops/diagonal.h"
#include "lazy_tensor_core/csrc/ops/diagonal_view_update.h"
#include "lazy_tensor_core/csrc/ops/expand.h"
#include "lazy_tensor_core/csrc/ops/generic_slice.h"
#include "lazy_tensor_core/csrc/ops/get_dimensions_size.h"
#include "lazy_tensor_core/csrc/ops/hardshrink.h"
#include "lazy_tensor_core/csrc/ops/hardtanh_backward.h"
#include "lazy_tensor_core/csrc/ops/leaky_relu.h"
#include "lazy_tensor_core/csrc/ops/leaky_relu_backward.h"
#include "lazy_tensor_core/csrc/ops/linear_interpolation.h"
#include "lazy_tensor_core/csrc/ops/log_base.h"
#include "lazy_tensor_core/csrc/ops/log_softmax.h"
#include "lazy_tensor_core/csrc/ops/log_softmax_backward.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/ops/masked_fill.h"
#include "lazy_tensor_core/csrc/ops/not_supported.h"
#include "lazy_tensor_core/csrc/ops/ops.h"
#include "lazy_tensor_core/csrc/ops/permute.h"
#include "lazy_tensor_core/csrc/ops/resize.h"
#include "lazy_tensor_core/csrc/ops/rrelu_with_noise.h"
#include "lazy_tensor_core/csrc/ops/rrelu_with_noise_backward.h"
#include "lazy_tensor_core/csrc/ops/scalar.h"
#include "lazy_tensor_core/csrc/ops/select.h"
#include "lazy_tensor_core/csrc/ops/shrink_backward.h"
#include "lazy_tensor_core/csrc/ops/softmax.h"
#include "lazy_tensor_core/csrc/ops/softmax_backward.h"
#include "lazy_tensor_core/csrc/ops/softshrink.h"
#include "lazy_tensor_core/csrc/ops/split.h"
#include "lazy_tensor_core/csrc/ops/squeeze.h"
#include "lazy_tensor_core/csrc/ops/threshold.h"
#include "lazy_tensor_core/csrc/ops/threshold_backward.h"
#include "lazy_tensor_core/csrc/ops/unselect.h"
#include "lazy_tensor_core/csrc/ops/unsqueeze.h"
#include "lazy_tensor_core/csrc/ops/update_slice.h"
#include "lazy_tensor_core/csrc/ops/upsample_bilinear2d.h"
#include "lazy_tensor_core/csrc/ops/upsample_bilinear2d_backward.h"
#include "lazy_tensor_core/csrc/ops/upsample_nearest2d.h"
#include "lazy_tensor_core/csrc/ops/upsample_nearest2d_backward.h"
#include "lazy_tensor_core/csrc/ops/view.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_xla/csrc/compiler/convert_ops.h"
#include "lazy_xla/csrc/compiler/data_ops.h"
#include "lazy_xla/csrc/compiler/elementwise.h"
#include "lazy_xla/csrc/compiler/helpers.h"
#include "lazy_xla/csrc/compiler/infer_output_shape.h"
#include "lazy_xla/csrc/compiler/matrix.h"
#include "lazy_xla/csrc/compiler/pooling.h"
#include "lazy_xla/csrc/compiler/reduction.h"
#include "lazy_xla/csrc/compiler/resize_ops.h"
#include "lazy_xla/csrc/compiler/softmax_builder.h"
#include "lazy_xla/csrc/compiler/xla_lower_util.h"
#include "lazy_xla/csrc/compiler/xla_lowering_context.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/util.h"

namespace torch_lazy_tensors {
namespace compiler {
namespace {

xla::XlaOp LowerAsStridedViewUpdate(xla::XlaOp target, xla::XlaOp input,
                                    absl::Span<const xla::int64> size,
                                    absl::Span<const xla::int64> stride,
                                    xla::int64 storage_offset) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::int64 input_element_count = xla::ShapeUtil::ElementsIn(input_shape);
  xla::int64 slice_size = lazy_tensors::util::Multiply<xla::int64>(size);
  LTC_CHECK_LE(storage_offset + input_element_count, slice_size);

  std::vector<xla::int64> permutation =
      ir::ops::AsStrided::GetArrayStridePermutation(stride,
                                                    input_shape.dimensions());
  xla::XlaOp transposed_input = xla::IsIdentityPermutation(permutation)
                                    ? input
                                    : xla::Transpose(input, permutation);
  if (storage_offset > 0 || input_element_count < slice_size) {
    xla::XlaOp r1_input = XlaHelpers::Flatten(transposed_input);
    xla::XlaOp r1_target = XlaHelpers::Flatten(target);
    transposed_input = xla::DynamicUpdateSlice(
        r1_target, r1_input,
        {XlaHelpers::ScalarValue<xla::int64>(storage_offset, input.builder())});
  }
  return XlaHelpers::DynamicReshape(transposed_input, size);
}

xla::XlaOp LowerAsStrided(xla::XlaOp input, absl::Span<const xla::int64> size,
                          absl::Span<const xla::int64> stride,
                          xla::int64 storage_offset) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::int64 input_element_count = xla::ShapeUtil::ElementsIn(input_shape);
  xla::int64 slice_size = lazy_tensors::util::Multiply<xla::int64>(size);
  LTC_CHECK_LE(storage_offset + slice_size, input_element_count);

  xla::XlaOp off_input = input;
  if (storage_offset > 0 || slice_size < input_element_count) {
    xla::XlaOp r1_input = XlaHelpers::Flatten(input);
    off_input = xla::SliceInDim(r1_input, storage_offset,
                                storage_offset + slice_size, 1, 0);
  }

  std::vector<xla::int64> permutation = xla::InversePermutation(
      ir::ops::AsStrided::GetArrayStridePermutation(stride, size));
  std::vector<xla::int64> new_sizes = xla::PermuteInverse(size, permutation);
  xla::XlaOp reshaped_input = XlaHelpers::DynamicReshape(off_input, new_sizes);
  return xla::IsIdentityPermutation(permutation)
             ? reshaped_input
             : xla::Transpose(reshaped_input, permutation);
}

xla::XlaOp LowerPad(xla::XlaOp input, const at::Scalar& value,
                    absl::Span<const xla::int64> pad) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  return xla::Pad(input,
                  XlaHelpers::ScalarValue(value, input_shape.element_type(),
                                          input.builder()),
                  torch_lazy_tensors::compiler::XlaHelpers::
                      MakeXlaPaddingConfigFromNdPadding(pad));
}

xla::XlaOp LowerSqueeze(xla::XlaOp input, int dim) {
  if (dim == -1) {
    return SqueezeAllTrivialDimensions(input);
  }
  LTC_CHECK_GE(dim, 0);
  return SqueezeTrivialDimension(input, dim);
}

template <typename F>
xla::XlaOp LowerBitwise(xla::XlaOp lhs, xla::XlaOp rhs, const F& bit_op) {
  std::tie(lhs, rhs) = XlaHelpers::PromoteValues(lhs, rhs);
  return bit_op(lhs, rhs);
}

lazy_tensors::Shape InferAddMatMul(const ir::Node* node) {
  auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMatMul(operands[0], operands[1], operands[2]);
  };
  const ir::Output& input = node->operand(0);
  const ir::Output& weight = node->operand(1);
  const ir::Output& bias = node->operand(2);
  return ir::ops::InferOutputShape(
      {input.shape(), weight.shape(), bias.shape()}, shape_fn);
}

lazy_tensors::Shape InferAll(const ir::ops::All* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildAll(operands[0], node->dimensions(),
                    node->keep_reduced_dimensions());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferAny(const ir::ops::Any* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildAny(operands[0], node->dimensions(),
                    node->keep_reduced_dimensions());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferArgMax(const ir::ops::ArgMax* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildArgMax(operands[0], node->dim(), node->keepdim());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferArgMin(const ir::ops::ArgMin* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildArgMin(operands[0], node->dim(), node->keepdim());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferBinaryCrossEntropy(
    const ir::ops::BinaryCrossEntropy* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    absl::optional<xla::XlaOp> weight;
    if (operands.size() > 2) {
      weight = operands[2];
    }
    return BuildBinaryCrossEntropy(operands[0], operands[1], weight,
                                   node->reduction());
  };
  const ir::Output& logits = node->operand(0);
  const ir::Output& labels = node->operand(1);
  absl::optional<ir::Output> weight;
  if (node->operands().size() > 2) {
    weight = node->operand(2);
  }
  std::vector<lazy_tensors::Shape> shapes;
  for (auto& input :
       xla::util::GetValuesVector<ir::Output>({logits, labels}, {&weight})) {
    shapes.push_back(input.shape());
  }
  return ir::ops::InferOutputShape(shapes, shape_fn);
}

lazy_tensors::Shape InferBinaryCrossEntropyBackward(
    const ir::ops::BinaryCrossEntropyBackward* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    absl::optional<xla::XlaOp> weight;
    if (operands.size() > 3) {
      weight = operands[3];
    }
    return BuildBinaryCrossEntropyBackward(
        operands[0], operands[1], operands[2], weight, node->reduction());
  };
  const ir::Output& grad_output = node->operand(0);
  const ir::Output& logits = node->operand(1);
  const ir::Output& labels = node->operand(2);
  absl::optional<ir::Output> weight;
  if (node->operands().size() > 3) {
    weight = node->operand(3);
  }
  std::vector<lazy_tensors::Shape> shapes;
  for (auto& input : xla::util::GetValuesVector<ir::Output>(
           {grad_output, logits, labels}, {&weight})) {
    shapes.push_back(input.shape());
  }
  return ir::ops::InferOutputShape(shapes, shape_fn);
}

lazy_tensors::Shape InferBaddBmm(const ir::Node* node) {
  auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMatMulWithMultiplier(operands[0], operands[1], operands[2],
                                     operands[3], operands[4]);
  };
  const ir::Output& lhs = node->operand(0);
  const ir::Output& rhs = node->operand(1);
  const ir::Output& bias = node->operand(2);
  const ir::Output& product_multiplier = node->operand(3);
  const ir::Output& bias_multiplier = node->operand(4);
  return ir::ops::InferOutputShape(
      {lhs.shape(), rhs.shape(), bias.shape(), product_multiplier.shape(),
       bias_multiplier.shape()},
      shape_fn);
}

lazy_tensors::Shape InferBitwise(const ir::Node* node) {
  const ir::Output& input0 = node->operand(0);
  const ir::Output& input1 = node->operand(1);
  switch (node->op().op) {
    case at::aten::__and__: {
      auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
        return compiler::LowerBitwise(
            operands[0], operands[1],
            [](xla::XlaOp lhs, xla::XlaOp rhs) { return lhs & rhs; });
      };
      return ir::ops::InferOutputShape({input0.shape(), input1.shape()},
                                       shape_fn);
    }
    case at::aten::__or__: {
      auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
        return compiler::LowerBitwise(
            operands[0], operands[1],
            [](xla::XlaOp lhs, xla::XlaOp rhs) { return lhs | rhs; });
      };
      return ir::ops::InferOutputShape({input0.shape(), input1.shape()},
                                       shape_fn);
    }
    case at::aten::__xor__: {
      auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
        return compiler::LowerBitwise(
            operands[0], operands[1],
            [](xla::XlaOp lhs, xla::XlaOp rhs) { return lhs ^ rhs; });
      };
      return ir::ops::InferOutputShape({input0.shape(), input1.shape()},
                                       shape_fn);
    }
    default: { LTC_LOG(FATAL) << "Invalid bitwise operator: " << node->op(); }
  }
}

lazy_tensors::Shape InferConstantPadNd(const ir::ops::ConstantPadNd* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return LowerPad(operands[0], node->value(), node->pad());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferExpand(const ir::ops::Expand* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildExpand(operands[0], node->size());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferPermute(const ir::ops::Permute* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    LTC_CHECK_EQ(operands.size(), 1);
    return xla::Transpose(operands[0], node->dims());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferSplit(const ir::ops::Split* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::Tuple(
        operands[0].builder(),
        BuildSplit(operands[0], node->split_sizes(), node->dim()));
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferSqueeze(const ir::ops::Squeeze* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    LTC_CHECK_EQ(operands.size(), 1);
    return LowerSqueeze(operands[0], node->dim());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferUpsampleBilinear(
    const ir::ops::UpsampleBilinear* node) {
  const ir::Output& input = node->operand(0);
  return resize::GetForwardOutputShape2d(input.shape(), node->output_size());
}

lazy_tensors::Shape InferUpsampleBilinearBackward(
    const ir::ops::UpsampleBilinearBackward* node) {
  const ir::Output& input = node->operand(0);
  return resize::GetBackwardOutputShape2d(input.shape(), node->input_size());
}

lazy_tensors::Shape InferUpsampleNearest(const ir::ops::UpsampleNearest* node) {
  const ir::Output& input = node->operand(0);
  return resize::GetForwardOutputShape2d(input.shape(), node->output_size());
}

lazy_tensors::Shape InferUpsampleNearestBackward(
    const ir::ops::UpsampleNearestBackward* node) {
  const ir::Output& input = node->operand(0);
  return resize::GetBackwardOutputShape2d(input.shape(), node->input_size());
}

lazy_tensors::Shape InferGenericSlice(const ir::ops::GenericSlice* node) {
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildSlice(operands[0], node->base_indices(), node->sizes());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferUpdateSlice(const ir::ops::UpdateSlice* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildUpdateSlice(operands[0], operands[1], node->base_indices());
  };
  const ir::Output& input = node->operand(0);
  const ir::Output& source = node->operand(1);
  return ir::ops::InferOutputShape({input.shape(), source.shape()}, shape_fn);
}

lazy_tensors::Shape InferRelu(const ir::Node* node) {
  auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    LTC_CHECK_EQ(operands.size(), 1) << "Unexpected number of operands";
    return BuildRelu(operands[0]);
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferComparisonOp(const ir::Node* node) {
  c10::Symbol kind = node->op().op;
  auto shape_fn = [kind](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildComparisonOp(kind, operands[0], operands[1]);
  };
  const ir::Output& input0 = node->operand(0);
  const ir::Output& input1 = node->operand(1);
  return ir::ops::InferOutputShape({input0.shape(), input1.shape()}, shape_fn);
}

#define DEFINE_INFER_BINARY_OP(name, xla_fn)                                  \
  lazy_tensors::Shape Infer##name(const ir::Node* node) {                     \
    auto shape_fn = [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp { \
      auto promoted = XlaHelpers::Promote(operands[0], operands[1]);          \
      return xla_fn(promoted.first, promoted.second);                         \
    };                                                                        \
    const ir::Output& input0 = node->operand(0);                              \
    const ir::Output& input1 = node->operand(1);                              \
    return ir::ops::InferOutputShape({input0.shape(), input1.shape()},        \
                                     shape_fn);                               \
  }

DEFINE_INFER_BINARY_OP(Min, xla::Min)
DEFINE_INFER_BINARY_OP(Max, xla::Max)
DEFINE_INFER_BINARY_OP(Pow, xla::Pow)
DEFINE_INFER_BINARY_OP(Fmod, xla::Rem)
DEFINE_INFER_BINARY_OP(Atan2, xla::Atan2)

#undef DEFINE_INFER_BINARY_OP

lazy_tensors::Shape InferAdaptiveAvgPool2d(
    const ir::ops::AdaptiveAvgPool2d* node) {
  auto lower_for_shape_fn =
      [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    LTC_CHECK_EQ(operands.size(), 1);
    return BuildAdaptiveAvgPool2d(operands[0], node->output_size());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, lower_for_shape_fn);
}

lazy_tensors::Shape InferAvgPoolNd(const ir::ops::AvgPoolNd* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    LTC_CHECK_EQ(operands.size(), 1)
        << "Unexpected number of operands: " << operands.size();
    return BuildAvgPoolNd(operands[0], node->spatial_dim_count(),
                          node->kernel_size(), node->stride(), node->padding(),
                          node->ceil_mode(), node->count_include_pad());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, shape_fn);
}

lazy_tensors::Shape InferAvgPoolNdBackward(
    const ir::ops::AvgPoolNdBackward* node) {
  auto shape_fn = [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    LTC_CHECK_EQ(operands.size(), 2)
        << "Unexpected number of operands: " << operands.size();
    return BuildAvgPoolNdBackward(
        /*out_backprop=*/operands[0],
        /*input=*/operands[1], node->spatial_dim_count(), node->kernel_size(),
        node->stride(), node->padding(), node->ceil_mode(),
        node->count_include_pad());
  };
  const ir::Output& grad_output = node->operand(0);
  const ir::Output& input = node->operand(1);
  return ir::ops::InferOutputShape({grad_output.shape(), input.shape()},
                                   shape_fn);
}

lazy_tensors::Shape InferAdaptiveAvgPool3d(
    const ir::ops::AdaptiveAvgPool3d* node) {
  auto lower_for_shape_fn =
      [node](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    LTC_CHECK_EQ(operands.size(), 1);
    return BuildAdaptiveAvgPool3d(operands[0], node->output_size());
  };
  const ir::Output& input = node->operand(0);
  return ir::ops::InferOutputShape({input.shape()}, lower_for_shape_fn);
}

lazy_tensors::Shape InferAdaptiveAvgPool2dBackward(const ir::Node* node) {
  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    LTC_CHECK_EQ(operands.size(), 2);
    return BuildAdaptiveAvgPool2dBackward(/*out_backprop=*/operands[0],
                                          /*input=*/operands[1]);
  };
  const ir::Output& grad_output = node->operand(0);
  const ir::Output& input = node->operand(1);
  return ir::ops::InferOutputShape({grad_output.shape(), input.shape()},
                                   lower_for_shape_fn);
}

lazy_tensors::Shape InferAdaptiveAvgPool3dBackward(const ir::Node* node) {
  auto lower_for_shape_fn =
      [](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    LTC_CHECK_EQ(operands.size(), 2);
    return BuildAdaptiveAvgPool3dBackward(/*out_backprop=*/operands[0],
                                          /*input=*/operands[1]);
  };
  const ir::Output& grad_output = node->operand(0);
  const ir::Output& input = node->operand(1);
  return ir::ops::InferOutputShape({grad_output.shape(), input.shape()},
                                   lower_for_shape_fn);
}

#define DECLARE_UNARY_OP(name) XlaOpVector Lower##name(const ir::Node* node)
#define DECLARE_UNARY_OP2(name) \
  XlaOpVector Lower##name(const ir::ops::name* node)
#define DECLARE_BINARY_OP(name) XlaOpVector Lower##name(const ir::Node* node)

class XlaNodeLowering : public NodeLowering {
 public:
  XlaNodeLowering(ir::LoweringContext* loctx) : NodeLowering(loctx) {}

  bool Lower(const ir::Node* node) override;

  lazy_tensors::Shape Infer(const ir::Node* node) override;

  xla_backend::XlaLoweringContext* loctx() {
    return static_cast<xla_backend::XlaLoweringContext*>(loctx_);
  }

  XlaOpVector LowerToXla(const ir::Node* node);

 private:
  XlaOpVector LowerBitwise(const ir::Node* node);
  XlaOpVector LowerAdd(const ir::Node* node);
  XlaOpVector LowerDiv(const ir::Node* node);
  XlaOpVector LowerMul(const ir::Node* node);
  XlaOpVector LowerSub(const ir::Node* node);
  XlaOpVector LowerAbs(const ir::Node* node);
  XlaOpVector LowerCast(const ir::ops::Cast* node);
  XlaOpVector LowerDiagonal(const ir::ops::Diagonal* node);
  XlaOpVector LowerDiagonalViewUpdate(const ir::ops::DiagonalViewUpdate* node);
  XlaOpVector LowerDeviceData(const ir::ops::DeviceData* node);
  XlaOpVector LowerSelect(const ir::ops::Select* node);
  XlaOpVector LowerUnselect(const ir::ops::Unselect* node);
  XlaOpVector LowerGenericSlice(const ir::ops::GenericSlice* node);
  XlaOpVector LowerUpdateSlice(const ir::ops::UpdateSlice* node);
  XlaOpVector LowerAsStridedViewUpdate(
      const ir::ops::AsStridedViewUpdate* node);
  XlaOpVector LowerAsStrided(const ir::ops::AsStrided* node);
  XlaOpVector LowerGetDimensionsSize(const ir::ops::GetDimensionsSize* node);
  XlaOpVector LowerExpand(const ir::ops::Expand* node);
  XlaOpVector LowerScalar(const ir::ops::Scalar* node);
  XlaOpVector LowerLinearInterpolation(
      const ir::ops::LinearInterpolation* node);
  XlaOpVector LowerMaxUnary(const ir::Node* node);
  XlaOpVector LowerNotSupported(const ir::ops::NotSupported* node);
  DECLARE_UNARY_OP(Acos);
  DECLARE_UNARY_OP(Acosh);
  DECLARE_UNARY_OP(Bernoulli);
  DECLARE_UNARY_OP(Cos);
  DECLARE_UNARY_OP(Cosh);
  DECLARE_UNARY_OP(Asin);
  DECLARE_UNARY_OP(Asinh);
  DECLARE_UNARY_OP(Sin);
  DECLARE_UNARY_OP(Sinh);
  DECLARE_UNARY_OP(Atan);
  DECLARE_UNARY_OP(Atanh);
  DECLARE_UNARY_OP(Tan);
  DECLARE_UNARY_OP(Tanh);
  DECLARE_UNARY_OP(Neg);
  DECLARE_UNARY_OP(Exp);
  DECLARE_UNARY_OP(Expm1);
  DECLARE_UNARY_OP(HardSigmoid);
  DECLARE_UNARY_OP(HardSigmoidBackward);
  DECLARE_UNARY_OP(Log);
  DECLARE_UNARY_OP(Log1p);
  DECLARE_UNARY_OP(Erf);
  DECLARE_UNARY_OP(Erfc);
  DECLARE_UNARY_OP(Erfinv);
  DECLARE_UNARY_OP(Reciprocal);
  DECLARE_UNARY_OP(Relu);
  DECLARE_UNARY_OP(Sigmoid);
  DECLARE_UNARY_OP(Sign);
  DECLARE_UNARY_OP(SiLU);
  DECLARE_UNARY_OP(Sqrt);
  DECLARE_UNARY_OP(Rsqrt);
  DECLARE_UNARY_OP(Ceil);
  DECLARE_UNARY_OP(Floor);
  DECLARE_UNARY_OP(Round);
  DECLARE_UNARY_OP(Not);
  DECLARE_UNARY_OP(Where);
  DECLARE_BINARY_OP(Min);
  DECLARE_BINARY_OP(Max);
  DECLARE_BINARY_OP(Pow);
  DECLARE_BINARY_OP(Fmod);
  DECLARE_BINARY_OP(Atan2);
  DECLARE_BINARY_OP(Eq);
  DECLARE_BINARY_OP(Ge);
  DECLARE_BINARY_OP(Gt);
  DECLARE_BINARY_OP(Le);
  DECLARE_BINARY_OP(Lt);
  DECLARE_BINARY_OP(Ne);
  DECLARE_UNARY_OP(AddMatMul);
  DECLARE_UNARY_OP(BaddBmm);
  DECLARE_UNARY_OP(Clamp);
  DECLARE_UNARY_OP2(AdaptiveAvgPool2d);
  DECLARE_UNARY_OP2(AdaptiveAvgPool3d);
  DECLARE_UNARY_OP2(AvgPoolNd);
  DECLARE_UNARY_OP2(AvgPoolNdBackward);
  DECLARE_UNARY_OP(AdaptiveAvgPool2dBackward);
  DECLARE_UNARY_OP(AdaptiveAvgPool3dBackward);
  DECLARE_UNARY_OP2(All);
  DECLARE_UNARY_OP2(Any);
  DECLARE_UNARY_OP2(AmpForachNonFiniteCheckAndUnscale);
  DECLARE_UNARY_OP2(AmpUpdateScale);
  DECLARE_UNARY_OP2(ArgMax);
  DECLARE_UNARY_OP2(ArgMin);
  DECLARE_UNARY_OP2(BinaryCrossEntropy);
  DECLARE_UNARY_OP2(BinaryCrossEntropyBackward);
  DECLARE_UNARY_OP2(Constant);
  DECLARE_UNARY_OP2(ConstantPadNd);
  DECLARE_UNARY_OP2(Hardshrink);
  DECLARE_UNARY_OP2(HardtanhBackward);
  DECLARE_UNARY_OP2(LeakyRelu);
  DECLARE_UNARY_OP2(LeakyReluBackward);
  DECLARE_UNARY_OP2(LogBase);
  DECLARE_UNARY_OP2(LogSoftmax);
  DECLARE_UNARY_OP2(LogSoftmaxBackward);
  DECLARE_UNARY_OP2(MaskedFill);
  DECLARE_UNARY_OP2(Permute);
  DECLARE_UNARY_OP2(Resize);
  DECLARE_UNARY_OP2(RreluWithNoise);
  DECLARE_UNARY_OP2(RreluWithNoiseBackward);
  DECLARE_UNARY_OP2(Softmax);
  DECLARE_UNARY_OP2(SoftmaxBackward);
  DECLARE_UNARY_OP2(Softshrink);
  DECLARE_UNARY_OP2(Split);
  DECLARE_UNARY_OP2(Squeeze);
  DECLARE_UNARY_OP2(ShrinkBackward);
  DECLARE_UNARY_OP2(Threshold);
  DECLARE_UNARY_OP2(ThresholdBackward);
  DECLARE_UNARY_OP2(Unsqueeze);
  DECLARE_UNARY_OP2(View);
};

#undef DECLARE_BINARY_OP
#undef DECLARE_UNARY_OP2
#undef DECLARE_UNARY_OP

bool XlaNodeLowering::Lower(const ir::Node* node) {
  XlaOpVector ops = LowerToXla(node);
  if (ops.empty()) {
    return false;
  }
  LTC_CHECK_EQ(node->num_outputs(), ops.size());
  for (size_t i = 0; i < ops.size(); ++i) {
    loctx()->AssignOutputOp(ir::Output(node, i), ops[i]);
  }
  return true;
}

#define HANDLE_GENERIC_OP(name, sym) \
  case sym: {                        \
    return Lower##name(node);        \
  }

#define HANDLE_GENERIC_OP2(name, sym)                                       \
  case sym: {                                                               \
    return Lower##name(ir::NodeCast<ir::ops::name>(node, ir::OpKind(sym))); \
  }

XlaOpVector XlaNodeLowering::LowerToXla(const ir::Node* node) {
  switch (node->op().op) {
    HANDLE_GENERIC_OP(Add, at::aten::add)
    HANDLE_GENERIC_OP(Div, at::aten::div)
    HANDLE_GENERIC_OP(Mul, at::aten::mul)
    HANDLE_GENERIC_OP(Sub, at::aten::sub)
    HANDLE_GENERIC_OP(Bitwise, at::aten::__and__)
    HANDLE_GENERIC_OP(Bitwise, at::aten::__or__)
    HANDLE_GENERIC_OP(Bitwise, at::aten::__xor__)
    HANDLE_GENERIC_OP(Abs, at::aten::abs)
    HANDLE_GENERIC_OP(Acos, at::aten::acos)
    HANDLE_GENERIC_OP(Acosh, at::aten::acosh)
    HANDLE_GENERIC_OP(Bernoulli, at::aten::bernoulli)
    HANDLE_GENERIC_OP(Cos, at::aten::cos)
    HANDLE_GENERIC_OP(Cosh, at::aten::cosh)
    HANDLE_GENERIC_OP(Asin, at::aten::asin)
    HANDLE_GENERIC_OP(Asinh, at::aten::asinh)
    HANDLE_GENERIC_OP(Sin, at::aten::sin)
    HANDLE_GENERIC_OP(Sinh, at::aten::sinh)
    HANDLE_GENERIC_OP(Atan, at::aten::atan)
    HANDLE_GENERIC_OP(Atanh, at::aten::atanh)
    HANDLE_GENERIC_OP(Tan, at::aten::tan)
    HANDLE_GENERIC_OP(Tanh, at::aten::tanh)
    HANDLE_GENERIC_OP(Neg, at::aten::neg)
    HANDLE_GENERIC_OP(Exp, at::aten::exp)
    HANDLE_GENERIC_OP(Expm1, at::aten::expm1)
    HANDLE_GENERIC_OP(HardSigmoid, at::aten::hardsigmoid)
    HANDLE_GENERIC_OP(HardSigmoidBackward, at::aten::hardsigmoid_backward)
    HANDLE_GENERIC_OP(Log, at::aten::log)
    HANDLE_GENERIC_OP(Log1p, at::aten::log1p)
    HANDLE_GENERIC_OP(Erf, at::aten::erf)
    HANDLE_GENERIC_OP(Erfc, at::aten::erfc)
    HANDLE_GENERIC_OP(Erfinv, at::aten::erfinv)
    HANDLE_GENERIC_OP(Reciprocal, at::aten::reciprocal)
    HANDLE_GENERIC_OP(Relu, at::aten::relu)
    HANDLE_GENERIC_OP(Sigmoid, at::aten::sigmoid)
    HANDLE_GENERIC_OP(Sign, at::aten::sign)
    HANDLE_GENERIC_OP(SiLU, at::aten::silu)
    HANDLE_GENERIC_OP(Sqrt, at::aten::sqrt)
    HANDLE_GENERIC_OP(Rsqrt, at::aten::rsqrt)
    HANDLE_GENERIC_OP(Ceil, at::aten::ceil)
    HANDLE_GENERIC_OP(Floor, at::aten::floor)
    HANDLE_GENERIC_OP(Round, at::aten::round)
    HANDLE_GENERIC_OP(Not, at::aten::bitwise_not)
    HANDLE_GENERIC_OP(Where, at::aten::where)
    HANDLE_GENERIC_OP(Min, at::aten::min)
    HANDLE_GENERIC_OP(Pow, at::aten::pow)
    HANDLE_GENERIC_OP(Fmod, at::aten::fmod)
    HANDLE_GENERIC_OP(Atan2, at::aten::atan2)
    HANDLE_GENERIC_OP(Eq, at::aten::eq)
    HANDLE_GENERIC_OP(Ge, at::aten::ge)
    HANDLE_GENERIC_OP(Gt, at::aten::gt)
    HANDLE_GENERIC_OP(Le, at::aten::le)
    HANDLE_GENERIC_OP(Lt, at::aten::lt)
    HANDLE_GENERIC_OP(Ne, at::aten::ne)
    HANDLE_GENERIC_OP(AddMatMul, at::aten::addmm)
    HANDLE_GENERIC_OP(BaddBmm, at::aten::baddbmm)
    HANDLE_GENERIC_OP(Clamp, at::aten::clamp)
    HANDLE_GENERIC_OP2(AdaptiveAvgPool2d, at::aten::adaptive_avg_pool2d)
    HANDLE_GENERIC_OP2(AdaptiveAvgPool3d, at::aten::adaptive_avg_pool3d)
    HANDLE_GENERIC_OP(AdaptiveAvgPool2dBackward,
                      at::aten::adaptive_avg_pool2d_backward)
    HANDLE_GENERIC_OP(AdaptiveAvgPool3dBackward,
                      at::aten::adaptive_avg_pool3d_backward)
    HANDLE_GENERIC_OP2(AvgPoolNd, at::aten::avg_pool2d)
    HANDLE_GENERIC_OP2(AvgPoolNd, at::aten::avg_pool3d)
    HANDLE_GENERIC_OP2(AvgPoolNdBackward, at::aten::avg_pool2d_backward)
    HANDLE_GENERIC_OP2(AvgPoolNdBackward, at::aten::avg_pool3d_backward)
    HANDLE_GENERIC_OP2(AsStrided, at::aten::as_strided)
    HANDLE_GENERIC_OP2(Diagonal, at::aten::diagonal)
    HANDLE_GENERIC_OP2(Expand, at::aten::expand)
    HANDLE_GENERIC_OP2(Hardshrink, at::aten::hardshrink)
    HANDLE_GENERIC_OP2(HardtanhBackward, at::aten::hardtanh_backward)
    HANDLE_GENERIC_OP2(ConstantPadNd, at::aten::constant_pad_nd)
    HANDLE_GENERIC_OP2(LeakyRelu, at::aten::leaky_relu)
    HANDLE_GENERIC_OP2(LeakyReluBackward, at::aten::leaky_relu_backward)
    HANDLE_GENERIC_OP2(LogBase, at::aten::log2)
    HANDLE_GENERIC_OP2(LogBase, at::aten::log10)
    HANDLE_GENERIC_OP2(LogSoftmax, at::aten::log_softmax)
    HANDLE_GENERIC_OP2(LogSoftmaxBackward, at::aten::_log_softmax_backward_data)
    HANDLE_GENERIC_OP2(MaskedFill, at::aten::masked_fill)
    HANDLE_GENERIC_OP2(Permute, at::aten::permute)
    HANDLE_GENERIC_OP2(Resize, at::aten::resize)
    HANDLE_GENERIC_OP2(RreluWithNoise, at::aten::rrelu_with_noise)
    HANDLE_GENERIC_OP2(RreluWithNoiseBackward,
                       at::aten::rrelu_with_noise_backward)
    HANDLE_GENERIC_OP2(ShrinkBackward, at::aten::hardshrink_backward)
    HANDLE_GENERIC_OP2(ShrinkBackward, at::aten::softshrink_backward)
    HANDLE_GENERIC_OP2(Softmax, at::aten::softmax)
    HANDLE_GENERIC_OP2(SoftmaxBackward, at::aten::_softmax_backward_data)
    HANDLE_GENERIC_OP2(Softshrink, at::aten::softshrink)
    HANDLE_GENERIC_OP2(Split, at::aten::split)
    HANDLE_GENERIC_OP2(Squeeze, at::aten::squeeze)
    HANDLE_GENERIC_OP2(Threshold, at::aten::threshold)
    HANDLE_GENERIC_OP2(ThresholdBackward, at::aten::threshold_backward)
    HANDLE_GENERIC_OP2(Unsqueeze, at::aten::unsqueeze)
    HANDLE_GENERIC_OP2(View, at::aten::view)
    HANDLE_GENERIC_OP2(All, at::aten::all)
    HANDLE_GENERIC_OP2(Any, at::aten::any)
    HANDLE_GENERIC_OP2(AmpForachNonFiniteCheckAndUnscale,
                       at::aten::_amp_foreach_non_finite_check_and_unscale_)
    HANDLE_GENERIC_OP2(AmpUpdateScale, at::aten::_amp_update_scale)
    HANDLE_GENERIC_OP2(ArgMax, at::aten::argmax)
    HANDLE_GENERIC_OP2(ArgMin, at::aten::argmin)
    HANDLE_GENERIC_OP2(BinaryCrossEntropy, at::aten::binary_cross_entropy)
    HANDLE_GENERIC_OP2(BinaryCrossEntropyBackward,
                       at::aten::binary_cross_entropy_backward)
    case at::aten::max: {
      size_t arity = node->operands().size();
      if (arity == 2) {
        return LowerMax(node);
      }
      LTC_CHECK_EQ(arity, 1);
      LTC_CHECK(dynamic_cast<const ir::ops::Generic*>(node));
      return LowerMaxUnary(node);
    }
    case at::prim::Constant: {
      // TODO(asuhan): rework to remove ambiguity between Scalar and Constant
      // nodes to make dynamic_cast unnecessary.
      auto scalar_node = dynamic_cast<const ir::ops::Scalar*>(node);
      if (scalar_node) {
        return LowerScalar(scalar_node);
      }
      auto constant_node = dynamic_cast<const ir::ops::Constant*>(node);
      LTC_CHECK(constant_node);
      return LowerConstant(constant_node);
    }
    default: {
      if (node->op() == *ir::ops::ltc_cast) {
        return LowerCast(ir::NodeCast<ir::ops::Cast>(node, *ir::ops::ltc_cast));
      }
      if (node->op() == *ir::ops::ltc_device_data) {
        return LowerDeviceData(
            ir::NodeCast<ir::ops::DeviceData>(node, *ir::ops::ltc_device_data));
      }
      if (node->op() == *ir::ops::ltc_select) {
        return LowerSelect(
            ir::NodeCast<ir::ops::Select>(node, *ir::ops::ltc_select));
      }
      if (node->op() == *ir::ops::ltc_unselect) {
        return LowerUnselect(
            ir::NodeCast<ir::ops::Unselect>(node, *ir::ops::ltc_unselect));
      }
      if (node->op() == *ir::ops::ltc_generic_slice) {
        return LowerGenericSlice(ir::NodeCast<ir::ops::GenericSlice>(
            node, *ir::ops::ltc_generic_slice));
      }
      if (node->op() == *ir::ops::ltc_update_slice) {
        return LowerUpdateSlice(ir::NodeCast<ir::ops::UpdateSlice>(
            node, *ir::ops::ltc_update_slice));
      }
      if (node->op() == *ir::ops::ltc_as_strided_view_update) {
        return LowerAsStridedViewUpdate(
            ir::NodeCast<ir::ops::AsStridedViewUpdate>(
                node, *ir::ops::ltc_as_strided_view_update));
      }
      if (node->op() == *ir::ops::ltc_diagonal_view_update) {
        return LowerDiagonalViewUpdate(
            ir::NodeCast<ir::ops::DiagonalViewUpdate>(
                node, *ir::ops::ltc_diagonal_view_update));
      }
      if (node->op() == *ir::ops::ltc_get_dimensions_size) {
        return LowerGetDimensionsSize(ir::NodeCast<ir::ops::GetDimensionsSize>(
            node, *ir::ops::ltc_get_dimensions_size));
      }
      if (node->op() == *ir::ops::ltc_moving_average) {
        return LowerLinearInterpolation(
            ir::NodeCast<ir::ops::LinearInterpolation>(
                node, *ir::ops::ltc_moving_average));
      }
      if (node->op() == *ir::ops::ltc_not_supported) {
        return LowerNotSupported(ir::NodeCast<ir::ops::NotSupported>(
            node, *ir::ops::ltc_not_supported));
      }
      break;
    }
  }
  return {};
}

#undef HANDLE_GENERIC_OP2
#undef HANDLE_GENERIC_OP

XlaOpVector XlaNodeLowering::LowerBernoulli(const ir::Node* node) {
  xla::XlaOp probability = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp rng_seed = loctx()->GetOutputOp(node->operand(1));
  const xla::Shape& probability_shape =
      compiler::XlaHelpers::ShapeOfXlaOp(probability);
  xla::Shape bcast_shape = compiler::XlaHelpers::XlaShape(node->shape());
  bcast_shape.set_element_type(probability_shape.element_type());
  xla::XlaOp bcast_probability = XlaHelpers::ImplicitBroadcast(
      probability, probability_shape, bcast_shape);
  return {BuildBernoulli(
      bcast_probability, rng_seed,
      xla::ComputationClient::XlaPrimitiveType(node->shape().element_type()))};
}

XlaOpVector XlaNodeLowering::LowerAdd(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp op0 = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp op1 = loctx()->GetOutputOp(node->operand(1));
  return {XlaHelpers::PromotedAdd(op0, op1)};
}

XlaOpVector XlaNodeLowering::LowerDiv(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp op0 = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp op1 = loctx()->GetOutputOp(node->operand(1));
  return {XlaHelpers::PromotedDiv(op0, op1)};
}

XlaOpVector XlaNodeLowering::LowerMul(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp op0 = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp op1 = loctx()->GetOutputOp(node->operand(1));
  return {XlaHelpers::PromotedMul(op0, op1)};
}

XlaOpVector XlaNodeLowering::LowerSub(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp op0 = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp op1 = loctx()->GetOutputOp(node->operand(1));
  return {XlaHelpers::PromotedSub(op0, op1)};
}

XlaOpVector XlaNodeLowering::LowerBitwise(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp op0 = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp op1 = loctx()->GetOutputOp(node->operand(1));
  std::tie(op0, op1) = XlaHelpers::PromoteValues(op0, op1);
  switch (node->op().op) {
    case at::aten::__and__: {
      return {op0 & op1};
    }
    case at::aten::__or__: {
      return {op0 | op1};
    }
    case at::aten::__xor__: {
      return {op0 ^ op1};
    }
    default: { LTC_LOG(FATAL) << "Invalid bitwise operator: " << node->op(); }
  }
}

XlaOpVector XlaNodeLowering::LowerAbs(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp xla_input = loctx()->GetOutputOp(node->operand(0));
  return {torch_lazy_tensors::BuildAbs(xla_input)};
}

XlaOpVector XlaNodeLowering::LowerCast(const ir::ops::Cast* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::PrimitiveType raw_from =
      node->stype()
          ? xla::ComputationClient::XlaPrimitiveType(
                torch_lazy_tensors::TensorTypeToLtcType(*node->stype()))
          : input_shape.element_type();
  xla::PrimitiveType raw_to = xla::ComputationClient::XlaPrimitiveType(
      node->dtype() ? torch_lazy_tensors::TensorTypeToLtcType(*node->dtype())
                    : node->type());
  return {torch_lazy_tensors::ConvertToRaw(
      input, input_shape.element_type(), raw_from,
      xla::ComputationClient::XlaPrimitiveType(node->type()), raw_to,
      /*device=*/nullptr)};
}

XlaOpVector XlaNodeLowering::LowerDiagonal(const ir::ops::Diagonal* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {torch_lazy_tensors::BuildDiagonal(input, node->offset(), node->dim1(),
                                            node->dim2())};
}

XlaOpVector XlaNodeLowering::LowerDiagonalViewUpdate(
    const ir::ops::DiagonalViewUpdate* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp target = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  return {BuildDiagonalViewUpdate(target, input, node->offset(), node->dim1(),
                                  node->dim2())};
}

XlaOpVector XlaNodeLowering::LowerDeviceData(const ir::ops::DeviceData* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  return {loctx()->GetParameter(node->data())};
}

XlaOpVector XlaNodeLowering::LowerSelect(const ir::ops::Select* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {xla::SliceInDim(
      input, node->start(), node->end(),
      ir::ops::Select::GetStride(node->start(), node->end(), node->stride()),
      node->dim())};
}

XlaOpVector XlaNodeLowering::LowerUnselect(const ir::ops::Unselect* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp target = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp source = loctx()->GetOutputOp(node->operand(1));
  return {BuildUnselect(
      target, source, node->dim(), node->start(), node->end(),
      ir::ops::Select::GetStride(node->start(), node->end(), node->stride()))};
}

XlaOpVector XlaNodeLowering::LowerGenericSlice(
    const ir::ops::GenericSlice* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildSlice(input, node->base_indices(), node->sizes())};
}

XlaOpVector XlaNodeLowering::LowerUpdateSlice(
    const ir::ops::UpdateSlice* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp source = loctx()->GetOutputOp(node->operand(1));
  return {BuildUpdateSlice(input, source, node->base_indices())};
}

XlaOpVector XlaNodeLowering::LowerAsStridedViewUpdate(
    const ir::ops::AsStridedViewUpdate* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp target = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  return {compiler::LowerAsStridedViewUpdate(
      target, input, node->size(), node->stride(), node->storage_offset())};
}

XlaOpVector XlaNodeLowering::LowerAsStrided(const ir::ops::AsStrided* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {compiler::LowerAsStrided(input, node->size(), node->stride(),
                                   node->storage_offset())};
}

XlaOpVector XlaNodeLowering::LowerGetDimensionsSize(
    const ir::ops::GetDimensionsSize* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {XlaHelpers::GetDimensionsSize({input}, node->dimensions()).size};
}

XlaOpVector XlaNodeLowering::LowerExpand(const ir::ops::Expand* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildExpand(input, node->size())};
}

XlaOpVector XlaNodeLowering::LowerScalar(const ir::ops::Scalar* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  using ir::ops::operator<<;
  xla::Literal literal(xla::ShapeUtil::MakeShape(
      xla::ComputationClient::XlaPrimitiveType(node->shape().element_type()),
      {}));
  switch (node->shape().element_type()) {
    case lazy_tensors::PrimitiveType::PRED:
      literal.Set<bool>({}, static_cast<bool>(node->value().toInt()));
      break;
    case lazy_tensors::PrimitiveType::S8:
      literal.Set<xla::int8>({},
                             static_cast<xla::int8>(node->value().toChar()));
      break;
    case lazy_tensors::PrimitiveType::U8:
      literal.Set<xla::uint8>({},
                              static_cast<xla::uint8>(node->value().toByte()));
      break;
    case lazy_tensors::PrimitiveType::S16:
      literal.Set<xla::int16>({},
                              static_cast<xla::int16>(node->value().toShort()));
      break;
    case lazy_tensors::PrimitiveType::U16:
      literal.Set<xla::uint16>(
          {}, static_cast<xla::uint16>(node->value().toShort()));
      break;
    case lazy_tensors::PrimitiveType::S32:
      literal.Set<xla::int32>({},
                              static_cast<xla::int32>(node->value().toInt()));
      break;
    case lazy_tensors::PrimitiveType::U32:
      literal.Set<xla::uint32>({},
                               static_cast<xla::uint32>(node->value().toInt()));
      break;
    case lazy_tensors::PrimitiveType::S64:
      literal.Set<xla::int64>({},
                              static_cast<xla::int64>(node->value().toLong()));
      break;
    case lazy_tensors::PrimitiveType::U64:
      literal.Set<xla::uint64>(
          {}, static_cast<xla::uint64>(node->value().toLong()));
      break;
    case lazy_tensors::PrimitiveType::F32:
      literal.Set<float>({}, static_cast<float>(node->value().toDouble()));
      break;
    case lazy_tensors::PrimitiveType::F64:
      literal.Set<double>({}, node->value().toDouble());
      break;
    case lazy_tensors::PrimitiveType::BF16:
      literal.Set<xla::bfloat16>(
          {}, static_cast<xla::bfloat16>(node->value().toDouble()));
      break;
    case lazy_tensors::PrimitiveType::F16:
      literal.Set<xla::half>({},
                             static_cast<xla::half>(node->value().toDouble()));
      break;
    case lazy_tensors::PrimitiveType::C64:
      literal.Set<xla::complex64>(
          {}, xla::complex64(node->value().toComplexFloat()));
      break;
    case lazy_tensors::PrimitiveType::C128:
      literal.Set<xla::complex128>(
          {}, xla::complex128(node->value().toComplexDouble()));
      break;
    default:
      LTC_ERROR() << "Unable to lower scalar " << node->value() << " of shape "
                  << node->shape();
  }

  xla::XlaOp op = xla::ConstantLiteral(loctx()->builder(), literal);
  if (node->shape().rank() > 0) {
    op = xla::Broadcast(op, node->shape().dimensions());
  }
  return {op};
}

XlaOpVector XlaNodeLowering::LowerLinearInterpolation(
    const ir::ops::LinearInterpolation* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp value = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp new_value = loctx()->GetOutputOp(node->operand(1));
  return {XlaHelpers::LinearInterpolation(value, new_value, node->alpha())};
}

XlaOpVector XlaNodeLowering::LowerAdaptiveAvgPool2d(
    const ir::ops::AdaptiveAvgPool2d* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildAdaptiveAvgPool2d(input, node->output_size())};
}

XlaOpVector XlaNodeLowering::LowerAdaptiveAvgPool3d(
    const ir::ops::AdaptiveAvgPool3d* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildAdaptiveAvgPool3d(input, node->output_size())};
}

XlaOpVector XlaNodeLowering::LowerAdaptiveAvgPool2dBackward(
    const ir::Node* node) {
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  return {BuildAdaptiveAvgPool2dBackward(
      /*out_backprop=*/grad_output, /*input=*/input)};
}

XlaOpVector XlaNodeLowering::LowerAdaptiveAvgPool3dBackward(
    const ir::Node* node) {
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  return {BuildAdaptiveAvgPool3dBackward(
      /*out_backprop=*/grad_output, /*input=*/input)};
}

XlaOpVector XlaNodeLowering::LowerAvgPoolNd(const ir::ops::AvgPoolNd* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildAvgPoolNd(input, node->spatial_dim_count(), node->kernel_size(),
                         node->stride(), node->padding(), node->ceil_mode(),
                         node->count_include_pad())};
}

XlaOpVector XlaNodeLowering::LowerAvgPoolNdBackward(
    const ir::ops::AvgPoolNdBackward* node) {
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  return {BuildAvgPoolNdBackward(
      /*out_backprop=*/grad_output, /*input=*/input, node->spatial_dim_count(),
      node->kernel_size(), node->stride(), node->padding(), node->ceil_mode(),
      node->count_include_pad())};
}

XlaOpVector XlaNodeLowering::LowerAll(const ir::ops::All* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildAll(input, node->dimensions(), node->keep_reduced_dimensions())};
}

XlaOpVector XlaNodeLowering::LowerAny(const ir::ops::Any* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildAny(input, node->dimensions(), node->keep_reduced_dimensions())};
}

XlaOpVector XlaNodeLowering::LowerAmpForachNonFiniteCheckAndUnscale(
    const ir::ops::AmpForachNonFiniteCheckAndUnscale* node) {
  LTC_CHECK_GE(node->operands().size(), 3);
  XlaOpVector inputs;
  for (size_t i = 0; i < node->operands().size() - 2; ++i) {
    inputs.push_back(loctx()->GetOutputOp(node->operand(i)));
  }
  return BuildAmpForeachNonFiniteCheckAndUnscale(
      inputs, loctx()->GetOutputOp(node->operand(node->operands().size() - 2)),
      loctx()->GetOutputOp(node->operand(node->operands().size() - 1)));
}

XlaOpVector XlaNodeLowering::LowerAmpUpdateScale(
    const ir::ops::AmpUpdateScale* node) {
  return BuildAmpUpdateScale(
      loctx()->GetOutputOp(node->operand(0)),
      loctx()->GetOutputOp(node->operand(1)),
      loctx()->GetOutputOp(node->operand(2)), node->scale_growth_factor(),
      node->scale_backoff_factor(), node->growth_interval());
}

XlaOpVector XlaNodeLowering::LowerArgMax(const ir::ops::ArgMax* node) {
  return {BuildArgMax(loctx()->GetOutputOp(node->operand(0)), node->dim(),
                      node->keepdim())};
}

XlaOpVector XlaNodeLowering::LowerArgMin(const ir::ops::ArgMin* node) {
  return {BuildArgMin(loctx()->GetOutputOp(node->operand(0)), node->dim(),
                      node->keepdim())};
}

XlaOpVector XlaNodeLowering::LowerBinaryCrossEntropy(
    const ir::ops::BinaryCrossEntropy* node) {
  xla::XlaOp logits = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp labels = loctx()->GetOutputOp(node->operand(1));
  absl::optional<xla::XlaOp> weight;
  if (node->operands().size() > 2) {
    weight = loctx()->GetOutputOp(node->operand(2));
  }
  return {BuildBinaryCrossEntropy(logits, labels, weight, node->reduction())};
}

XlaOpVector XlaNodeLowering::LowerBinaryCrossEntropyBackward(
    const ir::ops::BinaryCrossEntropyBackward* node) {
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp logits = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp labels = loctx()->GetOutputOp(node->operand(2));
  absl::optional<xla::XlaOp> weight;
  if (node->operands().size() > 3) {
    weight = loctx()->GetOutputOp(node->operand(3));
  }
  return {BuildBinaryCrossEntropyBackward(grad_output, logits, labels, weight,
                                          node->reduction())};
}

XlaOpVector XlaNodeLowering::LowerMaxUnary(const ir::Node* node) {
  xla::XlaOp xla_input = loctx()->GetOutputOp(node->operand(0));
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(xla_input);
  xla::PrimitiveType element_type = input_shape.element_type();
  torch_lazy_tensors::Helpers::MinMax min_max =
      torch_lazy_tensors::Helpers::MinMaxValues(
          XlaHelpers::LazyTensorPrimitiveType(element_type));
  xla::XlaOp init_value =
      XlaHelpers::ScalarValue(min_max.min, element_type, loctx()->builder());
  xla::XlaOp result = xla::Reduce(
      xla_input, init_value, XlaHelpers::CreateMaxComputation(element_type),
      xla::util::Iota<xla::int64>(input_shape.rank()));
  return {result};
}

XlaOpVector XlaNodeLowering::LowerNotSupported(
    const ir::ops::NotSupported* node) {
  LTC_ERROR() << "Node not supported: " << node->ToString();
}

XlaOpVector XlaNodeLowering::LowerWhere(const ir::Node* node) {
  xla::XlaOp xla_condition = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp xla_input = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp xla_other = loctx()->GetOutputOp(node->operand(2));
  xla::XlaOp pred_condition =
      ConvertTo(xla_condition, XlaHelpers::TypeOfXlaOp(xla_condition),
                xla::PrimitiveType::PRED, /*device=*/nullptr);
  auto promoted_branches = XlaHelpers::PromoteShapes(xla_input, xla_other);
  return {xla::Select(pred_condition, promoted_branches.first,
                      promoted_branches.second)};
}

XlaOpVector XlaNodeLowering::LowerAddMatMul(const ir::Node* node) {
  LTC_CHECK_EQ(node->operands().size(), 3) << "Unexpected number of operands";
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp weight = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp bias = loctx()->GetOutputOp(node->operand(2));
  return {BuildMatMul(input, weight, bias)};
}

XlaOpVector XlaNodeLowering::LowerBaddBmm(const ir::Node* node) {
  xla::XlaOp lhs = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp rhs = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp bias = loctx()->GetOutputOp(node->operand(2));
  xla::XlaOp product_multiplier = loctx()->GetOutputOp(node->operand(3));
  xla::XlaOp bias_multiplier = loctx()->GetOutputOp(node->operand(4));
  std::tie(lhs, rhs) = XlaHelpers::PromoteValues(lhs, rhs);
  return {BuildMatMulWithMultiplier(lhs, rhs, bias, product_multiplier,
                                    bias_multiplier)};
}

XlaOpVector XlaNodeLowering::LowerClamp(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp xla_input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp xla_min = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp xla_max = loctx()->GetOutputOp(node->operand(2));
  xla::PrimitiveType input_type = XlaHelpers::TypeOfXlaOp(xla_input);
  xla_min = ConvertTo(xla_min, XlaHelpers::TypeOfXlaOp(xla_min), input_type,
                      /*device=*/nullptr);
  xla_max = ConvertTo(xla_max, XlaHelpers::TypeOfXlaOp(xla_max), input_type,
                      /*device=*/nullptr);
  return {xla::Clamp(xla_min, xla_input, xla_max)};
}

XlaOpVector XlaNodeLowering::LowerLeakyRelu(const ir::ops::LeakyRelu* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildLeakyRelu(input, node->negative_slope())};
}

XlaOpVector XlaNodeLowering::LowerLeakyReluBackward(
    const ir::ops::LeakyReluBackward* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  return {BuildLeakyReluBackward(grad_output, input, node->negative_slope())};
}

XlaOpVector XlaNodeLowering::LowerLogBase(const ir::ops::LogBase* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp xla_input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp result = xla::Log(xla_input);
  xla::XlaOp ln_base = XlaHelpers::ScalarValue<float>(
      1.0 / std::log(node->base()),
      xla::ComputationClient::XlaPrimitiveType(node->shape().element_type()),
      xla_input.builder());
  return {result * ln_base};
}

XlaOpVector XlaNodeLowering::LowerLogSoftmax(const ir::ops::LogSoftmax* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp result = BuildLogSoftmax(input, node->dim());
  return {CastToScalarType(result, node->dtype())};
}

XlaOpVector XlaNodeLowering::LowerLogSoftmaxBackward(
    const ir::ops::LogSoftmaxBackward* node) {
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp output = loctx()->GetOutputOp(node->operand(1));
  return {BuildLogSoftmaxGrad(/*grad_output=*/grad_output, /*output=*/output,
                              node->dim())};
}

XlaOpVector XlaNodeLowering::LowerMaskedFill(const ir::ops::MaskedFill* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp mask = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp zero =
      xla::Zero(loctx()->builder(), XlaHelpers::TypeOfXlaOp(mask));
  xla::XlaOp mask_pred = xla::Ne(mask, zero);
  // Input shape is the same as output shape.
  const lazy_tensors::Shape& input_shape = node->shape();
  xla::XlaOp value = xla::Broadcast(
      XlaHelpers::ScalarValue(
          node->value(),
          xla::ComputationClient::XlaPrimitiveType(input_shape.element_type()),
          input.builder()),
      input_shape.dimensions());
  return {xla::Select(mask_pred, value, input)};
}

XlaOpVector XlaNodeLowering::LowerPermute(const ir::ops::Permute* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {xla::Transpose(input, node->dims())};
}

XlaOpVector XlaNodeLowering::LowerResize(const ir::ops::Resize* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildResize(input, node->size())};
}

XlaOpVector XlaNodeLowering::LowerRreluWithNoise(
    const ir::ops::RreluWithNoise* node) {
  LTC_CHECK_EQ(node->num_outputs(), 2);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp rng_seed = loctx()->GetOutputOp(node->operand(1));
  return {BuildRrelu(input, node->lower(), node->upper(), node->training(),
                     rng_seed)};
}

XlaOpVector XlaNodeLowering::LowerRreluWithNoiseBackward(
    const ir::ops::RreluWithNoiseBackward* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  xla::XlaOp noise = loctx()->GetOutputOp(node->operand(2));
  return {BuildRreluBackward(grad_output, input, noise, node->lower(),
                             node->upper(), node->training())};
}

XlaOpVector XlaNodeLowering::LowerSiLU(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp xla_input = loctx()->GetOutputOp(node->operand(0));
  return {xla_input * BuildSigmoid(xla_input)};
}

XlaOpVector XlaNodeLowering::LowerSoftmax(const ir::ops::Softmax* node) {
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp result = BuildSoftmax(input, node->dim());
  return {CastToScalarType(result, node->dtype())};
}

XlaOpVector XlaNodeLowering::LowerSoftmaxBackward(
    const ir::ops::SoftmaxBackward* node) {
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp output = loctx()->GetOutputOp(node->operand(1));
  return {BuildSoftmaxGrad(/*grad_output=*/grad_output, /*output=*/output,
                           node->dim())};
}

XlaOpVector XlaNodeLowering::LowerSoftshrink(const ir::ops::Softshrink* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildSoftshrink(input, node->lambda())};
}

XlaOpVector XlaNodeLowering::LowerSplit(const ir::ops::Split* node) {
  LTC_CHECK_EQ(node->num_outputs(), node->split_sizes().size());
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildSplit(input, node->split_sizes(), node->dim())};
}

XlaOpVector XlaNodeLowering::LowerSqueeze(const ir::ops::Squeeze* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {compiler::LowerSqueeze(input, node->dim())};
}

XlaOpVector XlaNodeLowering::LowerShrinkBackward(
    const ir::ops::ShrinkBackward* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  return {BuildShrinkBackward(grad_output, input, node->lambda())};
}

XlaOpVector XlaNodeLowering::LowerThreshold(const ir::ops::Threshold* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildThreshold(input, input, node->threshold(), node->value())};
}

XlaOpVector XlaNodeLowering::LowerThresholdBackward(
    const ir::ops::ThresholdBackward* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  return {BuildThreshold(input, grad_output, node->threshold(), 0)};
}

XlaOpVector XlaNodeLowering::LowerUnsqueeze(const ir::ops::Unsqueeze* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildUnsqueeze(input, node->dim())};
}

XlaOpVector XlaNodeLowering::LowerView(const ir::ops::View* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildView(input, node->output_size())};
}

#define HANDLE_CASE(type1, type2)                  \
  case lazy_tensors::PrimitiveType::type1: {       \
    xla_literal.Set({}, literal.data<type2>()[0]); \
    break;                                         \
  }

xla::Literal XlaLiteral(const lazy_tensors::Literal& literal) {
  const lazy_tensors::Shape& shape = literal.shape();
  LTC_CHECK_EQ(shape.rank(), 0);
  xla::Literal xla_literal;
  switch (shape.element_type()) {
    HANDLE_CASE(PRED, bool);
    HANDLE_CASE(S8, int8_t);
    HANDLE_CASE(S16, int16_t);
    HANDLE_CASE(S32, int32_t);
    HANDLE_CASE(S64, int64_t);
    HANDLE_CASE(U8, uint8_t);
    HANDLE_CASE(F32, float);
    HANDLE_CASE(F64, double);
    default: {
      LTC_LOG(FATAL) << "Not implemented yet: " << shape.element_type();
    }
  }
  return xla_literal;
}

#undef HANDLE_CASE

XlaOpVector XlaNodeLowering::LowerConstant(const ir::ops::Constant* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  return {xla::ConstantLiteral(loctx()->builder(), XlaLiteral(node->value()))};
}

XlaOpVector XlaNodeLowering::LowerConstantPadNd(
    const ir::ops::ConstantPadNd* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {LowerPad(input, node->value(), node->pad())};
}

XlaOpVector XlaNodeLowering::LowerHardshrink(const ir::ops::Hardshrink* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(0));
  return {BuildHardshrink(input, node->lambda())};
}

XlaOpVector XlaNodeLowering::LowerHardtanhBackward(
    const ir::ops::HardtanhBackward* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp input = loctx()->GetOutputOp(node->operand(1));
  return {BuildHardtanhBackward(grad_output, input, node->min_val(),
                                node->max_val())};
}

XlaOpVector XlaNodeLowering::LowerHardSigmoidBackward(const ir::Node* node) {
  LTC_CHECK_EQ(node->num_outputs(), 1);
  xla::XlaOp xla_grad_output = loctx()->GetOutputOp(node->operand(0));
  xla::XlaOp xla_input = loctx()->GetOutputOp(node->operand(1));
  return {BuildHardSigmoidBackward(xla_grad_output, xla_input)};
}

#define DEFINE_UNARY_OP(name, xla_fn)                              \
  XlaOpVector XlaNodeLowering::Lower##name(const ir::Node* node) { \
    LTC_CHECK_EQ(node->num_outputs(), 1);                          \
    xla::XlaOp xla_input = loctx()->GetOutputOp(node->operand(0)); \
    return {xla_fn(xla_input)};                                    \
  }

#define DEFINE_BINARY_OP(name, xla_fn)                              \
  XlaOpVector XlaNodeLowering::Lower##name(const ir::Node* node) {  \
    LTC_CHECK_EQ(node->num_outputs(), 1);                           \
    xla::XlaOp xla_input0 = loctx()->GetOutputOp(node->operand(0)); \
    xla::XlaOp xla_input1 = loctx()->GetOutputOp(node->operand(1)); \
    auto promoted = XlaHelpers::Promote(xla_input0, xla_input1);    \
    return {xla_fn(promoted.first, promoted.second)};               \
  }

#define DEFINE_COMPARISON_OP(name, kind)                           \
  XlaOpVector XlaNodeLowering::Lower##name(const ir::Node* node) { \
    LTC_CHECK_EQ(node->num_outputs(), 1);                          \
    xla::XlaOp xla_input = loctx()->GetOutputOp(node->operand(0)); \
    xla::XlaOp xla_other = loctx()->GetOutputOp(node->operand(1)); \
    return {BuildComparisonOp(kind, xla_input, xla_other)};        \
  }

DEFINE_UNARY_OP(Acos, xla::Acos)
DEFINE_UNARY_OP(Acosh, xla::Acosh)
DEFINE_UNARY_OP(Cos, xla::Cos)
DEFINE_UNARY_OP(Cosh, xla::Cosh)
DEFINE_UNARY_OP(Asin, xla::Asin)
DEFINE_UNARY_OP(Asinh, xla::Asinh)
DEFINE_UNARY_OP(Sin, xla::Sin)
DEFINE_UNARY_OP(Sinh, xla::Sinh)
DEFINE_UNARY_OP(Atan, xla::Atan)
DEFINE_UNARY_OP(Atanh, xla::Atanh)
DEFINE_UNARY_OP(Tan, xla::Tan)
DEFINE_UNARY_OP(Tanh, xla::Tanh)
DEFINE_UNARY_OP(Neg, xla::Neg)
DEFINE_UNARY_OP(Exp, xla::Exp)
DEFINE_UNARY_OP(Expm1, xla::Expm1)
DEFINE_UNARY_OP(HardSigmoid, torch_lazy_tensors::BuildHardSigmoid)
DEFINE_UNARY_OP(Log, xla::Log)
DEFINE_UNARY_OP(Log1p, xla::Log1p)
DEFINE_UNARY_OP(Erf, xla::Erf)
DEFINE_UNARY_OP(Erfc, xla::Erfc)
DEFINE_UNARY_OP(Erfinv, xla::ErfInv)
DEFINE_UNARY_OP(Reciprocal, torch_lazy_tensors::BuildReciprocal)
DEFINE_UNARY_OP(Relu, torch_lazy_tensors::BuildRelu)
DEFINE_UNARY_OP(Sigmoid, torch_lazy_tensors::BuildSigmoid)
DEFINE_UNARY_OP(Sign, torch_lazy_tensors::BuildSign)
DEFINE_UNARY_OP(Sqrt, xla::Sqrt)
DEFINE_UNARY_OP(Rsqrt, xla::Rsqrt)
DEFINE_UNARY_OP(Ceil, xla::Ceil)
DEFINE_UNARY_OP(Floor, xla::Floor)
DEFINE_UNARY_OP(Round, xla::RoundToEven)
DEFINE_UNARY_OP(Not, xla::Not)
DEFINE_BINARY_OP(Min, xla::Min)
DEFINE_BINARY_OP(Max, xla::Max)
DEFINE_BINARY_OP(Pow, xla::Pow)
DEFINE_BINARY_OP(Fmod, xla::Rem)
DEFINE_BINARY_OP(Atan2, xla::Atan2)
DEFINE_COMPARISON_OP(Eq, at::aten::eq)
DEFINE_COMPARISON_OP(Ge, at::aten::ge)
DEFINE_COMPARISON_OP(Gt, at::aten::gt)
DEFINE_COMPARISON_OP(Le, at::aten::le)
DEFINE_COMPARISON_OP(Lt, at::aten::lt)
DEFINE_COMPARISON_OP(Ne, at::aten::ne)

#undef DEFINE_COMPARISON_OP
#undef DEFINE_BINARY_OP
#undef DEFINE_UNARY_OP

lazy_tensors::Shape XlaNodeLowering::Infer(const ir::Node* node) {
  const ir::OpKind& kind = node->op();
  switch (kind.op) {
    case at::aten::addmm: {
      return InferAddMatMul(node);
    }
    case at::aten::all: {
      return InferAll(
          ir::NodeCast<ir::ops::All>(node, ir::OpKind(at::aten::all)));
    }
    case at::aten::any: {
      return InferAny(
          ir::NodeCast<ir::ops::Any>(node, ir::OpKind(at::aten::any)));
    }
    case at::aten::argmax: {
      return InferArgMax(
          ir::NodeCast<ir::ops::ArgMax>(node, ir::OpKind(at::aten::argmax)));
    }
    case at::aten::argmin: {
      return InferArgMin(
          ir::NodeCast<ir::ops::ArgMin>(node, ir::OpKind(at::aten::argmin)));
    }
    case at::aten::baddbmm: {
      return InferBaddBmm(node);
    }
    case at::aten::binary_cross_entropy: {
      return InferBinaryCrossEntropy(ir::NodeCast<ir::ops::BinaryCrossEntropy>(
          node, ir::OpKind(at::aten::binary_cross_entropy)));
    }
    case at::aten::binary_cross_entropy_backward: {
      return InferBinaryCrossEntropyBackward(
          ir::NodeCast<ir::ops::BinaryCrossEntropyBackward>(
              node, ir::OpKind(at::aten::binary_cross_entropy_backward)));
    }
    case at::aten::__and__:
    case at::aten::__or__:
    case at::aten::__xor__: {
      return InferBitwise(node);
    }
    case at::aten::min: {
      return InferMin(node);
    }
    case at::aten::max: {
      LTC_CHECK_EQ(node->operands().size(), 2);
      return InferMax(node);
    }
    case at::aten::pow: {
      return InferPow(node);
    }
    case at::aten::fmod: {
      return InferFmod(node);
    }
    case at::aten::atan2: {
      return InferAtan2(node);
    }
    case at::aten::relu: {
      return InferRelu(node);
    }
    case at::aten::eq:
    case at::aten::ge:
    case at::aten::gt:
    case at::aten::le:
    case at::aten::lt:
    case at::aten::ne: {
      return InferComparisonOp(node);
    }
    case at::aten::constant_pad_nd: {
      return InferConstantPadNd(ir::NodeCast<ir::ops::ConstantPadNd>(
          node, ir::OpKind(at::aten::constant_pad_nd)));
    }
    case at::aten::expand: {
      return InferExpand(
          ir::NodeCast<ir::ops::Expand>(node, ir::OpKind(at::aten::expand)));
    }
    case at::aten::permute: {
      return InferPermute(
          ir::NodeCast<ir::ops::Permute>(node, ir::OpKind(at::aten::permute)));
    }
    case at::aten::split: {
      return InferSplit(
          ir::NodeCast<ir::ops::Split>(node, ir::OpKind(at::aten::split)));
    }
    case at::aten::squeeze: {
      return InferSqueeze(
          ir::NodeCast<ir::ops::Squeeze>(node, ir::OpKind(at::aten::squeeze)));
    }
    case at::aten::upsample_bilinear2d: {
      return InferUpsampleBilinear(ir::NodeCast<ir::ops::UpsampleBilinear>(
          node, ir::OpKind(at::aten::upsample_bilinear2d)));
    }
    case at::aten::upsample_bilinear2d_backward: {
      return InferUpsampleBilinearBackward(
          ir::NodeCast<ir::ops::UpsampleBilinearBackward>(
              node, ir::OpKind(at::aten::upsample_bilinear2d_backward)));
    }
    case at::aten::upsample_nearest2d: {
      return InferUpsampleNearest(ir::NodeCast<ir::ops::UpsampleNearest>(
          node, ir::OpKind(at::aten::upsample_nearest2d)));
    }
    case at::aten::upsample_nearest2d_backward: {
      return InferUpsampleNearestBackward(
          ir::NodeCast<ir::ops::UpsampleNearestBackward>(
              node, ir::OpKind(at::aten::upsample_nearest2d_backward)));
    }
    case at::aten::avg_pool2d: {
      return InferAvgPoolNd(ir::NodeCast<ir::ops::AvgPoolNd>(
          node, ir::OpKind(at::aten::avg_pool2d)));
    }
    case at::aten::avg_pool3d: {
      return InferAvgPoolNd(ir::NodeCast<ir::ops::AvgPoolNd>(
          node, ir::OpKind(at::aten::avg_pool3d)));
    }
    case at::aten::avg_pool2d_backward: {
      return InferAvgPoolNdBackward(ir::NodeCast<ir::ops::AvgPoolNdBackward>(
          node, ir::OpKind(at::aten::avg_pool2d_backward)));
    }
    case at::aten::avg_pool3d_backward: {
      return InferAvgPoolNdBackward(ir::NodeCast<ir::ops::AvgPoolNdBackward>(
          node, ir::OpKind(at::aten::avg_pool3d_backward)));
    }
    case at::aten::adaptive_avg_pool2d: {
      return InferAdaptiveAvgPool2d(ir::NodeCast<ir::ops::AdaptiveAvgPool2d>(
          node, ir::OpKind(at::aten::adaptive_avg_pool2d)));
    }
    case at::aten::adaptive_avg_pool3d: {
      return InferAdaptiveAvgPool3d(ir::NodeCast<ir::ops::AdaptiveAvgPool3d>(
          node, ir::OpKind(at::aten::adaptive_avg_pool3d)));
    }
    case at::aten::adaptive_avg_pool2d_backward: {
      return InferAdaptiveAvgPool2dBackward(node);
    }
    case at::aten::adaptive_avg_pool3d_backward: {
      return InferAdaptiveAvgPool3dBackward(node);
    }
    default: {
      if (kind == *ir::ops::ltc_generic_slice) {
        return InferGenericSlice(ir::NodeCast<ir::ops::GenericSlice>(
            node, *ir::ops::ltc_generic_slice));
      }
      if (kind == *ir::ops::ltc_update_slice) {
        return InferUpdateSlice(ir::NodeCast<ir::ops::UpdateSlice>(
            node, *ir::ops::ltc_update_slice));
      }
      LTC_LOG(FATAL) << "Shape inference not supported for operator: " << kind;
    }
  }
}

}  // namespace

std::unique_ptr<NodeLowering> NodeLowering::Create(ir::LoweringContext* loctx) {
  return std::make_unique<compiler::XlaNodeLowering>(loctx);
}

NodeLowering* NodeLowering::Get() {
  static XlaNodeLowering* xla_node_lowering = new XlaNodeLowering(nullptr);
  return xla_node_lowering;
}

namespace xla_backend {

XlaOpVector LowerNodeToXla(const ir::Node* node, XlaLoweringContext* loctx) {
  auto node_lowering = NodeLowering::Create(loctx);
  XlaNodeLowering* xla_node_lowering =
      static_cast<XlaNodeLowering*>(node_lowering.get());
  return xla_node_lowering->LowerToXla(node);
}

}  // namespace xla_backend

NodeLowering* GetXlaNodeLowering() { return NodeLowering::Get(); }

std::unique_ptr<NodeLowering> CreateXlaNodeLowering(
    ir::LoweringContext* loctx) {
  return NodeLowering::Create(loctx);
}

}  // namespace compiler
}  // namespace torch_lazy_tensors
