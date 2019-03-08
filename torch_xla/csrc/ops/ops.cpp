#include "torch_xla/csrc/ops/ops.h"

#include <cmath>

#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/nll_loss.h"
#include "torch_xla/csrc/ops/arithmetic_ir_ops.h"
#include "torch_xla/csrc/ops/constant.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/permute.h"
#include "torch_xla/csrc/ops/sum.h"
#include "torch_xla/csrc/pooling.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace ir {
namespace ops {

#define PTXLA_UNARY_OP(name, sym, xla_fn)                         \
  NodePtr name(const Value& input) {                              \
    auto lower_fn = [](const Node& node,                          \
                       LoweringContext* loctx) -> XlaOpVector {   \
      xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0)); \
      return node.ReturnOp(xla_fn(xla_input), loctx);             \
    };                                                            \
    return GenericOp(OpKind(sym), OpList{input}, input.shape(),   \
                     std::move(lower_fn));                        \
  }

#define PTXLA_BINARY_OP(name, sym, xla_fn)                                \
  NodePtr name(const Value& input0, const Value& input1) {                \
    auto lower_fn = [](const Node& node,                                  \
                       LoweringContext* loctx) -> XlaOpVector {           \
      xla::XlaOp xla_input0 = loctx->GetOutputOp(node.operand(0));        \
      xla::XlaOp xla_input1 = loctx->GetOutputOp(node.operand(1));        \
      return node.ReturnOp(xla_fn(xla_input0, xla_input1), loctx);        \
    };                                                                    \
    return GenericOp(OpKind(sym), OpList{input0, input1}, input0.shape(), \
                     std::move(lower_fn));                                \
  }

PTXLA_UNARY_OP(Acos, at::aten::acos, xla::Acos);
PTXLA_UNARY_OP(Cos, at::aten::cos, xla::Cos);
PTXLA_UNARY_OP(Cosh, at::aten::cosh, xla::Cosh);
PTXLA_UNARY_OP(Asin, at::aten::asin, xla::Asin);
PTXLA_UNARY_OP(Sin, at::aten::sin, xla::Sin);
PTXLA_UNARY_OP(Sinh, at::aten::sinh, xla::Sinh);
PTXLA_UNARY_OP(Atan, at::aten::atan, xla::Atan);
PTXLA_UNARY_OP(Tan, at::aten::tan, xla::Tan);
PTXLA_UNARY_OP(Tanh, at::aten::tanh, xla::Tanh);
PTXLA_UNARY_OP(Neg, at::aten::neg, xla::Neg);
PTXLA_UNARY_OP(Abs, at::aten::abs, xla::Abs);
PTXLA_UNARY_OP(Exp, at::aten::exp, xla::Exp);
PTXLA_UNARY_OP(Expm1, at::aten::expm1, xla::Expm1);
PTXLA_UNARY_OP(Log, at::aten::log, xla::Log);
PTXLA_UNARY_OP(Log1p, at::aten::log1p, xla::Log1p);
PTXLA_UNARY_OP(Erf, at::aten::erf, xla::Erf);
PTXLA_UNARY_OP(Erfc, at::aten::erfc, xla::Erfc);
PTXLA_UNARY_OP(Erfinv, at::aten::erfinv, xla::ErfInv);
PTXLA_UNARY_OP(Sqrt, at::aten::sqrt, xla::Sqrt);
PTXLA_UNARY_OP(Rsqrt, at::aten::rsqrt, xla::Rsqrt);
PTXLA_UNARY_OP(Ceil, at::aten::ceil, xla::Ceil);
PTXLA_UNARY_OP(Floor, at::aten::floor, xla::Floor);

PTXLA_BINARY_OP(Min, at::aten::min, xla::Min);
PTXLA_BINARY_OP(Max, at::aten::max, xla::Max);
PTXLA_BINARY_OP(Pow, at::aten::pow, xla::Pow);
PTXLA_BINARY_OP(Fmod, at::aten::fmod, xla::Rem);
PTXLA_BINARY_OP(Atan2, at::aten::atan2, xla::Atan2);

NodePtr Trunc(const Value& input) { return Floor(Abs(input)) * SignOp(input); }

NodePtr FracOp(const Value& input) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_input_floor = xla::Floor(xla_input);
    return node.ReturnOp(xla_input - xla_input_floor, loctx);
  };
  return GenericOp(OpKind(at::aten::frac), OpList{input}, input.shape(),
                   std::move(lower_fn));
}

NodePtr LogBase(const Value& input, OpKind op, double base) {
  auto lower_fn = [base](const Node& node,
                         LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp result = xla::Log(xla_input);
    xla::XlaOp ln_base = XlaHelpers::ScalarValue<float>(
        1.0 / std::log(base), node.shape().element_type(), xla_input.builder());
    return node.ReturnOp(result * ln_base, loctx);
  };
  return GenericOp(op, OpList{input}, input.shape(), std::move(lower_fn));
}

NodePtr ReciprocalOp(const Value& input) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    return node.ReturnOp(BuildReciprocal(xla_input), loctx);
  };
  return GenericOp(OpKind(at::aten::reciprocal), OpList{input}, input.shape(),
                   std::move(lower_fn));
}

NodePtr SignOp(const Value& input) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    return node.ReturnOp(BuildSign(xla_input), loctx);
  };
  return GenericOp(OpKind(at::aten::sign), OpList{input}, input.shape(),
                   std::move(lower_fn));
}

NodePtr ReluOp(const Value& input) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_output = BuildRelu(xla_input);
    return node.ReturnOp(xla_output, loctx);
  };
  auto lower_for_shape_fn =
      [](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1) << "Unexpected number of operands";
    return BuildRelu(operands[0]);
  };
  xla::Shape output_shape =
      InferOutputShape({input.shape()}, lower_for_shape_fn);
  return GenericOp(OpKind(at::aten::relu), OpList{input}, output_shape,
                   std::move(lower_fn));
}

NodePtr TransposeOp(const Value& input, xla::int64 dim0, xla::int64 dim1) {
  return ir::MakeNode<ir::ops::Permute>(
      input,
      XlaHelpers::MakeTransposePermutation(/*dim0=*/dim0, /*dim1=*/dim1,
                                           /*rank=*/input.shape().rank()));
}

NodePtr Sigmoid(const Value& input) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    return node.ReturnOp(BuildSigmoid(xla_input), loctx);
  };
  return GenericOp(OpKind(at::aten::sigmoid), OpList{input}, input.shape(),
                   std::move(lower_fn));
}

NodePtr Clamp(const Value& input, c10::optional<at::Scalar> min,
              c10::optional<at::Scalar> max) {
  const xla::Shape& input_shape = input.shape();
  XlaHelpers::MinMax min_max =
      XlaHelpers::MinMaxValues(input_shape.element_type());
  if (!min) {
    min = min_max.min;
  }
  if (!max) {
    max = min_max.max;
  }
  NodePtr min_value = ScalarOp(*min, input_shape.element_type());
  NodePtr max_value = ScalarOp(*max, input_shape.element_type());
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_min = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_max = loctx->GetOutputOp(node.operand(2));
    return node.ReturnOp(xla::Clamp(xla_min, xla_input, xla_max), loctx);
  };
  return GenericOp(OpKind(at::aten::clamp), OpList{input, min_value, max_value},
                   input_shape, std::move(lower_fn));
}

NodePtr AddMatMulOp(const Value& input, const Value& weight,
                    const Value& bias) {
  const xla::PrecisionConfig::Precision precision_level =
      XlaHelpers::mat_mul_precision();
  auto lower_fn = [precision_level](const Node& node,
                                    LoweringContext* loctx) -> XlaOpVector {
    XLA_CHECK_EQ(node.operands().size(), 3) << "Unexpected number of operands";
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_weight = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_bias = loctx->GetOutputOp(node.operand(2));
    const auto bias_sizes = XlaHelpers::SizesOfXlaOp(xla_bias);
    xla::PrecisionConfig precision_config =
        XlaHelpers::BuildPrecisionConfig(precision_level);
    xla::XlaOp xla_dot = xla::Dot(xla_input, xla_weight, &precision_config);
    const auto dot_sizes = XlaHelpers::SizesOfXlaOp(xla_dot);
    if (bias_sizes != dot_sizes) {
      xla_bias = BuildExpand(xla_bias, dot_sizes);
    }
    xla::XlaOp xla_output = xla_dot + xla_bias;
    return node.ReturnOp(xla_output, loctx);
  };
  auto lower_for_shape_fn =
      [](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 2) << "Unexpected number of operands";
    return xla::Dot(operands[0], operands[1]);
  };
  xla::Shape output_shape =
      InferOutputShape({input.shape(), weight.shape()}, lower_for_shape_fn);
  return GenericOp(OpKind(at::aten::addmm), OpList{input, weight, bias},
                   output_shape, std::move(lower_fn));
}

NodePtr Dot(const Value& input, const Value& weight) {
  const xla::PrecisionConfig::Precision precision_level =
      XlaHelpers::mat_mul_precision();
  auto lower_fn = [precision_level](const Node& node,
                                    LoweringContext* loctx) -> XlaOpVector {
    XLA_CHECK_EQ(node.operands().size(), 2) << "Unexpected number of operands";
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_weight = loctx->GetOutputOp(node.operand(1));
    xla::PrecisionConfig precision_config =
        XlaHelpers::BuildPrecisionConfig(precision_level);
    return node.ReturnOp(xla::Dot(xla_input, xla_weight, &precision_config),
                         loctx);
  };
  auto lower_for_shape_fn =
      [](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 2) << "Unexpected number of operands";
    return xla::Dot(operands[0], operands[1]);
  };
  xla::Shape output_shape =
      InferOutputShape({input.shape(), weight.shape()}, lower_for_shape_fn);
  return GenericOp(OpKind(at::aten::mm), OpList{input, weight}, output_shape,
                   std::move(lower_fn));
}

NodePtr MatMul(const Value& lhs, const Value& rhs) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_lhs = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_rhs = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(CreateMatMul(xla_lhs, xla_rhs), loctx);
  };
  auto lower_for_shape_fn =
      [](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands) -> xla::XlaOp {
    return CreateMatMul(operands[0], operands[1]);
  };
  xla::Shape output_shape =
      InferOutputShape({lhs.shape(), rhs.shape()}, lower_for_shape_fn);
  return GenericOp(OpKind(at::aten::matmul), OpList{lhs, rhs}, output_shape,
                   std::move(lower_fn));
}

NodePtr NllLossOp(const Value& logits, const Value& labels) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp logits = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp labels = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_output = BuildNllLoss(logits, labels);
    return node.ReturnOp(xla_output, loctx);
  };
  auto lower_for_shape_fn =
      [](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 2) << "Unexpected number of operands";
    return BuildNllLoss(/*logits=*/operands[0], /*labels=*/operands[1]);
  };
  xla::Shape output_shape =
      InferOutputShape({logits.shape(), labels.shape()}, lower_for_shape_fn);
  return GenericOp(OpKind(at::aten::nll_loss), OpList{logits, labels},
                   output_shape, std::move(lower_fn));
}

NodePtr NllLossBackwardOp(const Value& logits, const Value& labels) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp logits = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp labels = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_output = BuildNllLossBackward(logits, labels);
    return node.ReturnOp(xla_output, loctx);
  };
  auto lower_for_shape_fn =
      [](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 2) << "Unexpected number of operands";
    return BuildNllLossBackward(/*logits=*/operands[0], /*labels=*/operands[1]);
  };
  xla::Shape output_shape =
      InferOutputShape({logits.shape(), labels.shape()}, lower_for_shape_fn);
  return GenericOp(OpKind(at::aten::nll_loss_backward), OpList{logits, labels},
                   output_shape, std::move(lower_fn));
}

NodePtr AdaptiveAvgPool2dBackward(const Value& grad_output,
                                  const Value& input) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp grad_output = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp input = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_output = BuildAdaptiveAvgPool2dBackward(
        /*out_backprop=*/grad_output, /*input=*/input);
    return node.ReturnOp(xla_output, loctx);
  };
  auto lower_for_shape_fn =
      [](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 2);
    return BuildAdaptiveAvgPool2dBackward(/*out_backprop=*/operands[0],
                                          /*input=*/operands[1]);
  };
  xla::Shape output_shape = InferOutputShape(
      {grad_output.shape(), input.shape()}, lower_for_shape_fn);
  return GenericOp(OpKind(at::aten::adaptive_avg_pool2d_backward),
                   OpList{grad_output, input}, output_shape,
                   std::move(lower_fn));
}

NodePtr ComparisonOp(c10::Symbol kind, const Value& input, const Value& other) {
  auto lower_fn = [kind](const Node& node,
                         LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_other = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_output = BuildComparisonOp(kind, xla_input, xla_other);
    return node.ReturnOp(xla_output, loctx);
  };
  xla::Shape output_shape = input.shape();
  output_shape.set_element_type(xla::PrimitiveType::PRED);
  return GenericOp(OpKind(kind), {input, other}, std::move(output_shape),
                   std::move(lower_fn));
}

NodePtr ComparisonOp(c10::Symbol kind, const Value& input,
                     const at::Scalar& other) {
  return ComparisonOp(kind, input, MakeNode<Scalar>(other, input.shape()));
}

NodePtr Where(const Value& condition, const Value& input, const Value& other) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp xla_condition = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_other = loctx->GetOutputOp(node.operand(2));
    xla::XlaOp pred_condition =
        ConvertTo(xla_condition, XlaHelpers::TypeOfXlaOp(xla_condition),
                  xla::PrimitiveType::PRED, /*device=*/nullptr);
    return node.ReturnOp(xla::Select(pred_condition, xla_input, xla_other),
                         loctx);
  };
  return GenericOp(OpKind(at::aten::where), {condition, input, other},
                   input.shape(), std::move(lower_fn));
}

NodePtr ARange(const at::Scalar& start, const at::Scalar& end,
               const at::Scalar& step, at::ScalarType scalar_type) {
  xla::PrimitiveType type = MakeXlaPrimitiveType(scalar_type,
                                                 /*device=*/nullptr);
  xla::Literal values;
  switch (type) {
    case xla::PrimitiveType::BF16:
      values = XlaHelpers::Range<tensorflow::bfloat16>(
          static_cast<tensorflow::bfloat16>(start.toFloat()),
          static_cast<tensorflow::bfloat16>(end.toFloat()),
          static_cast<tensorflow::bfloat16>(step.toFloat()));
      break;
    case xla::PrimitiveType::F32:
      values = XlaHelpers::Range<float>(start.toFloat(), end.toFloat(),
                                        step.toFloat());
      break;
    case xla::PrimitiveType::F64:
      values = XlaHelpers::Range<double>(start.toDouble(), end.toDouble(),
                                         step.toDouble());
      break;
    case xla::PrimitiveType::U8:
      values = XlaHelpers::Range<xla::uint8>(start.toByte(), end.toByte(),
                                             step.toByte());
      break;
    case xla::PrimitiveType::S8:
      values = XlaHelpers::Range<xla::int8>(start.toChar(), end.toChar(),
                                            step.toChar());
      break;
    case xla::PrimitiveType::S16:
      values = XlaHelpers::Range<xla::int16>(start.toShort(), end.toShort(),
                                             step.toShort());
      break;
    case xla::PrimitiveType::S32:
      values = XlaHelpers::Range<xla::int32>(start.toInt(), end.toInt(),
                                             step.toInt());
      break;
    case xla::PrimitiveType::S64:
      values = XlaHelpers::Range<xla::int64>(start.toLong(), end.toLong(),
                                             step.toLong());
      break;
    default:
      XLA_ERROR() << "XLA type not supported: " << type;
  }
  return MakeNode<Constant>(std::move(values));
}

NodePtr BroadcastTensors(tensorflow::gtl::ArraySlice<const Value> tensors) {
  auto lower_fn = [](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    std::vector<xla::XlaOp> xla_operands;
    for (const Output& operand : node.operands()) {
      xla_operands.push_back(loctx->GetOutputOp(operand));
    }
    return node.ReturnOps(CreateBroadcastTensors(xla_operands), loctx);
  };
  std::vector<xla::Shape> tensor_shapes;
  for (const Value& tensor : tensors) {
    tensor_shapes.push_back(tensor.shape());
  }
  auto lower_for_shape_fn =
      [&](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    auto results = CreateBroadcastTensors(operands);
    return xla::Tuple(results.front().builder(), results);
  };
  return GenericOp(OpKind(at::aten::broadcast_tensors), tensors,
                   InferOutputShape(tensor_shapes, lower_for_shape_fn),
                   std::move(lower_fn), /*num_outputs=*/tensors.size());
}

NodePtr Norm(const Value& input, c10::optional<at::Scalar> p,
             c10::optional<at::ScalarType> dtype,
             tensorflow::gtl::ArraySlice<const xla::int64> dims, bool keepdim) {
  auto dimensions = xla::util::ToVector<xla::int64>(dims);
  if (dimensions.empty()) {
    dimensions = xla::util::Iota<xla::int64>(input.shape().rank());
  }
  if (!p.has_value() || p->toDouble() == 2.0) {
    NodePtr square = input * input;
    NodePtr result = MakeNode<Sum>(square, dimensions, keepdim, dtype);
    return Sqrt(result);
  }
  double norm_value = p->toDouble();
  if (norm_value == 1.0) {
    // Contrary to documentation, norm(p=1) has nothing to do with traces and
    // standard mathematical definitions of nuclear norms:
    //
    //   >>> import torch
    //   >>> x = torch.randn(4, 4)
    //   >>> print(torch.norm(x, 1))
    //   tensor(11.9437)
    //   >>> print(torch.trace(x.abs()))
    //   tensor(3.1235)
    //   >>> print(x.abs().sum())
    //   tensor(11.9437)
    return MakeNode<Sum>(Abs(input), dimensions, keepdim, dtype);
  }
  // Generic sum(x^p)^(1/p) norms.
  NodePtr norm_exp = ScalarOp(norm_value, input.shape().element_type());
  NodePtr norm_exp_inv =
      ScalarOp(1.0 / norm_value, input.shape().element_type());
  NodePtr exp = Pow(input, norm_exp);
  NodePtr result = MakeNode<Sum>(exp, dimensions, keepdim, dtype);
  return Pow(result, norm_exp_inv);
}

NodePtr Identity(xla::int64 lines, xla::int64 cols,
                 xla::PrimitiveType element_type) {
  auto lower_fn = [=](const Node& node, LoweringContext* loctx) -> XlaOpVector {
    return node.ReturnOp(
        xla::IdentityMatrix(loctx->builder(), element_type, lines, cols),
        loctx);
  };
  return GenericOp(OpKind(at::aten::eye), {},
                   xla::ShapeUtil::MakeShape(element_type, {lines, cols}),
                   std::move(lower_fn));
}

NodePtr Elu(const Value& input, at::Scalar alpha, at::Scalar scale,
            at::Scalar input_scale) {
  const xla::Shape& shape = input.shape();
  NodePtr scaled_input = input * ScalarOp(input_scale, shape);
  NodePtr zero = ScalarOp(0, shape);
  NodePtr one = ScalarOp(1, shape);
  NodePtr alpha_scalar = ScalarOp(alpha, shape);
  return (Max(zero, scaled_input) +
          Min(zero, alpha_scalar * (Exp(scaled_input) - one))) *
         ScalarOp(scale, shape);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
