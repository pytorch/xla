#include "torch_xla/csrc/ops/ops.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/nll_loss.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/pooling.h"

namespace torch_xla {
namespace ir {
namespace ops {

#define PTXLA_UNARY_OP(name, sym, xla_fn)                               \
  NodePtr name(const Value& input) {                                    \
    auto lower_fn = [](const ir::Node& node,                            \
                       ir::LoweringContext* loctx) -> ir::XlaOpVector { \
      xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));       \
      return node.ReturnOp(xla_fn(xla_input), loctx);                   \
    };                                                                  \
    return ir::ops::GenericOp(ir::OpKind(sym), ir::OpList{input},       \
                              input.shape(), std::move(lower_fn));      \
  }

#define PTXLA_BINARY_OP(name, sym, xla_fn)                                 \
  NodePtr name(const Value& input0, const Value& input1) {                 \
    auto lower_fn = [](const ir::Node& node,                               \
                       ir::LoweringContext* loctx) -> ir::XlaOpVector {    \
      xla::XlaOp xla_input0 = loctx->GetOutputOp(node.operand(0));         \
      xla::XlaOp xla_input1 = loctx->GetOutputOp(node.operand(1));         \
      return node.ReturnOp(xla_fn(xla_input0, xla_input1), loctx);         \
    };                                                                     \
    return ir::ops::GenericOp(ir::OpKind(sym), ir::OpList{input0, input1}, \
                              input0.shape(), std::move(lower_fn));        \
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
PTXLA_UNARY_OP(Sqrt, at::aten::sqrt, xla::Sqrt);

PTXLA_BINARY_OP(Min, at::aten::min, xla::Min);
PTXLA_BINARY_OP(Max, at::aten::max, xla::Max);
PTXLA_BINARY_OP(Pow, at::aten::pow, xla::Pow);

NodePtr ReluOp(const Value& input) {
  auto lower_fn = [](const ir::Node& node,
                     ir::LoweringContext* loctx) -> ir::XlaOpVector {
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
      ir::ops::InferOutputShape({input.shape()}, lower_for_shape_fn);
  return ir::ops::GenericOp(ir::OpKind(at::aten::relu), ir::OpList{input},
                            output_shape, std::move(lower_fn));
}

NodePtr TransposeOp(const Value& input) {
  auto lower_fn = [](const ir::Node& node,
                     ir::LoweringContext* loctx) -> ir::XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_output = xla::Transpose(xla_input, {1, 0});
    return node.ReturnOp(xla_output, loctx);
  };
  auto lower_for_shape_fn =
      [](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1) << "Unexpected number of operands";
    return xla::Transpose(operands[0], {1, 0});
  };
  xla::Shape output_shape =
      ir::ops::InferOutputShape({input.shape()}, lower_for_shape_fn);
  return ir::ops::GenericOp(ir::OpKind(at::aten::t), ir::OpList{input},
                            output_shape, std::move(lower_fn));
}

NodePtr Sigmoid(const Value& input) {
  auto lower_fn = [](const ir::Node& node,
                     ir::LoweringContext* loctx) -> ir::XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    return node.ReturnOp(BuildSigmoid(xla_input), loctx);
  };
  return ir::ops::GenericOp(ir::OpKind(at::aten::sigmoid), ir::OpList{input},
                            input.shape(), std::move(lower_fn));
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
  ir::NodePtr min_value = ir::ops::ScalarOp(*min, input_shape.element_type());
  ir::NodePtr max_value = ir::ops::ScalarOp(*max, input_shape.element_type());
  auto lower_fn = [](const ir::Node& node,
                     ir::LoweringContext* loctx) -> ir::XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_min = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_max = loctx->GetOutputOp(node.operand(2));
    return node.ReturnOp(xla::Clamp(xla_min, xla_input, xla_max), loctx);
  };
  return ir::ops::GenericOp(ir::OpKind(at::aten::clamp),
                            ir::OpList{input, min_value, max_value},
                            input_shape, std::move(lower_fn));
}

NodePtr AddMatMulOp(const Value& input, const Value& weight, const Value& bias,
                    bool use_full_conv_precision) {
  const auto precision_level = use_full_conv_precision
                                   ? xla::PrecisionConfig::HIGHEST
                                   : xla::PrecisionConfig::DEFAULT;
  auto lower_fn = [precision_level](
                      const ir::Node& node,
                      ir::LoweringContext* loctx) -> ir::XlaOpVector {
    XLA_CHECK_EQ(node.operands().size(), 3) << "Unexpected number of operands";
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_weight = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_bias = loctx->GetOutputOp(node.operand(2));
    const auto bias_sizes =
        XlaHelpers::ShapeSizes(XlaHelpers::ShapeOfXlaOp(xla_bias));
    xla::PrecisionConfig precision_config =
        XlaHelpers::BuildPrecisionConfig(precision_level);
    xla::XlaOp xla_dot = xla::Dot(xla_input, xla_weight, &precision_config);
    const auto dot_sizes =
        XlaHelpers::ShapeSizes(XlaHelpers::ShapeOfXlaOp(xla_dot));
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
  xla::Shape output_shape = ir::ops::InferOutputShape(
      {input.shape(), weight.shape()}, lower_for_shape_fn);
  return ir::ops::GenericOp(ir::OpKind(at::aten::addmm),
                            ir::OpList{input, weight, bias}, output_shape,
                            std::move(lower_fn));
}

NodePtr MatMulOp(const Value& input, const Value& weight,
                 bool use_full_conv_precision) {
  const auto precision_level = use_full_conv_precision
                                   ? xla::PrecisionConfig::HIGHEST
                                   : xla::PrecisionConfig::DEFAULT;
  auto lower_fn = [precision_level](
                      const ir::Node& node,
                      ir::LoweringContext* loctx) -> ir::XlaOpVector {
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
  xla::Shape output_shape = ir::ops::InferOutputShape(
      {input.shape(), weight.shape()}, lower_for_shape_fn);
  return ir::ops::GenericOp(ir::OpKind(at::aten::mm), ir::OpList{input, weight},
                            output_shape, std::move(lower_fn));
}

NodePtr NllLossOp(const Value& logits, const Value& labels) {
  auto lower_fn = [](const ir::Node& node,
                     ir::LoweringContext* loctx) -> ir::XlaOpVector {
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
  xla::Shape output_shape = ir::ops::InferOutputShape(
      {logits.shape(), labels.shape()}, lower_for_shape_fn);
  return ir::ops::GenericOp(ir::OpKind(at::aten::nll_loss),
                            ir::OpList{logits, labels}, output_shape,
                            std::move(lower_fn));
}

NodePtr NllLossBackwardOp(const Value& logits, const Value& labels) {
  auto lower_fn = [](const ir::Node& node,
                     ir::LoweringContext* loctx) -> ir::XlaOpVector {
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
  xla::Shape output_shape = ir::ops::InferOutputShape(
      {logits.shape(), labels.shape()}, lower_for_shape_fn);
  return ir::ops::GenericOp(ir::OpKind(at::aten::nll_loss_backward),
                            ir::OpList{logits, labels}, output_shape,
                            std::move(lower_fn));
}

NodePtr AdaptiveAvgPool2dBackward(const Value& grad_output,
                                  const Value& input) {
  auto lower_fn = [](const ir::Node& node,
                     ir::LoweringContext* loctx) -> ir::XlaOpVector {
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
  xla::Shape output_shape = ir::ops::InferOutputShape(
      {grad_output.shape(), input.shape()}, lower_for_shape_fn);
  return ir::ops::GenericOp(ir::OpKind(at::aten::adaptive_avg_pool2d_backward),
                            ir::OpList{grad_output, input}, output_shape,
                            std::move(lower_fn));
}

NodePtr ComparisonOp(c10::Symbol kind, const Value& input, const Value& other) {
  auto lower_fn = [kind](const ir::Node& node,
                         ir::LoweringContext* loctx) -> ir::XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_other = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_output = BuildComparisonOp(kind, xla_input, xla_other);
    return node.ReturnOp(xla_output, loctx);
  };
  xla::Shape output_shape = input.shape();
  output_shape.set_element_type(xla::PrimitiveType::PRED);
  return ir::ops::GenericOp(ir::OpKind(kind), {input, other},
                            std::move(output_shape), std::move(lower_fn));
}

NodePtr ComparisonOp(c10::Symbol kind, const Value& input,
                     const at::Scalar& other) {
  return ComparisonOp(kind, input,
                      ir::MakeNode<ir::ops::Scalar>(other, input.shape()));
}

NodePtr Where(const Value& condition, const Value& input, const Value& other) {
  auto lower_fn = [](const ir::Node& node,
                     ir::LoweringContext* loctx) -> ir::XlaOpVector {
    xla::XlaOp xla_condition = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_other = loctx->GetOutputOp(node.operand(2));
    xla::XlaOp pred_condition =
        ConvertTo(xla_condition, XlaHelpers::TypeOfXlaOp(xla_condition),
                  xla::PrimitiveType::PRED, /*device=*/nullptr);
    return node.ReturnOp(xla::Select(pred_condition, xla_input, xla_other),
                         loctx);
  };
  return ir::ops::GenericOp(ir::OpKind(at::aten::where),
                            {condition, input, other}, input.shape(),
                            std::move(lower_fn));
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
