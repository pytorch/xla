#include "torch_xla/csrc/translator.h"
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch_xla/csrc/batch_norm.h"
#include "torch_xla/csrc/convolution.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/nll_loss.h"
#include "torch_xla/csrc/pooling.h"
#include "torch_xla/csrc/reduction.h"
#include "torch_xla/csrc/size_ops.h"
#include "torch_xla/csrc/softmax_builder.h"
#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace {

xla::XlaOp GetConstantOp(xla::XlaBuilder* builder,
                         const torch::jit::Node* node) {
  auto value = toIValue(node->output()).value();
  if (value.isTensor()) {
    auto tensor = value.toTensor();
    xla::Shape shape = CreateComputationShapeFromTensor(tensor,
                                                        /*device=*/nullptr);
    auto literal = GetTensorLiteral(tensor, &shape, /*device=*/nullptr);
    return xla::ConstantLiteral(builder, literal);
  } else if (value.isDouble()) {
    return xla::ConstantR0<float>(builder, value.toDouble());
  } else if (value.isInt()) {
    return xla::ConstantR0<xla::int64>(builder, value.toInt());
  } else if (value.isIntList()) {
    auto value_list = value.toIntList();
    std::vector<xla::int64> elements(value_list->elements().begin(),
                                     value_list->elements().end());
    return xla::ConstantR1<xla::int64>(builder, elements);
  } else if (value.isBoolList()) {
    auto value_list = value.toBoolList();
    std::vector<xla::int64> elements(value_list->elements().begin(),
                                     value_list->elements().end());
    return xla::ConstantR1<xla::int64>(builder, elements);
  } else if (value.isDoubleList()) {
    auto value_list = value.toDoubleList();
    std::vector<float> elements(value_list->elements().begin(),
                                value_list->elements().end());
    return xla::ConstantR1<float>(builder, elements);
  } else if (value.isBool()) {
    return xla::ConstantR0<bool>(builder, value.toBool());
  } else {
    XLA_ERROR() << "Unsupported constant: " << value;
  }
}

// Context class to hold together all the necessary state for the XLA
// computation building process out of a PyTorch graph.
class ComputationContext {
 public:
  static size_t OutputId(const torch::jit::Node* node) {
    const auto node_outputs = node->outputs();
    XLA_CHECK_EQ(node_outputs.size(), 1)
        << node->kind().toDisplayString() << "\nGraph:\n"
        << node->owningGraph()->toString();
    return node_outputs[0]->unique();
  }

  void AddNodeOpById(size_t id, xla::XlaOp op) {
    const auto it_ok = node_xla_ops_.emplace(id, std::move(op));
    XLA_CHECK(it_ok.second) << "Duplicated IR node ID: " << id;
  }

  void AddNodeOp(const torch::jit::Node* node, xla::XlaOp op) {
    AddNodeOpById(OutputId(node), op);
  }

  void AddValueOp(const torch::jit::Value* value, xla::XlaOp op) {
    AddNodeOpById(value->unique(), std::move(op));
  }

  void AddInputOp(xla::XlaOp op) { input_ops_.push_back(std::move(op)); }

  void AddUndefinedInput(size_t index) { undefined_inputs_.insert(index); }

  void AddSizeOpResult(const torch::jit::Value* value,
                       const std::vector<xla::int64>& size_op_result) {
    const auto it_ok = size_op_values_.emplace(value->unique(), size_op_result);
    XLA_CHECK(it_ok.second)
        << "Duplicated at::aten::size id: " << value->uniqueName();
  }

  const xla::XlaOp& GetOpForValue(const torch::jit::Value* value) const {
    auto it = node_xla_ops_.find(value->unique());
    XLA_CHECK(it != node_xla_ops_.end()) << value->uniqueName() << "\nGraph:\n"
                                         << value->owningGraph()->toString();
    return it->second;
  }

  c10::optional<xla::XlaOp> GetOpForInput(const torch::jit::Node* node,
                                          size_t input_index) const {
    const auto node_inputs = node->inputs();
    const auto input = node_inputs.at(input_index);
    // Check if is at::prim::Undefined.
    if (undefined_inputs_.count(input->unique()) > 0) {
      return at::nullopt;
    }
    // Check in constructed xla ops.
    auto it = node_xla_ops_.find(input->unique());
    if (it == node_xla_ops_.end()) {
      return at::nullopt;
    }
    return it->second;
  }

  xla::XlaOp OpForInput(const torch::jit::Node* node,
                        size_t input_index) const {
    auto op = GetOpForInput(node, input_index);
    if (!op) {
      const auto node_inputs = node->inputs();
      const auto input = node_inputs.at(input_index);
      XLA_ERROR() << "Missing op for input: unique_name=" << input->uniqueName()
                  << " kind=" << node->kind().toDisplayString() << "\nGraph:\n"
                  << node->owningGraph()->toString() << "\n"
                  << tensorflow::CurrentStackTrace();
    }
    return *op;
  }

  std::vector<xla::XlaOp> ReleaseInputs() { return std::move(input_ops_); }

  size_t GetInputsSize() const { return input_ops_.size(); }

  const XlaComputationInOut::SizeOpValues& GetSizeOpValues() const {
    return size_op_values_;
  }

  c10::optional<std::vector<xla::int64>> GetSizeOpValueForId(
      const size_t id) const {
    const auto it = size_op_values_.find(id);
    if (it != size_op_values_.end()) {
      return it->second;
    }
    return c10::nullopt;
  }

  const std::unordered_map<size_t, xla::XlaOp>& GetNodeOps() const {
    return node_xla_ops_;
  }

  const std::unordered_set<size_t>& GetUndefinedInputs() const {
    return undefined_inputs_;
  }

 private:
  std::vector<xla::XlaOp> input_ops_;
  std::unordered_map<size_t, xla::XlaOp> node_xla_ops_;
  std::unordered_set<size_t> undefined_inputs_;
  XlaComputationInOut::SizeOpValues size_op_values_;
};

// Definitions for the translation handlers per node kind.

void TranslateArithmetic(const torch::jit::Node* node, ComputationContext* cctx,
                         xla::PrecisionConfig::Precision /*conv_precision*/,
                         xla::XlaBuilder* b) {
  const auto node_inputs = node->inputs();
  if (node_inputs.size() < 2) {
    XLA_ERROR() << "Unsupported arity for binary operator "
                << node->kind().toQualString();
  }
  xla::XlaOp input_op_0 = cctx->OpForInput(node, 0);
  auto input_op_1_optional = cctx->GetOpForInput(node, 1);
  if (!input_op_1_optional) {
    xla::Shape input_op_0_shape = XlaHelpers::ShapeOfXlaOp(input_op_0);
    input_op_1_optional = XlaHelpers::ScalarValue(
        node->get<at::Scalar>(at::attr::other).value().to<float>(),
        input_op_0_shape.element_type(), b);
  }
  xla::XlaOp xla_output =
      BuildArithmeticOp(node, input_op_0, *input_op_1_optional);
  cctx->AddNodeOp(node, xla_output);
}

void TranslateComparison(const torch::jit::Node* node, ComputationContext* cctx,
                         xla::PrecisionConfig::Precision /*conv_precision*/,
                         xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), 2);
  xla::XlaOp xla_output = BuildComparisonOp(node, cctx->OpForInput(node, 0));
  cctx->AddNodeOp(node, xla_output);
}

void TranslateTypeAs(const torch::jit::Node* node, ComputationContext* cctx,
                     xla::PrecisionConfig::Precision /*conv_precision*/,
                     xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), 2);
  xla::XlaOp xla_output = BuildTypeAs(node, cctx->OpForInput(node, 0));
  cctx->AddNodeOp(node, xla_output);
}

void TranslateConvolution(const torch::jit::Node* node,
                          ComputationContext* cctx,
                          xla::PrecisionConfig::Precision conv_precision,
                          xla::XlaBuilder* /*b*/) {
  XLA_CHECK_GE(node->inputs().size(), 3);

  xla::XlaOp xla_output;
  auto opt_op = cctx->GetOpForInput(node, 3);
  if (opt_op) {  // bias exists
    xla_output = BuildConvolutionBias(node, cctx->OpForInput(node, 0),
                                      cctx->OpForInput(node, 1), *opt_op,
                                      conv_precision);
  } else {
    xla_output = BuildConvolution(node, cctx->OpForInput(node, 0),
                                  cctx->OpForInput(node, 1), conv_precision);
  }
  cctx->AddNodeOp(node, xla_output);
}

void TranslateConvolutionBackward(
    const torch::jit::Node* node, ComputationContext* cctx,
    xla::PrecisionConfig::Precision conv_precision, xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), 9);
  Conv2DGrads conv2d_grads = BuildConv2dBackward(
      node, cctx->OpForInput(node, 0), cctx->OpForInput(node, 1),
      cctx->OpForInput(node, 2), conv_precision);
  const auto node_outputs = node->outputs();
  cctx->AddValueOp(node_outputs[0], conv2d_grads.grad_input);
  cctx->AddValueOp(node_outputs[1], conv2d_grads.grad_weight);
  cctx->AddValueOp(node_outputs[2], conv2d_grads.grad_bias);
}

void TranslateTranspose(const torch::jit::Node* node, ComputationContext* cctx,
                        xla::PrecisionConfig::Precision /*conv_precision*/,
                        xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), 1);
  xla::XlaOp xla_output = xla::Transpose(cctx->OpForInput(node, 0), {1, 0});
  cctx->AddNodeOp(node, xla_output);
}

void TranslateAddMatMul(const torch::jit::Node* node, ComputationContext* cctx,
                        xla::PrecisionConfig::Precision conv_precision,
                        xla::XlaBuilder* /*b*/) {
  XLA_CHECK_GE(node->inputs().size(), 3);
  xla::PrecisionConfig precision_config =
      XlaHelpers::BuildPrecisionConfig(conv_precision);
  xla::XlaOp xla_output =
      xla::Dot(cctx->OpForInput(node, 1), cctx->OpForInput(node, 2),
               &precision_config) +
      cctx->OpForInput(node, 0);
  cctx->AddNodeOp(node, xla_output);
}

void TranslateMatMul(const torch::jit::Node* node, ComputationContext* cctx,
                     xla::PrecisionConfig::Precision conv_precision,
                     xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), 2);
  xla::PrecisionConfig precision_config =
      XlaHelpers::BuildPrecisionConfig(conv_precision);
  xla::XlaOp xla_output = xla::Dot(
      cctx->OpForInput(node, 0), cctx->OpForInput(node, 1), &precision_config);
  cctx->AddNodeOp(node, xla_output);
}

void TranslateMaxPool(const torch::jit::Node* node, ComputationContext* cctx,
                      xla::PrecisionConfig::Precision /*conv_precision*/,
                      xla::XlaBuilder* /*b*/) {
  XLA_CHECK_GE(node->inputs().size(), 1);
  XLA_CHECK_GE(node->outputs().size(), 1);
  xla::XlaOp xla_output = BuildMaxPool2d(node, cctx->OpForInput(node, 0));
  const auto node_outputs = node->outputs();
  XLA_CHECK_GE(node_outputs.size(), 1);
  cctx->AddValueOp(node_outputs[0], xla_output);
}

void TranslateMaxPoolBackward(
    const torch::jit::Node* node, ComputationContext* cctx,
    xla::PrecisionConfig::Precision /*conv_precision*/,
    xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), 8);
  xla::XlaOp xla_output = BuildMaxPool2dBackward(
      node, cctx->OpForInput(node, 0), cctx->OpForInput(node, 1));
  cctx->AddNodeOp(node, xla_output);
}

void TranslateAvgPool(const torch::jit::Node* node, ComputationContext* cctx,
                      xla::PrecisionConfig::Precision /*conv_precision*/,
                      xla::XlaBuilder* /*b*/) {
  XLA_CHECK_GE(node->inputs().size(), 1);
  xla::XlaOp xla_output = BuildAvgPool2d(node, cctx->OpForInput(node, 0));
  cctx->AddNodeOp(node, xla_output);
}

void TranslateAvgPoolBackward(
    const torch::jit::Node* node, ComputationContext* cctx,
    xla::PrecisionConfig::Precision /*conv_precision*/,
    xla::XlaBuilder* /*b*/) {
  XLA_CHECK_GE(node->inputs().size(), 2);
  xla::XlaOp xla_output = BuildAvgPool2dBackward(
      node, cctx->OpForInput(node, 0), cctx->OpForInput(node, 1));
  cctx->AddNodeOp(node, xla_output);
}

void TranslateAdaptiveAvgPool(
    const torch::jit::Node* node, ComputationContext* cctx,
    xla::PrecisionConfig::Precision /*conv_precision*/,
    xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), 2);
  xla::XlaOp xla_output =
      BuildAdaptiveAvgPool2d(node, cctx->OpForInput(node, 0));
  cctx->AddNodeOp(node, xla_output);
}

void TranslateAdaptiveAvgPoolBackward(
    const torch::jit::Node* node, ComputationContext* cctx,
    xla::PrecisionConfig::Precision /*conv_precision*/,
    xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), 2);
  xla::XlaOp xla_output = BuildAdaptiveAvgPool2dBackward(
      cctx->OpForInput(node, 0), cctx->OpForInput(node, 1));
  cctx->AddNodeOp(node, xla_output);
}

void TranslateSqrt(const torch::jit::Node* node, ComputationContext* cctx,
                   xla::PrecisionConfig::Precision /*conv_precision*/,
                   xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), 1);
  xla::XlaOp xla_input = cctx->OpForInput(node, 0);
  xla::XlaOp xla_output = xla::Sqrt(xla_input);
  cctx->AddNodeOp(node, xla_output);
}

void TranslateRsqrt(const torch::jit::Node* node, ComputationContext* cctx,
                    xla::PrecisionConfig::Precision /*conv_precision*/,
                    xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), 1);
  xla::XlaOp xla_input = cctx->OpForInput(node, 0);
  xla::XlaOp xla_output = xla::Rsqrt(xla_input);
  cctx->AddNodeOp(node, xla_output);
}

void TranslateNeg(const torch::jit::Node* node, ComputationContext* cctx,
                  xla::PrecisionConfig::Precision /*conv_precision*/,
                  xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), 1);
  xla::XlaOp xla_input = cctx->OpForInput(node, 0);
  xla::XlaOp xla_output = xla::Neg(xla_input);
  cctx->AddNodeOp(node, xla_output);
}

void TranslateTanh(const torch::jit::Node* node, ComputationContext* cctx,
                   xla::PrecisionConfig::Precision /*conv_precision*/,
                   xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), 1);
  xla::XlaOp xla_input = cctx->OpForInput(node, 0);
  xla::XlaOp xla_output = xla::Tanh(xla_input);
  cctx->AddNodeOp(node, xla_output);
}

void TranslateSigmoid(const torch::jit::Node* node, ComputationContext* cctx,
                      xla::PrecisionConfig::Precision /*conv_precision*/,
                      xla::XlaBuilder* b) {
  XLA_CHECK_EQ(node->inputs().size(), 1);
  xla::XlaOp xla_input = cctx->OpForInput(node, 0);
  cctx->AddNodeOp(node, BuildSigmoid(xla_input));
}

void TranslateRelu(const torch::jit::Node* node, ComputationContext* cctx,
                   xla::PrecisionConfig::Precision /*conv_precision*/,
                   xla::XlaBuilder* b) {
  XLA_CHECK_EQ(node->inputs().size(), 1);
  xla::XlaOp xla_input = cctx->OpForInput(node, 0);
  xla::XlaOp xla_output = BuildRelu(xla_input);
  cctx->AddNodeOp(node, xla_output);
}

void TranslateThreshold(const torch::jit::Node* node, ComputationContext* cctx,
                        xla::PrecisionConfig::Precision /*conv_precision*/,
                        xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), 3);
  xla::XlaOp xla_output = BuildThreshold(
      cctx->OpForInput(node, 0), cctx->OpForInput(node, 0),
      node->get<at::Scalar>(at::attr::threshold).value().to<float>(),
      node->get<at::Scalar>(at::attr::value).value().to<float>());
  cctx->AddNodeOp(node, xla_output);
}

void TranslateThresholdBackward(
    const torch::jit::Node* node, ComputationContext* cctx,
    xla::PrecisionConfig::Precision /*conv_precision*/,
    xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), 3);
  xla::XlaOp xla_output = BuildThreshold(
      cctx->OpForInput(node, 1), cctx->OpForInput(node, 0),
      node->get<at::Scalar>(at::attr::threshold).value().to<float>(), 0);
  cctx->AddNodeOp(node, xla_output);
}

void TranslateLogSoftmax(const torch::jit::Node* node, ComputationContext* cctx,
                         xla::PrecisionConfig::Precision /*conv_precision*/,
                         xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), size_t(2));
  xla::XlaOp xla_output = BuildLogSoftmax(node, cctx->OpForInput(node, 0));
  cctx->AddNodeOp(node, xla_output);
}

void TranslateLogSoftmaxBackward(
    const torch::jit::Node* node, ComputationContext* cctx,
    xla::PrecisionConfig::Precision /*conv_precision*/,
    xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), 4);
  xla::XlaOp xla_output = BuildLogSoftmaxGrad(node, cctx->OpForInput(node, 0),
                                              cctx->OpForInput(node, 1));
  cctx->AddNodeOp(node, xla_output);
}

void TranslateView(const torch::jit::Node* node, ComputationContext* cctx,
                   xla::PrecisionConfig::Precision /*conv_precision*/,
                   xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), 2);
  xla::XlaOp xla_output = BuildView(node, cctx->OpForInput(node, 0));
  cctx->AddNodeOp(node, xla_output);
}

void TranslateReshape(const torch::jit::Node* node, ComputationContext* cctx,
                      xla::PrecisionConfig::Precision /*conv_precision*/,
                      xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), 2);
  xla::XlaOp xla_output =
      BuildReshape(node, cctx->OpForInput(node, 0), cctx->GetSizeOpValues());
  cctx->AddNodeOp(node, xla_output);
}

void TranslateExpand(const torch::jit::Node* node, ComputationContext* cctx,
                     xla::PrecisionConfig::Precision /*conv_precision*/,
                     xla::XlaBuilder* /*b*/) {
  XLA_CHECK_GE(node->inputs().size(), 1);
  xla::XlaOp xla_output = BuildExpand(node, cctx->OpForInput(node, 0));
  cctx->AddNodeOp(node, xla_output);
}

void TranslateStack(const torch::jit::Node* node, ComputationContext* cctx,
                    xla::PrecisionConfig::Precision /*conv_precision*/,
                    xla::XlaBuilder* b) {
  XLA_CHECK_EQ(node->inputs().size(), 2);
  xla::XlaOp xla_output = BuildStack(
      node,
      [cctx](const torch::jit::Value* node) -> xla::XlaOp {
        return cctx->GetOpForValue(node);
      },
      b);
  cctx->AddNodeOp(node, xla_output);
}

void TranslateCat(const torch::jit::Node* node, ComputationContext* cctx,
                  xla::PrecisionConfig::Precision /*conv_precision*/,
                  xla::XlaBuilder* b) {
  XLA_CHECK_EQ(node->inputs().size(), 2);
  xla::XlaOp xla_output = BuildCat(
      node,
      [cctx](const torch::jit::Value* node) -> xla::XlaOp {
        return cctx->GetOpForValue(node);
      },
      b);
  cctx->AddNodeOp(node, xla_output);
}

void TranslateChunk(const torch::jit::Node* node, ComputationContext* cctx,
                    xla::PrecisionConfig::Precision /*conv_precision*/,
                    xla::XlaBuilder* /*b*/) {
  std::vector<xla::XlaOp> xla_outputs =
      BuildChunk(node, cctx->OpForInput(node, 0));
  const auto node_outputs = node->outputs();
  for (size_t i = 0; i < node_outputs.size(); ++i) {
    cctx->AddValueOp(node_outputs[i], xla_outputs[i]);
  }
}

void TranslateBatchNorm(const torch::jit::Node* node, ComputationContext* cctx,
                        xla::PrecisionConfig::Precision /*conv_precision*/,
                        xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), 8);
  const float eps_value =
      node->get<at::Scalar>(at::attr::eps).value().to<float>();
  BatchNormOutput outputs =
      BuildBatchNorm(cctx->OpForInput(node, 0), cctx->OpForInput(node, 1),
                     cctx->OpForInput(node, 2), eps_value);
  const auto node_outputs = node->outputs();
  cctx->AddValueOp(node_outputs[0], outputs.output);
  if (node->kind() == at::aten::batch_norm) {
    XLA_CHECK_EQ(node->outputs().size(), 1);
  }
  // at::aten::batch_norm only has 1 output
  // native_batch_norm_forward has output, save_mean, save_std
  if (node->kind() == at::aten::native_batch_norm) {
    cctx->AddValueOp(node_outputs[1], outputs.save_mean);
    cctx->AddValueOp(node_outputs[2], outputs.save_invstd_eps);
  }
}

void TranslateBatchNormBackward(
    const torch::jit::Node* node, ComputationContext* cctx,
    xla::PrecisionConfig::Precision /*conv_precision*/,
    xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), 10);
  const float eps_value =
      node->get<at::Scalar>(at::attr::eps).value().to<float>();
  BatchNormGrads grads =
      BuildBatchNormBackward(cctx->OpForInput(node, 0),  // grad_output
                             cctx->OpForInput(node, 1),  // input
                             cctx->OpForInput(node, 2),  // weight
                             cctx->OpForInput(node, 5),  // save_mean
                             cctx->OpForInput(node, 6),  // save_std
                             eps_value);                 // eps_value
  const auto node_outputs = node->outputs();
  cctx->AddValueOp(node_outputs[0], grads.grad_input);
  cctx->AddValueOp(node_outputs[1], grads.grad_weight);
  cctx->AddValueOp(node_outputs[2], grads.grad_bias);
}

void TranslateSum(const torch::jit::Node* node, ComputationContext* cctx,
                  xla::PrecisionConfig::Precision /*conv_precision*/,
                  xla::XlaBuilder* /*b*/) {
  XLA_CHECK_GE(node->inputs().size(), 1);
  xla::XlaOp xla_output = BuildSum(node, cctx->OpForInput(node, 0));
  cctx->AddNodeOp(node, xla_output);
}

void TranslateNllLoss(const torch::jit::Node* node, ComputationContext* cctx,
                      xla::PrecisionConfig::Precision /*conv_precision*/,
                      xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), 5);
  xla::XlaOp xla_output =
      BuildNllLoss(cctx->OpForInput(node, 0), cctx->OpForInput(node, 1));
  cctx->AddNodeOp(node, xla_output);
}

void TranslateNllLossBackward(
    const torch::jit::Node* node, ComputationContext* cctx,
    xla::PrecisionConfig::Precision /*conv_precision*/,
    xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), 7);
  xla::XlaOp xla_output = BuildNllLossBackward(cctx->OpForInput(node, 1),
                                               cctx->OpForInput(node, 2));
  cctx->AddNodeOp(node, xla_output);
}

void TranslateSize(const torch::jit::Node* node, ComputationContext* cctx,
                   xla::PrecisionConfig::Precision /*conv_precision*/,
                   xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), 1);
  std::vector<xla::int64> size_op_result;
  xla::XlaOp xla_output =
      BuildSize(node, cctx->OpForInput(node, 0), &size_op_result);
  cctx->AddSizeOpResult(node->output(0), size_op_result);
  cctx->AddNodeOp(node, xla_output);
}

void TranslateConstant(const torch::jit::Node* node, ComputationContext* cctx,
                       xla::PrecisionConfig::Precision /*conv_precision*/,
                       xla::XlaBuilder* b) {
  cctx->AddNodeOp(node, GetConstantOp(b, node));
}

void TranslateUndefined(const torch::jit::Node* node, ComputationContext* cctx,
                        xla::PrecisionConfig::Precision /*conv_precision*/,
                        xla::XlaBuilder* /*b*/) {
  cctx->AddUndefinedInput(ComputationContext::OutputId(node));
}

void TranslateGradSumToSize(const torch::jit::Node* node,
                            ComputationContext* cctx,
                            xla::PrecisionConfig::Precision /*conv_precision*/,
                            xla::XlaBuilder* /*b*/) {
  XLA_CHECK_EQ(node->inputs().size(), 2);
  xla::XlaOp xla_output = BuildGradSumToSize(node, cctx->OpForInput(node, 0),
                                             cctx->GetSizeOpValues());
  cctx->AddNodeOp(node, xla_output);
}

void TranslateNop(const torch::jit::Node* /*node*/,
                  ComputationContext* /*cctx*/,
                  xla::PrecisionConfig::Precision /*conv_precision*/,
                  xla::XlaBuilder* /*b*/) {}

using TranslationHandler = std::function<void(
    const torch::jit::Node* node, ComputationContext* cctx,
    xla::PrecisionConfig::Precision conv_precision, xla::XlaBuilder* b)>;

std::unordered_map<torch::jit::NodeKind, TranslationHandler>*
CreateTranslationHandlers() {
  auto t = new std::unordered_map<torch::jit::NodeKind, TranslationHandler>();
  (*t)[at::aten::add] = TranslateArithmetic;
  (*t)[at::aten::div] = TranslateArithmetic;
  (*t)[at::aten::sub] = TranslateArithmetic;
  (*t)[at::aten::mul] = TranslateArithmetic;
  (*t)[at::aten::ne] = TranslateComparison;
  (*t)[at::aten::eq] = TranslateComparison;
  (*t)[at::aten::ge] = TranslateComparison;
  (*t)[at::aten::le] = TranslateComparison;
  (*t)[at::aten::gt] = TranslateComparison;
  (*t)[at::aten::lt] = TranslateComparison;
  (*t)[at::aten::type_as] = TranslateTypeAs;
  (*t)[at::aten::convolution] = TranslateConvolution;
  (*t)[at::aten::thnn_conv2d_forward] = TranslateConvolution;
  (*t)[at::aten::thnn_conv2d_backward] = TranslateConvolutionBackward;
  (*t)[at::aten::t] = TranslateTranspose;
  (*t)[at::aten::addmm] = TranslateAddMatMul;
  (*t)[at::aten::mm] = TranslateMatMul;
  (*t)[at::aten::max_pool2d_with_indices] = TranslateMaxPool;
  (*t)[at::aten::max_pool2d_with_indices_backward] = TranslateMaxPoolBackward;
  (*t)[at::aten::avg_pool2d] = TranslateAvgPool;
  (*t)[at::aten::avg_pool2d_backward] = TranslateAvgPoolBackward;
  (*t)[at::aten::adaptive_avg_pool2d] = TranslateAdaptiveAvgPool;
  // TODO: Switch to normal form once the string is preoperly externalized.
  (*t)[torch::jit::NodeKind::fromQualString(
      "aten::_adaptive_avg_pool2d_backward")] =
      TranslateAdaptiveAvgPoolBackward;
  (*t)[at::aten::sqrt] = TranslateSqrt;
  (*t)[at::aten::rsqrt] = TranslateRsqrt;
  (*t)[at::aten::neg] = TranslateNeg;
  (*t)[at::aten::tanh] = TranslateTanh;
  (*t)[at::aten::sigmoid] = TranslateSigmoid;
  (*t)[at::aten::relu] = TranslateRelu;
  (*t)[at::aten::threshold] = TranslateThreshold;
  (*t)[at::aten::threshold_backward] = TranslateThresholdBackward;
  (*t)[at::aten::log_softmax] = TranslateLogSoftmax;
  (*t)[at::aten::_log_softmax_backward_data] = TranslateLogSoftmaxBackward;
  (*t)[at::aten::reshape] = TranslateReshape;
  (*t)[at::aten::view] = TranslateView;
  (*t)[at::aten::expand] = TranslateExpand;
  (*t)[at::aten::stack] = TranslateStack;
  (*t)[at::aten::cat] = TranslateCat;
  (*t)[at::aten::chunk] = TranslateChunk;
  (*t)[at::aten::native_batch_norm] = TranslateBatchNorm;
  (*t)[at::aten::batch_norm] = TranslateBatchNorm;
  (*t)[at::aten::native_batch_norm_backward] = TranslateBatchNormBackward;
  (*t)[at::aten::sum] = TranslateSum;
  (*t)[at::aten::nll_loss] = TranslateNllLoss;
  (*t)[at::aten::nll_loss_backward] = TranslateNllLossBackward;
  (*t)[at::aten::size] = TranslateSize;
  (*t)[at::prim::Constant] = TranslateConstant;
  (*t)[at::prim::Undefined] = TranslateUndefined;
  (*t)[at::prim::None] = TranslateUndefined;
  (*t)[at::aten::_grad_sum_to_size] = TranslateGradSumToSize;
  (*t)[at::prim::ListConstruct] = TranslateNop;
  return t;
}

std::unordered_map<torch::jit::NodeKind, TranslationHandler>*
GetTranslationHandlers() {
  static auto translation_handlers = CreateTranslationHandlers();
  return translation_handlers;
}

}  // namespace

XlaTranslator::XlaTranslator(
    const std::shared_ptr<torch::jit::Graph>& graph,
    const xla::PrecisionConfig::Precision conv_precision)
    : graph_(graph), conv_precision_(conv_precision) {}

XlaTranslationResult XlaTranslator::BuildComputation(
    const std::string& name,
    const std::vector<ParameterShape>& parameter_shapes,
    const XlaComputationInOut::SizeOpValues& param_size_op_values,
    const BuildOptions& options) const {
  xla::XlaBuilder b(name);
  XlaComputationInOut computation_program =
      BuildComputationProgram(parameter_shapes, param_size_op_values, &b);
  if (options.output_transform) {
    for (size_t i = 0; i < computation_program.outputs.size(); ++i) {
      computation_program.outputs[i] =
          options.output_transform(computation_program.outputs[i], i);
    }
  }
  XlaHelpers::CreateReturnValue(&b, computation_program.outputs);
  return {b.Build().ValueOrDie(), computation_program.ret_size_op_values};
}

XlaComputationInOut XlaTranslator::BuildComputationProgram(
    const std::vector<ParameterShape>& parameter_shapes,
    const XlaComputationInOut::SizeOpValues& param_size_op_values,
    xla::XlaBuilder* b) const {
  ComputationContext cctx;
  const auto graph_inputs = graph_->inputs();
  XLA_CHECK_EQ(graph_inputs.size(), parameter_shapes.size())
      << "Graph:\n"
      << graph_->toString();
  // Add graph inputs to the context.
  for (size_t parameter_number = 0; parameter_number < graph_inputs.size();
       ++parameter_number) {
    torch::jit::Value* graph_input = graph_inputs[parameter_number];
    if (parameter_shapes[parameter_number].kind == ParameterKind::kGraphInput) {
      size_t param_no = cctx.GetInputsSize();
      const auto parameter_op =
          xla::Parameter(b, param_no, parameter_shapes[parameter_number].shape,
                         "param_" + std::to_string(param_no));
      cctx.AddValueOp(graph_input, parameter_op);
      cctx.AddInputOp(parameter_op);
    } else if (parameter_shapes[parameter_number].kind ==
               ParameterKind::kZeroInput) {
      // The backward method of the model creates all-zeros grad outputs we
      // represent as XLATensor with no data and empty shape.
      cctx.AddValueOp(graph_input,
                      XlaHelpers::ScalarBroadcast<float>(
                          0, parameter_shapes[parameter_number].shape, b));
    }
    // Seed at::aten::size tracking info with the values in
    // param_size_op_values.
    const auto size_op_value_it = param_size_op_values.find(parameter_number);
    if (size_op_value_it != param_size_op_values.end()) {
      cctx.AddSizeOpResult(graph_input, size_op_value_it->second);
    }
  }
  // Translate the nodes in the graph.
  auto nodes = graph_->block()->nodes();
  for (auto node : nodes) {
    const auto translation_handler_it =
        GetTranslationHandlers()->find(node->kind());
    XLA_CHECK(translation_handler_it != GetTranslationHandlers()->end())
        << "Unsupported operator: " << node->kind().toQualString();
    translation_handler_it->second(node, &cctx, conv_precision_, b);
  }
  // Fill in the return tuple.
  const auto return_node = graph_->return_node();
  const auto node_inputs = return_node->inputs();
  if (return_node->kind() != at::prim::Return || node_inputs.empty()) {
    XLA_ERROR() << "Unexpected end of graph";
  }
  std::vector<xla::XlaOp> returned_tuple;
  XlaComputationInOut::SizeOpValues ret_size_op_values;
  for (size_t return_input_idx = 0; return_input_idx < node_inputs.size();
       ++return_input_idx) {
    const auto return_input = node_inputs[return_input_idx];
    const auto size_op_value_maybe =
        cctx.GetSizeOpValueForId(return_input->unique());
    // Add evaluated at::aten::size values to the return tuple.
    if (size_op_value_maybe) {
      const auto it_ok =
          ret_size_op_values.emplace(return_input_idx, *size_op_value_maybe);
      XLA_CHECK(it_ok.second)
          << "Duplicated return component index " << return_input_idx;
    }
    returned_tuple.push_back(cctx.GetOpForValue(return_input));
  }
  return XlaComputationInOut{cctx.ReleaseInputs(), std::move(returned_tuple),
                             std::move(ret_size_op_values)};
}

}  // namespace torch_xla
