#include "translator.h"
#include <memory>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include "batch_norm.h"
#include "convolution.h"
#include "data_ops.h"
#include "elementwise.h"
#include "helpers.h"
#include "log_softmax.h"
#include "nll_loss.h"
#include "pooling.h"
#include "reduction.h"
#include "tensor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"

namespace torch {
namespace jit {

namespace {

xla::ComputationClient* CreateClient() {
  return xla::ComputationClient::Create().ConsumeValueOrDie().release();
}

xla::XlaOp GetConstantOp(xla::XlaBuilder* builder, Node* node) {
  auto value = toIValue(node->output()).value();
  if (value.isTensor()) {
    auto literal = GetTensorLiteral(value.toTensor(), /*shape=*/nullptr);
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
  } else if (value.isDoubleList()) {
    auto value_list = value.toDoubleList();
    std::vector<float> elements(value_list->elements().begin(),
                                value_list->elements().end());
    return xla::ConstantR1<float>(builder, elements);
  } else if (value.isBool()) {
    return xla::ConstantR0<bool>(builder, value.toBool());
  } else {
    std::stringstream ss;
    ss << value;
    AT_ERROR("Unsupported constant: ", ss.str());
  }
}

// Context class to hold together all the necessary state for the XLA
// computation building process out of a PyTorch graph.
class ComputationContext {
 public:
  static size_t OutputId(const Node* node) {
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

  void AddNodeOp(const Node* node, xla::XlaOp op) {
    AddNodeOpById(OutputId(node), op);
  }

  void AddValueOp(const Value* value, xla::XlaOp op) {
    AddNodeOpById(value->unique(), std::move(op));
  }

  void AddInputOp(xla::XlaOp op) { input_ops_.push_back(std::move(op)); }

  void AddUndefinedInput(size_t index) { undefined_inputs_.insert(index); }

  const xla::XlaOp& GetOpForValue(const Value* value) const {
    auto it = node_xla_ops_.find(value->unique());
    XLA_CHECK(it != node_xla_ops_.end()) << value->uniqueName() << "\nGraph:\n"
                                         << value->owningGraph()->toString();
    return it->second;
  }

  c10::optional<xla::XlaOp> GetOpForInput(const Node* node,
                                          size_t input_index) const {
    const auto node_inputs = node->inputs();
    const auto input = node_inputs.at(input_index);
    // Check if is prim::Undefined.
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

  xla::XlaOp OpForInput(const Node* node, size_t input_index) const {
    auto op = GetOpForInput(node, input_index);
    if (!op) {
      const auto node_inputs = node->inputs();
      const auto input = node_inputs.at(input_index);
      TF_LOG(FATAL) << "Missing op for input: unique_name="
                    << input->uniqueName()
                    << " kind=" << node->kind().toDisplayString()
                    << "\nGraph:\n"
                    << node->owningGraph()->toString() << "\n"
                    << tensorflow::CurrentStackTrace();
    }
    return *op;
  }

  std::vector<xla::XlaOp> ReleaseInputs() { return std::move(input_ops_); }

  size_t GetInputsSize() const { return input_ops_.size(); }

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
};

}  // namespace

xla::ComputationClient* XlaGetClient() {
  static xla::ComputationClient* computation_client = CreateClient();
  return computation_client;
}

XlaTranslator::XlaTranslator(
    const std::shared_ptr<Graph>& graph,
    const xla::PrecisionConfig::Precision conv_precision)
    : graph_(graph), conv_precision_(conv_precision) {}

xla::XlaComputation XlaTranslator::BuildComputation(
    const std::string& name,
    const std::vector<ParameterShape>& parameter_shapes,
    const BuildOptions& options) const {
  xla::XlaBuilder b(name);
  auto returned_tuple = BuildComputationProgram(parameter_shapes, &b);
  if (options.output_transform) {
    for (size_t i = 0; i < returned_tuple.outputs.size(); ++i) {
      returned_tuple.outputs[i] =
          options.output_transform(returned_tuple.outputs[i], i);
    }
  }
  XlaHelpers::CreateReturnValue(&b, returned_tuple.outputs);
  return b.Build().ValueOrDie();
}

XlaComputationInOut XlaTranslator::BuildComputationProgram(
    const std::vector<ParameterShape>& parameter_shapes,
    xla::XlaBuilder* b) const {
  ComputationContext cctx;
  const auto graph_inputs = graph_->inputs();
  XLA_CHECK_EQ(graph_inputs.size(), parameter_shapes.size())
      << "Graph:\n"
      << graph_->toString();
  for (size_t parameter_number = 0; parameter_number < graph_inputs.size();
       ++parameter_number) {
    Value* graph_input = graph_inputs[parameter_number];
    if (parameter_shapes[parameter_number].kind == ParameterKind::kGraphInput) {
      auto param_no = cctx.GetInputsSize();
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
  }
  auto nodes = graph_->block()->nodes();
  for (auto node : nodes) {
    switch (node->kind()) {
      case aten::add:
      case aten::sub:
      case aten::mul: {
        const auto node_inputs = node->inputs();
        if (node_inputs.size() < 2) {
          AT_ERROR("Unsupported arity for binary operator ",
                   node->kind().toQualString());
        }
        xla::XlaOp input_op_0 = cctx.OpForInput(node, 0);
        auto input_op_1_optional = cctx.GetOpForInput(node, 1);
        if (!input_op_1_optional) {
          xla::Shape input_op_0_shape = XlaHelpers::ShapeOfXlaOp(input_op_0);
          input_op_1_optional = XlaHelpers::ScalarValue(
              node->get<at::Scalar>(attr::other).value().to<float>(),
              input_op_0_shape.element_type(), b);
        }
        auto inputs =
            XlaHelpers::PromoteValues(input_op_0, *input_op_1_optional);
        xla::XlaOp xla_output =
            BuildArithmeticOp(node, inputs.first, inputs.second);
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::gt: {
        if (node->inputs().size() != 2) {
          AT_ERROR("Unsupported arity for aten::gt");
        }
        xla::XlaOp xla_output =
            BuildComparisonOp(node, cctx.OpForInput(node, 0));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::type_as: {
        CHECK_EQ(node->inputs().size(), 2);
        xla::XlaOp xla_output = BuildTypeAs(node, cctx.OpForInput(node, 0));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::convolution:
      case aten::thnn_conv2d_forward: {
        if (node->inputs().size() < 3) {
          AT_ERROR("Unsupported number of inputs for convolution: ",
                   node->inputs().size());
        }

        xla::XlaOp xla_output;
        auto opt_op = cctx.GetOpForInput(node, 3);
        if (opt_op) {  // bias exists
          xla_output = BuildConvolutionBias(node, cctx.OpForInput(node, 0),
                                            cctx.OpForInput(node, 1), *opt_op,
                                            conv_precision_);
        } else {
          xla_output =
              BuildConvolution(node, cctx.OpForInput(node, 0),
                               cctx.OpForInput(node, 1), conv_precision_);
        }
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::thnn_conv2d_backward: {
        CHECK_EQ(node->inputs().size(), 9);
        const auto conv2d_grads = BuildConv2dBackward(
            node, cctx.OpForInput(node, 0), cctx.OpForInput(node, 1),
            cctx.OpForInput(node, 2), conv_precision_);
        const auto node_outputs = node->outputs();
        cctx.AddValueOp(node_outputs[0], conv2d_grads.grad_input);
        cctx.AddValueOp(node_outputs[1], conv2d_grads.grad_weight);
        cctx.AddValueOp(node_outputs[2], conv2d_grads.grad_bias);
        break;
      }
      case aten::t: {
        CHECK_EQ(node->inputs().size(), 1);
        xla::XlaOp xla_output =
            xla::Transpose(cctx.OpForInput(node, 0), {1, 0});
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::addmm: {
        if (node->inputs().size() < 3) {
          AT_ERROR("Unsupported number of inputs for linear layer: ",
                   node->inputs().size());
        }
        xla::PrecisionConfig precision_config =
            XlaHelpers::BuildPrecisionConfig(conv_precision_);
        xla::XlaOp xla_output =
            xla::Dot(cctx.OpForInput(node, 1), cctx.OpForInput(node, 2),
                     &precision_config) +
            cctx.OpForInput(node, 0);
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::mm: {
        CHECK_EQ(node->inputs().size(), 2);
        xla::PrecisionConfig precision_config =
            XlaHelpers::BuildPrecisionConfig(conv_precision_);
        xla::XlaOp xla_output =
            xla::Dot(cctx.OpForInput(node, 0), cctx.OpForInput(node, 1),
                     &precision_config);
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::max_pool2d_with_indices: {
        CHECK_GE(node->inputs().size(), 1);
        CHECK_GE(node->outputs().size(), 1);
        xla::XlaOp xla_output = BuildMaxPool2d(node, cctx.OpForInput(node, 0));
        const auto node_outputs = node->outputs();
        CHECK_GE(node_outputs.size(), 1);
        cctx.AddValueOp(node_outputs[0], xla_output);
        break;
      }
      case aten::max_pool2d_with_indices_backward: {
        CHECK_EQ(node->inputs().size(), 8);
        xla::XlaOp xla_output = BuildMaxPool2dBackward(
            node, cctx.OpForInput(node, 0), cctx.OpForInput(node, 1));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::avg_pool2d: {
        CHECK_GE(node->inputs().size(), 1);
        xla::XlaOp xla_output = BuildAvgPool2d(node, cctx.OpForInput(node, 0));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::avg_pool2d_backward: {
        CHECK_GE(node->inputs().size(), 2);
        xla::XlaOp xla_output = BuildAvgPool2dBackward(
            node, cctx.OpForInput(node, 0), cctx.OpForInput(node, 1));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::neg: {
        CHECK_EQ(node->inputs().size(), 1);
        const auto xla_input = cctx.OpForInput(node, 0);
        xla::XlaOp xla_output = Neg(xla_input);
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::tanh: {
        CHECK_EQ(node->inputs().size(), 1);
        const auto xla_input = cctx.OpForInput(node, 0);
        xla::XlaOp xla_output = Tanh(xla_input);
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::sigmoid: {
        CHECK_EQ(node->inputs().size(), 1);
        const auto xla_input = cctx.OpForInput(node, 0);
        xla::Shape xla_input_shape = XlaHelpers::ShapeOfXlaOp(xla_input);
        const auto half = XlaHelpers::ScalarValue<float>(
            0.5, xla_input_shape.element_type(), b);
        xla::XlaOp xla_output = half + half * Tanh(half * xla_input);
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::relu: {
        CHECK_EQ(node->inputs().size(), 1);
        const auto xla_input = cctx.OpForInput(node, 0);
        xla::Shape xla_input_shape = XlaHelpers::ShapeOfXlaOp(xla_input);
        xla::XlaOp xla_output =
            xla::Max(xla_input, XlaHelpers::ScalarValue<float>(
                                    0, xla_input_shape.element_type(), b));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::threshold: {
        CHECK_EQ(node->inputs().size(), 3);
        xla::XlaOp xla_output = BuildThreshold(
            node, cctx.OpForInput(node, 0), cctx.OpForInput(node, 0),
            node->get<at::Scalar>(attr::threshold).value().to<float>(),
            node->get<at::Scalar>(attr::value).value().to<float>(), b);
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::threshold_backward: {
        CHECK_EQ(node->inputs().size(), 3);
        xla::XlaOp xla_output = BuildThreshold(
            node, cctx.OpForInput(node, 1), cctx.OpForInput(node, 0),
            node->get<at::Scalar>(attr::threshold).value().to<float>(), 0, b);
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::log_softmax: {
        CHECK_EQ(node->inputs().size(), size_t(2));
        xla::XlaOp xla_output = BuildLogSoftmax(node, cctx.OpForInput(node, 0));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::_log_softmax_backward_data: {
        CHECK_EQ(node->inputs().size(), 4);
        xla::XlaOp xla_output = BuildLogSoftmaxGrad(
            node, cctx.OpForInput(node, 0), cctx.OpForInput(node, 1));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::reshape:
      case aten::view: {
        CHECK_EQ(node->inputs().size(), 2);
        xla::XlaOp xla_output = BuildView(node, cctx.OpForInput(node, 0));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::expand: {
        CHECK_GE(node->inputs().size(), 1);
        xla::XlaOp xla_output = BuildExpand(node, cctx.OpForInput(node, 0));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::stack: {
        CHECK_EQ(node->inputs().size(), 2);
        xla::XlaOp xla_output =
            BuildStack(node,
                       [&cctx](const Value* node) -> xla::XlaOp {
                         return cctx.GetOpForValue(node);
                       },
                       b);
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::cat: {
        CHECK_EQ(node->inputs().size(), 2);
        xla::XlaOp xla_output =
            BuildCat(node,
                     [&cctx](const Value* node) -> xla::XlaOp {
                       return cctx.GetOpForValue(node);
                     },
                     b);
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::chunk: {
        std::vector<xla::XlaOp> xla_outputs =
            BuildChunk(node, cctx.OpForInput(node, 0));
        const auto node_outputs = node->outputs();
        for (size_t i = 0; i < node_outputs.size(); ++i) {
          cctx.AddValueOp(node_outputs[i], xla_outputs[i]);
        }
        break;
      }
      case aten::native_batch_norm:
      case aten::batch_norm: {
        CHECK_EQ(node->inputs().size(), 8);
        const auto outputs =
            BuildBatchNorm(node, cctx.OpForInput(node, 0),
                           cctx.OpForInput(node, 1), cctx.OpForInput(node, 2));
        const auto node_outputs = node->outputs();
        cctx.AddValueOp(node_outputs[0], outputs.output);
        if (node->kind() == aten::batch_norm) {
          CHECK_EQ(node->outputs().size(), 1);
        }
        // aten::batch_norm only has 1 output
        // native_batch_norm_forward has output, save_mean, save_std
        if (node->kind() == aten::native_batch_norm) {
          cctx.AddValueOp(node_outputs[1], outputs.save_mean);
          cctx.AddValueOp(node_outputs[2], outputs.save_invstd_eps);
        }
        break;
      }
      case aten::native_batch_norm_backward: {
        CHECK_EQ(node->inputs().size(), 10);
        auto grads = BuildBatchNormBackward(
            node, cctx.OpForInput(node, 0),  // grad_output
            cctx.OpForInput(node, 1),        // input
            cctx.OpForInput(node, 2),        // weight
            cctx.OpForInput(node, 5),        // save_mean
            cctx.OpForInput(node, 6));       // save_std
        const auto node_outputs = node->outputs();
        cctx.AddValueOp(node_outputs[0], grads.grad_input);
        cctx.AddValueOp(node_outputs[1], grads.grad_weight);
        cctx.AddValueOp(node_outputs[2], grads.grad_bias);
        break;
      }
      case aten::sum: {
        CHECK_GE(node->inputs().size(), 1);
        xla::XlaOp xla_output = BuildSum(node, cctx.OpForInput(node, 0));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::nll_loss: {
        CHECK_EQ(node->inputs().size(), 5);
        xla::XlaOp xla_output = BuildNllLoss(node, cctx.OpForInput(node, 0),
                                             cctx.OpForInput(node, 1));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::nll_loss_backward: {
        CHECK_EQ(node->inputs().size(), 7);
        xla::XlaOp xla_output = BuildNllLossBackward(
            node, cctx.OpForInput(node, 1), cctx.OpForInput(node, 2));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case prim::Constant: {
        cctx.AddNodeOp(node, GetConstantOp(b, node));
        break;
      }
      case prim::ListConstruct: {
        break;
      }
      case prim::Undefined: {
        cctx.AddUndefinedInput(ComputationContext::OutputId(node));
        break;
      }
      default:
        AT_ERROR("Unsupported operator: ", node->kind().toQualString());
    }
  }
  const auto return_node = graph_->return_node();
  const auto node_inputs = return_node->inputs();
  // TODO: tighten the id check for returned tuples.
  if (return_node->kind() != prim::Return || node_inputs.empty()) {
    AT_ERROR("Unexpected end of graph");
  }
  std::vector<xla::XlaOp> returned_tuple;
  for (const auto return_input : node_inputs) {
    returned_tuple.push_back(cctx.GetOpForValue(return_input));
  }
  return XlaComputationInOut{cctx.ReleaseInputs(), returned_tuple};
}

}  // namespace jit
}  // namespace torch
