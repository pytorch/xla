#include "translator.h"
#include <memory>
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
#include "size_ops.h"
#include "tensor.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
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
    XLA_ERROR() << "Unsupported constant: " << value;
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

  void AddSizeOpResult(const Value* value,
                       const std::vector<xla::int64>& size_op_result) {
    const auto it_ok = size_op_values_.emplace(value->unique(), size_op_result);
    XLA_CHECK(it_ok.second)
        << "Duplicated aten::size id: " << value->uniqueName();
  }

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

}  // namespace

xla::ComputationClient* XlaGetClient() {
  static xla::ComputationClient* computation_client = CreateClient();
  return computation_client;
}

XlaTranslator::XlaTranslator(
    const std::shared_ptr<Graph>& graph,
    const xla::PrecisionConfig::Precision conv_precision)
    : graph_(graph), conv_precision_(conv_precision) {}

XlaTranslationResult XlaTranslator::BuildComputation(
    const std::string& name,
    const std::vector<ParameterShape>& parameter_shapes,
    const XlaComputationInOut::SizeOpValues& param_size_op_values,
    const BuildOptions& options) const {
  xla::XlaBuilder b(name);
  auto computation_program =
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
    // Seed aten::size tracking info with the values in param_size_op_values.
    const auto size_op_value_it = param_size_op_values.find(parameter_number);
    if (size_op_value_it != param_size_op_values.end()) {
      cctx.AddSizeOpResult(graph_input, size_op_value_it->second);
    }
  }
  auto nodes = graph_->block()->nodes();
  for (auto node : nodes) {
    switch (node->kind()) {
      case aten::add:
      case aten::div:
      case aten::sub:
      case aten::mul: {
        const auto node_inputs = node->inputs();
        if (node_inputs.size() < 2) {
          XLA_ERROR() << "Unsupported arity for binary operator "
                      << node->kind().toQualString();
        }
        xla::XlaOp input_op_0 = cctx.OpForInput(node, 0);
        auto input_op_1_optional = cctx.GetOpForInput(node, 1);
        if (!input_op_1_optional) {
          xla::Shape input_op_0_shape = XlaHelpers::ShapeOfXlaOp(input_op_0);
          input_op_1_optional = XlaHelpers::ScalarValue(
              node->get<at::Scalar>(attr::other).value().to<float>(),
              input_op_0_shape.element_type(), b);
        }
        xla::XlaOp xla_output =
            BuildArithmeticOp(node, input_op_0, *input_op_1_optional);
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::gt: {
        XLA_CHECK_EQ(node->inputs().size(), 2);
        xla::XlaOp xla_output =
            BuildComparisonOp(node, cctx.OpForInput(node, 0));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::type_as: {
        XLA_CHECK_EQ(node->inputs().size(), 2);
        xla::XlaOp xla_output = BuildTypeAs(node, cctx.OpForInput(node, 0));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::convolution:
      case aten::thnn_conv2d_forward: {
        XLA_CHECK_GE(node->inputs().size(), 3);

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
        XLA_CHECK_EQ(node->inputs().size(), 9);
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
        XLA_CHECK_EQ(node->inputs().size(), 1);
        xla::XlaOp xla_output =
            xla::Transpose(cctx.OpForInput(node, 0), {1, 0});
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::addmm: {
        XLA_CHECK_GE(node->inputs().size(), 3);
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
        XLA_CHECK_EQ(node->inputs().size(), 2);
        xla::PrecisionConfig precision_config =
            XlaHelpers::BuildPrecisionConfig(conv_precision_);
        xla::XlaOp xla_output =
            xla::Dot(cctx.OpForInput(node, 0), cctx.OpForInput(node, 1),
                     &precision_config);
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::max_pool2d_with_indices: {
        XLA_CHECK_GE(node->inputs().size(), 1);
        XLA_CHECK_GE(node->outputs().size(), 1);
        xla::XlaOp xla_output = BuildMaxPool2d(node, cctx.OpForInput(node, 0));
        const auto node_outputs = node->outputs();
        XLA_CHECK_GE(node_outputs.size(), 1);
        cctx.AddValueOp(node_outputs[0], xla_output);
        break;
      }
      case aten::max_pool2d_with_indices_backward: {
        XLA_CHECK_EQ(node->inputs().size(), 8);
        xla::XlaOp xla_output = BuildMaxPool2dBackward(
            node, cctx.OpForInput(node, 0), cctx.OpForInput(node, 1));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::avg_pool2d: {
        XLA_CHECK_GE(node->inputs().size(), 1);
        xla::XlaOp xla_output = BuildAvgPool2d(node, cctx.OpForInput(node, 0));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::avg_pool2d_backward: {
        XLA_CHECK_GE(node->inputs().size(), 2);
        xla::XlaOp xla_output = BuildAvgPool2dBackward(
            node, cctx.OpForInput(node, 0), cctx.OpForInput(node, 1));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::sqrt: {
        XLA_CHECK_EQ(node->inputs().size(), 1);
        const auto xla_input = cctx.OpForInput(node, 0);
        xla::XlaOp xla_output = xla::Sqrt(xla_input);
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::rsqrt: {
        XLA_CHECK_EQ(node->inputs().size(), 1);
        const auto xla_input = cctx.OpForInput(node, 0);
        xla::XlaOp xla_output = xla::Rsqrt(xla_input);
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::neg: {
        XLA_CHECK_EQ(node->inputs().size(), 1);
        const auto xla_input = cctx.OpForInput(node, 0);
        xla::XlaOp xla_output = xla::Neg(xla_input);
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::tanh: {
        XLA_CHECK_EQ(node->inputs().size(), 1);
        const auto xla_input = cctx.OpForInput(node, 0);
        xla::XlaOp xla_output = xla::Tanh(xla_input);
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::sigmoid: {
        XLA_CHECK_EQ(node->inputs().size(), 1);
        const auto xla_input = cctx.OpForInput(node, 0);
        xla::Shape xla_input_shape = XlaHelpers::ShapeOfXlaOp(xla_input);
        const auto half = XlaHelpers::ScalarValue<float>(
            0.5, xla_input_shape.element_type(), b);
        xla::XlaOp xla_output = half + half * xla::Tanh(half * xla_input);
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::relu: {
        XLA_CHECK_EQ(node->inputs().size(), 1);
        const auto xla_input = cctx.OpForInput(node, 0);
        xla::Shape xla_input_shape = XlaHelpers::ShapeOfXlaOp(xla_input);
        xla::XlaOp xla_output =
            xla::Max(xla_input, XlaHelpers::ScalarValue<float>(
                                    0, xla_input_shape.element_type(), b));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::threshold: {
        XLA_CHECK_EQ(node->inputs().size(), 3);
        xla::XlaOp xla_output = BuildThreshold(
            node, cctx.OpForInput(node, 0), cctx.OpForInput(node, 0),
            node->get<at::Scalar>(attr::threshold).value().to<float>(),
            node->get<at::Scalar>(attr::value).value().to<float>(), b);
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::threshold_backward: {
        XLA_CHECK_EQ(node->inputs().size(), 3);
        xla::XlaOp xla_output = BuildThreshold(
            node, cctx.OpForInput(node, 1), cctx.OpForInput(node, 0),
            node->get<at::Scalar>(attr::threshold).value().to<float>(), 0, b);
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::log_softmax: {
        XLA_CHECK_EQ(node->inputs().size(), size_t(2));
        xla::XlaOp xla_output = BuildLogSoftmax(node, cctx.OpForInput(node, 0));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::_log_softmax_backward_data: {
        XLA_CHECK_EQ(node->inputs().size(), 4);
        xla::XlaOp xla_output = BuildLogSoftmaxGrad(
            node, cctx.OpForInput(node, 0), cctx.OpForInput(node, 1));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::reshape:
      case aten::view: {
        XLA_CHECK_EQ(node->inputs().size(), 2);
        xla::XlaOp xla_output = BuildView(node, cctx.OpForInput(node, 0));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::expand: {
        XLA_CHECK_GE(node->inputs().size(), 1);
        xla::XlaOp xla_output = BuildExpand(node, cctx.OpForInput(node, 0));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::stack: {
        XLA_CHECK_EQ(node->inputs().size(), 2);
        xla::XlaOp xla_output = BuildStack(
            node,
            [&cctx](const Value* node) -> xla::XlaOp {
              return cctx.GetOpForValue(node);
            },
            b);
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::cat: {
        XLA_CHECK_EQ(node->inputs().size(), 2);
        xla::XlaOp xla_output = BuildCat(
            node,
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
        XLA_CHECK_EQ(node->inputs().size(), 8);
        const auto outputs =
            BuildBatchNorm(node, cctx.OpForInput(node, 0),
                           cctx.OpForInput(node, 1), cctx.OpForInput(node, 2));
        const auto node_outputs = node->outputs();
        cctx.AddValueOp(node_outputs[0], outputs.output);
        if (node->kind() == aten::batch_norm) {
          XLA_CHECK_EQ(node->outputs().size(), 1);
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
        XLA_CHECK_EQ(node->inputs().size(), 10);
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
        XLA_CHECK_GE(node->inputs().size(), 1);
        xla::XlaOp xla_output = BuildSum(node, cctx.OpForInput(node, 0));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::nll_loss: {
        XLA_CHECK_EQ(node->inputs().size(), 5);
        xla::XlaOp xla_output = BuildNllLoss(node, cctx.OpForInput(node, 0),
                                             cctx.OpForInput(node, 1));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::nll_loss_backward: {
        XLA_CHECK_EQ(node->inputs().size(), 7);
        xla::XlaOp xla_output = BuildNllLossBackward(
            node, cctx.OpForInput(node, 1), cctx.OpForInput(node, 2));
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      case aten::size: {
        XLA_CHECK_EQ(node->inputs().size(), 1);
        std::vector<xla::int64> size_op_result;
        xla::XlaOp xla_output =
            BuildSize(node, cctx.OpForInput(node, 0), &size_op_result);
        cctx.AddSizeOpResult(node->output(0), size_op_result);
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
      case prim::SumToSize: {
        XLA_CHECK_EQ(node->inputs().size(), 2);
        xla::XlaOp xla_output = BuildSumToSize(node, cctx.OpForInput(node, 0),
                                               cctx.GetSizeOpValues());
        cctx.AddNodeOp(node, xla_output);
        break;
      }
      default:
        XLA_ERROR() << "Unsupported operator: " << node->kind().toQualString();
    }
  }
  const auto return_node = graph_->return_node();
  const auto node_inputs = return_node->inputs();
  // TODO: tighten the id check for returned tuples.
  if (return_node->kind() != prim::Return || node_inputs.empty()) {
    XLA_ERROR() << "Unexpected end of graph";
  }
  std::vector<xla::XlaOp> returned_tuple;
  XlaComputationInOut::SizeOpValues ret_size_op_values;
  for (size_t return_input_idx = 0; return_input_idx < node_inputs.size();
       ++return_input_idx) {
    const auto return_input = node_inputs[return_input_idx];
    const auto size_op_value_maybe =
        cctx.GetSizeOpValueForId(return_input->unique());
    // Add evaluated aten::size values to the return tuple.
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

}  // namespace jit
}  // namespace torch
