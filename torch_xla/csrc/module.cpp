#include "module.h"
#include "helpers.h"

#include <set>
#include "c10/util/Exception.h"
#include "cross_replica_reduces.h"
#include "passes/eval_static_size.h"
#include "passes/remove_unused_forward_outputs.h"
#include "passes/replace_untraced_operators.h"
#include "passes/threshold_backward_peephole.h"
#include "torch/csrc/jit/passes/canonicalize_ops.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/specialize_undef.h"

namespace torch {
namespace jit {
namespace {

void GatherParameters(std::vector<at::Tensor*>* values,
                      std::vector<bool>* requires_grad,
                      const script::Module& m) {
  for (auto& param : m.get_parameters()) {
    values->push_back(param->slot());
    requires_grad->push_back(!param->is_buffer);
  }
  for (const auto& sub : m.get_modules()) {
    GatherParameters(values, requires_grad, *sub->module);
  }
}

XlaModule::TensorBatchVector DecomposeComputationResult(
    const std::vector<std::shared_ptr<xla::ComputationClient::Data>>& results,
    const xla::Shape& result_shape, uint64_t module_id) {
  std::vector<xla::Shape> shapes = GetComponentShapes(result_shape);
  XlaModule::TensorBatchVector batch_tensors;
  if (shapes.size() > 1) {
    auto result_components = XlaGetClient()->DeconstructTuple(results);
    for (auto& replica_result_components : result_components) {
      XlaModule::TensorBatchVector::value_type replica_tensors;
      for (auto& replica_data : replica_result_components) {
        replica_tensors.push_back(std::make_shared<XLATensor>(
            std::move(replica_data), module_id, /*requires_grad=*/false));
      }
      batch_tensors.push_back(std::move(replica_tensors));
    }
  } else {
    for (auto& replica_data : results) {
      XlaModule::TensorBatchVector::value_type replica_tensors;
      replica_tensors.push_back(std::make_shared<XLATensor>(
          replica_data, module_id, /*requires_grad=*/false));
      batch_tensors.push_back(std::move(replica_tensors));
    }
  }
  return batch_tensors;
}

}  // namespace

std::atomic<uint64_t> XlaModule::s_module_id_(1);
constexpr uint64_t XlaModule::kInvalidModuleId;

XlaModule::XlaModule(const std::shared_ptr<script::Module> module,
                     bool use_full_conv_precision, bool differentiate)
    : use_full_conv_precision_(use_full_conv_precision),
      enable_trace_fusion_(differentiate),
      differentiate_(differentiate),
      module_id_(s_module_id_++),
      script_module_(module) {}

void XlaModule::Initialize(const TensorBatchVector& inputs) {
  if (script_module_ == nullptr) {
    return;
  }

  // Get forward graph.
  const auto forward = script_module_->find_method("forward");
  JIT_ASSERT(forward);
  std::shared_ptr<Graph> forward_graph = forward->graph()->copy();
  // Run forward passes.
  CanonicalizeOps(forward_graph);
  EvalStaticSize(forward_graph);
  ConstantPropagation(forward_graph);
  ReplaceUntracedOperators(forward_graph);
  EliminateDeadCode(forward_graph);

  // Convert model parameters to vector of XLATensors.
  std::vector<at::Tensor*> params_buffers_regather;
  std::vector<bool> param_requires_grad;
  GatherParameters(&params_buffers_regather, &param_requires_grad,
                   *script_module_);
  // The loop below is going to send individual parameters to the different
  // cores. We might need to do something smarter here.
  auto devices = CommonDevicesForReplicas(inputs);
  for (const auto& device : devices) {
    TensorBatchVector::value_type replica_params;
    TensorBatchVector::value_type optimizable_replica_params;
    for (size_t j = 0; j < params_buffers_regather.size(); ++j) {
      replica_params.push_back(std::make_shared<XLATensor>(
          autograd::as_variable_ref(*params_buffers_regather[j]), device));
      if (param_requires_grad[j]) {
        optimizable_replica_params.push_back(replica_params.back());
      }
    }
    all_params_.push_back(std::move(replica_params));
    optimizable_params_.push_back(std::move(optimizable_replica_params));
  }
  if (!differentiate_) {
    f_ = forward_graph;
    f_real_outputs_ = f_->outputs().size();
    return;
  }
  // Collect the requires-gradient property making sure all the replica inputs
  // agree on it.
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& replica_inputs = inputs[i];
    if (i == 0) {
      for (const auto& p : replica_inputs) {
        inputs_require_grad_.push_back(p->RequiresGrad());
      }
    } else {
      for (size_t j = 0; j < replica_inputs.size(); ++j) {
        CHECK(inputs_require_grad_[j] == replica_inputs[j]->RequiresGrad())
            << "Input " << j << " of replica " << i
            << " does not match the requires-grad property";
      }
    }
  }
  inputs_require_grad_.insert(inputs_require_grad_.end(),
                              param_requires_grad.begin(),
                              param_requires_grad.end());

  // Automatically differentiate the forward graph to get the backward graph.
  // Since differentiation is mutating the graph, do it on a copy.
  auto forward_graph_copy = forward_graph->copy();
  Gradient gradient = differentiate(forward_graph_copy);

  // Run the forward passes.
  CanonicalizeOps(gradient.f);
  EvalStaticSize(gradient.f);
  ConstantPropagation(gradient.f);
  ReplaceUntracedOperators(gradient.f);
  EliminateDeadCode(gradient.f);
  // Run the backward passes.
  specializeUndef(*(gradient.df.get()));
  EvalStaticSize(gradient.df);
  ConstantPropagation(gradient.df);
  ThresholdBackwardPeephole(gradient.df);
  EliminateDeadCode(gradient.df);
  // Run pass on forward and backward graphs that drops outputs that XLA doesn't
  // need.
  RemoveUnusedForwardOutputs(gradient);

  // Record the number of outputs for the forward computation and the captured
  // input and output indices to be used by the backward computation.
  f_real_outputs_ = gradient.f_real_outputs;
  df_input_captured_inputs_ = gradient.df_input_captured_inputs;
  df_input_captured_outputs_ = gradient.df_input_captured_outputs;

  // Take ownership of the forward and differentiated graphs and release the
  // reference to the script module to mark initialization as done.
  f_ = gradient.f;
  df_ = gradient.df;
  // Mark the module as initialized.
  script_module_ = nullptr;
}

void XlaModule::CheckInitialized() const {
  // script_module_ is null after initialization.
  if (script_module_ != nullptr) {
    AT_ERROR("Module not initialized; did forward method run?");
  }
}

XlaModule::TensorBatchVector XlaModule::forward(
    const TensorBatchVector& inputs) {
  Initialize(inputs);
  if (enable_trace_fusion_) {
    const auto return_node = df_->return_node();
    const auto node_inputs = return_node->inputs();
    if (!node_inputs.empty()) {
      return RunFusedTrain(inputs);
    }
  }
  return RunUnfusedForward(inputs);
}

void XlaModule::backward(const TensorBatchVector& grad_outputs) {
  JIT_ASSERTM(differentiate_,
              "Calling backward() on a module with differentiate not set");
  CheckInitialized();
  // Tensors could have pending in-place operations, apply them first to reset
  // their parent module and thus invalidate the gradients we set aside from the
  // fused computation.
  FlushTensorsOperations({&grad_outputs, &optimizable_params_});

  // If we're in trace fusion mode, we start with the assumption that the input
  // gradients are still valid and invalidate it if we don't receive the output
  // from the forward trace to compute the gradient on. If not, we have no
  // gradients by definition, since only forward pass has executed.
  bool input_gradients_valid = enable_trace_fusion_;
  for (size_t i = 0; forward_computation_ && i < grad_outputs.size(); ++i) {
    for (const auto& grad_output : grad_outputs[i]) {
      if (grad_output->ForwardModuleId() != module_id_ &&
          enable_trace_fusion_) {
        // This is not a direct output of the forward pass. Redo the forward
        // computation to capture the intermediate outputs correctly and set
        // enable_trace_fusion_ to false to avoid doing fusion for the next
        // training batches.
        forward_computation_ = at::nullopt;
        RunUnfusedForward(inputs_);
        input_gradients_valid = false;
        enable_trace_fusion_ = false;
        break;
      }
    }
  }
  if (input_gradients_valid) {
    // We already have the gradients from the fused computation, just set the
    // gradients for input and parameters.
    for (size_t i = 0; i < inputs_.size(); ++i) {
      auto& replica_inputs = inputs_[i];
      auto& replica_grad_inputs = grad_inputs_[i];
      auto& replica_optimizable_params = optimizable_params_[i];
      JIT_ASSERT(inputs_require_grad_.size() >=
                 replica_inputs.size() + replica_optimizable_params.size());
      size_t grad_index = 0;
      for (size_t j = 0; j < replica_inputs.size(); j++) {
        if (inputs_require_grad_[j]) {
          replica_inputs[j]->setGrad(replica_grad_inputs[grad_index]);
          ++grad_index;
        }
      }
      for (size_t j = 0; j < replica_optimizable_params.size(); j++) {
        replica_optimizable_params[j]->setGrad(replica_grad_inputs[grad_index]);
        ++grad_index;
      }
    }
    return;
  }
  // NOTE: The order of the input parameters passed to the BuildComputation()
  // call to build the backward computation is critical, as they have to match
  // the sequence of the graph->inputs() vector. Before the gradients passed in
  // from the backward() call, then then zeroed virtual inputs, and then the
  // captured inputs/outputs.
  TensorBatchVector raw_grad_outputs;
  std::vector<bool> zero_input;
  for (size_t i = 0; i < grad_outputs.size(); ++i) {
    TensorBatchVector::value_type replica_raw_grad_outputs;
    for (auto p : grad_outputs[i]) {
      replica_raw_grad_outputs.push_back(p);
      if (i == 0) {
        zero_input.push_back(false);
      }
    }
    for (auto p : captured_outputs_[i]) {
      // TODO(asuhan): Remove the all zero grad outputs from the forward trace
      // output.
      replica_raw_grad_outputs.push_back(p);
      if (i == 0) {
        zero_input.push_back(true);
      }
    }
    for (auto p : captured_inputs_outputs_[i]) {
      replica_raw_grad_outputs.push_back(p);
      if (i == 0) {
        zero_input.push_back(false);
      }
    }
    raw_grad_outputs.push_back(std::move(replica_raw_grad_outputs));
  }
  // If backward graph is not compiled, compile it.
  if (!backward_computation_) {
    // The shape for all the replicas are the same, so use replica[0] for
    // building the shapes vector for the BuildComputation() call.
    const auto& replica_raw_grad_outputs = raw_grad_outputs.front();
    std::vector<XlaTranslator::ParameterShape> backward_shapes;
    for (size_t j = 0; j < replica_raw_grad_outputs.size(); ++j) {
      backward_shapes.push_back(XlaTranslator::ParameterShape(
          replica_raw_grad_outputs[j]->shape(), zero_input[j]));
    }

    XlaTranslator xla_bwd_impl(df_, GetPrecisionConfig());
    backward_computation_ = xla_bwd_impl.BuildComputation(
        backward_shapes, GetBackwardBuildOptions(0, inputs_.size()));
  }
  // Collect the computation client data vector.
  DataBatchVector raw_grad_outputs_data =
      GetDataBatchVector(raw_grad_outputs, &zero_input);
  auto devices = CommonDevicesForReplicas(grad_outputs);
  const auto program_shape =
      backward_computation_->GetProgramShape().ValueOrDie();
  const auto result_shape = program_shape.result();
  auto result_shape_with_layout =
      MakeShapeWithDeviceLayout(result_shape, devices.front().hw_type);

  TensorBatchVector grad_inputs =
      Execute(*backward_computation_, raw_grad_outputs_data, result_shape,
              &result_shape_with_layout, kInvalidModuleId);

  for (size_t i = 0; i < inputs_.size(); ++i) {
    auto& replica_grad_inputs = grad_inputs[i];
    auto& replica_inputs = inputs_[i];
    auto& replica_optimizable_params = optimizable_params_[i];
    JIT_ASSERT((replica_inputs.size() + replica_optimizable_params.size()) ==
               replica_grad_inputs.size());
    // Set .grad attributes of the input and parameter tensors.
    for (size_t j = 0; j < replica_inputs.size(); j++) {
      replica_inputs[j]->setGrad(replica_grad_inputs[j]);
    }
    for (size_t j = 0; j < replica_optimizable_params.size(); j++) {
      auto t = replica_grad_inputs[j + replica_inputs.size()];
      replica_optimizable_params[j]->setGrad(t);
    }
  }
  // Release handles to saved / captured inputs and outputs.
  inputs_.clear();
  captured_outputs_.clear();
  captured_inputs_outputs_.clear();
}

XlaModule::TensorBatchVector XlaModule::RunFusedTrain(
    const TensorBatchVector& inputs) {
  Initialize(inputs);
  TensorBatchVector inputs_params_buffers = PrepareForwardInput(inputs);
  if (!forward_computation_) {
    // Shapes are going to be the same for all replicas, so use the ones of the
    // first replica here.
    std::vector<XlaTranslator::ParameterShape> forward_shapes;
    for (auto p : inputs_params_buffers.front()) {
      forward_shapes.push_back(
          XlaTranslator::ParameterShape(p->shape(), /*zero_input=*/false));
    }
    BuildFusedTrainComputation(forward_shapes);
  }
  DataBatchVector inputs_params_buffers_data =
      GetDataBatchVector(inputs_params_buffers, /*zero_input=*/nullptr);
  const auto program_shape =
      forward_computation_->GetProgramShape().ValueOrDie();
  const auto result_shape = program_shape.result();
  // The result is always a tuple of outputs and gradients.
  CHECK(xla::ShapeUtil::IsTuple(result_shape))
      << xla::ShapeUtil::HumanString(result_shape);
  const auto device = XLATensor::CommonDeviceForTensors(inputs.front());
  auto result_shape_with_layout =
      MakeShapeWithDeviceLayout(result_shape, device.hw_type);

  TensorBatchVector result_components =
      Execute(*forward_computation_, inputs_params_buffers_data, result_shape,
              &result_shape_with_layout, module_id_);

  // First f_real_outputs_ are the forward outputs returned to user code.
  CHECK_LE(f_real_outputs_, result_components.front().size());
  grad_inputs_.clear();
  TensorBatchVector forward_result;
  for (auto& replica_result_components : result_components) {
    TensorBatchVector::value_type replica_forward_result;
    TensorBatchVector::value_type replica_grad_inputs;
    for (size_t j = 0; j < f_real_outputs_; ++j) {
      replica_forward_result.push_back(replica_result_components[j]);
    }
    for (size_t j = f_real_outputs_; j < replica_result_components.size();
         ++j) {
      replica_grad_inputs.push_back(replica_result_components[j]);
    }
    forward_result.push_back(std::move(replica_forward_result));
    grad_inputs_.push_back(std::move(replica_grad_inputs));
  }
  return forward_result;
}

const XlaModule::TensorBatchVector& XlaModule::parameters() {
  CheckInitialized();
  return optimizable_params_;
}

const XlaModule::TensorBatchVector& XlaModule::parameters_buffers() {
  CheckInitialized();
  return all_params_;
}

xla::PrecisionConfig::Precision XlaModule::GetPrecisionConfig() const {
  return use_full_conv_precision_ ? xla::PrecisionConfig::HIGHEST
                                  : xla::PrecisionConfig::DEFAULT;
}

void XlaModule::BuildFusedTrainComputation(
    const std::vector<XlaTranslator::ParameterShape>& forward_shapes) {
  XlaTranslator xla_fwd_impl(f_, GetPrecisionConfig());
  xla::XlaBuilder b("XlaFusedComputation");
  // Build the forward pass program without compiling it, the backward pass
  // needs to be called before finalizing it.
  auto computation_in_outs =
      xla_fwd_impl.BuildComputationProgram(forward_shapes, &b);
  // Take the XLA outputs from the forward pass and set them for the backward
  // call in the same order the standalone, unfused version takes its arguments.
  CHECK(!computation_in_outs.outputs.empty());
  std::vector<xla::XlaOp> grad_outputs;
  for (size_t i = 0; i < f_real_outputs_; i++) {
    grad_outputs.push_back(computation_in_outs.outputs[i]);
  }
  std::vector<xla::XlaOp> captured_outputs;
  for (size_t i = f_real_outputs_; i < computation_in_outs.outputs.size();
       i++) {
    captured_outputs.push_back(computation_in_outs.outputs[i]);
  }
  std::vector<xla::XlaOp> captured_inputs_outputs;
  for (auto i : df_input_captured_inputs_) {
    captured_inputs_outputs.push_back(computation_in_outs.inputs[i]);
  }
  for (auto i : df_input_captured_outputs_) {
    captured_inputs_outputs.push_back(computation_in_outs.outputs[i]);
  }
  // NOTE: The order of the input parameters passed to the BuildComputation()
  // call to build the backward computation is critical, as they have to match
  // the sequence of the graph->inputs() vector. Before the gradients returned
  // by the forward pass, then then zeroed virtual inputs, and then the captured
  // inputs/outputs.
  std::vector<XlaTranslator::ParameterShape> backward_shapes;
  std::vector<xla::XlaOp> backward_operands;
  for (auto p : grad_outputs) {
    backward_shapes.push_back(XlaTranslator::ParameterShape(
        XlaHelpers::ShapeOfXlaOp(p), /*zero_input=*/false));
    backward_operands.push_back(p);
  }
  for (auto p : captured_outputs) {
    backward_shapes.push_back(XlaTranslator::ParameterShape(
        XlaHelpers::ShapeOfXlaOp(p), /*zero_input=*/true));
  }
  for (auto p : captured_inputs_outputs) {
    backward_shapes.push_back(XlaTranslator::ParameterShape(
        XlaHelpers::ShapeOfXlaOp(p), /*zero_input=*/false));
    backward_operands.push_back(p);
  }
  // The arguments are set up correctly, call into the backward computation.
  XlaTranslator xla_bwd_impl(df_, GetPrecisionConfig());
  auto backward_computation = xla_bwd_impl.BuildComputation(
      backward_shapes,
      GetBackwardBuildOptions(f_real_outputs_, inputs_.size()));
  xla::Call(&b, backward_computation, backward_operands);
  forward_computation_ = b.Build().ValueOrDie();
}

XlaModule::TensorBatchVector XlaModule::RunUnfusedForward(
    const TensorBatchVector& inputs) {
  TensorBatchVector inputs_params_buffers = PrepareForwardInput(inputs);

  // Lazy-convert forward graph to XlaComputation.
  if (!forward_computation_) {
    // Shapes are going to be the same for all replicas, so use the ones of the
    // first replica here.
    std::vector<XlaTranslator::ParameterShape> forward_shapes;
    for (auto p : inputs_params_buffers.front()) {
      forward_shapes.push_back(
          XlaTranslator::ParameterShape(p->shape(), /*zero_input=*/false));
    }

    XlaTranslator xla_fwd_impl(f_, GetPrecisionConfig());
    forward_computation_ = xla_fwd_impl.BuildComputation(forward_shapes);
  }
  DataBatchVector inputs_params_buffers_data =
      GetDataBatchVector(inputs_params_buffers, /*zero_input=*/nullptr);
  const auto program_shape =
      forward_computation_->GetProgramShape().ValueOrDie();
  const auto result_shape = program_shape.result();
  const auto device = XLATensor::CommonDeviceForTensors(inputs.front());
  auto result_shape_with_layout =
      MakeShapeWithDeviceLayout(result_shape, device.hw_type);

  TensorBatchVector raw_outputs =
      Execute(*forward_computation_, inputs_params_buffers_data, result_shape,
              &result_shape_with_layout, kInvalidModuleId);

  TensorBatchVector outputs;
  for (size_t i = 0; i < raw_outputs.size(); ++i) {
    auto& replica_raw_outputs = raw_outputs[i];
    TensorBatchVector::value_type replica_outputs;
    for (size_t j = 0; j < f_real_outputs_; j++) {
      replica_outputs.push_back(replica_raw_outputs[j]);
    }
    outputs.push_back(std::move(replica_outputs));

    TensorBatchVector::value_type replica_captured_outputs;
    for (size_t j = f_real_outputs_; j < replica_raw_outputs.size(); j++) {
      replica_captured_outputs.push_back(replica_raw_outputs[j]);
    }
    captured_outputs_.push_back(std::move(replica_captured_outputs));

    auto& replica_inputs_params_buffers = inputs_params_buffers[i];
    TensorBatchVector::value_type replica_captured_inputs_outputs;
    for (auto j : df_input_captured_inputs_) {
      replica_captured_inputs_outputs.push_back(
          replica_inputs_params_buffers[j]);
    }
    for (auto j : df_input_captured_outputs_) {
      replica_captured_inputs_outputs.push_back(replica_raw_outputs[j]);
    }
    captured_inputs_outputs_.push_back(
        std::move(replica_captured_inputs_outputs));
  }
  return outputs;
}

XlaModule::TensorBatchVector XlaModule::PrepareForwardInput(
    const TensorBatchVector& inputs) {
  FlushTensorsOperations({&inputs, &optimizable_params_});
  // Clear the previous forward's captured vectors.
  // This is needed in case backward is not yet run, but two forward calls were
  // made.
  captured_outputs_.clear();
  captured_inputs_outputs_.clear();
  // Needed so that in backward, we can set .grad attributes correctly.
  inputs_ = inputs;

  TensorBatchVector inputs_params_buffers;
  CHECK_EQ(inputs_.size(), all_params_.size());
  for (size_t i = 0; i < inputs_.size(); ++i) {
    TensorBatchVector::value_type replica_inputs_params_buffers;
    for (auto p : inputs_[i]) {
      replica_inputs_params_buffers.push_back(p);
    }
    for (auto p : all_params_[i]) {
      replica_inputs_params_buffers.push_back(p);
    }
    inputs_params_buffers.push_back(std::move(replica_inputs_params_buffers));
  }
  return inputs_params_buffers;
}

XlaModule::TensorBatchVector XlaModule::Execute(
    const xla::XlaComputation& computation, const DataBatchVector& inputs,
    const xla::Shape& result_shape, const xla::Shape* output_shape,
    uint64_t module_id) {
  auto client = XlaGetClient();
  std::vector<std::shared_ptr<xla::ComputationClient::Data>> exec_results;
  if (inputs.size() == 1) {
    exec_results.push_back(
        client->ExecuteComputation(computation, inputs.front(), output_shape));
  } else {
    exec_results = client->ExecuteReplicated(computation, inputs, output_shape);
  }
  return DecomposeComputationResult(std::move(exec_results), result_shape,
                                    module_id);
}

XlaTranslator::BuildOptions XlaModule::GetBackwardBuildOptions(
    size_t param_to_return_count, size_t num_replicas) {
  XlaTranslator::BuildOptions options;
  options.param_to_return_count = param_to_return_count;
  if (num_replicas > 1) {
    options.output_transform = [this, num_replicas](const xla::XlaOp& op,
                                                    size_t) {
      return BuildCrossReplicaSum(op, num_replicas);
    };
  }
  return options;
}

void XlaModule::FlushTensorsOperations(
    std::initializer_list<const TensorBatchVector*> batch_tensors) {
  std::vector<std::shared_ptr<XLATensor>> tensors;
  for (auto batch_tensor : batch_tensors) {
    for (const auto& replica_tensors : *batch_tensor) {
      tensors.insert(tensors.end(), replica_tensors.begin(),
                     replica_tensors.end());
    }
  }
  XLATensor::ApplyPendingGraph(tensors);
}

XlaModule::DataBatchVector XlaModule::GetDataBatchVector(
    const TensorBatchVector& inputs, const std::vector<bool>* zero_input) {
  DataBatchVector inputs_data;
  for (auto& replica_inputs : inputs) {
    DataBatchVector::value_type replica_inputs_data;
    for (size_t j = 0; j < replica_inputs.size(); ++j) {
      if (zero_input == nullptr || !zero_input->at(j)) {
        replica_inputs_data.push_back(replica_inputs[j]->GetXlaData().get());
      }
    }
    inputs_data.push_back(std::move(replica_inputs_data));
  }
  return inputs_data;
}

std::vector<XLATensor::Device> XlaModule::CommonDevicesForReplicas(
    const TensorBatchVector& inputs) {
  std::vector<XLATensor::Device> devices;
  std::set<XLATensor::Device> unique_devices;
  for (auto& replica_inputs : inputs) {
    devices.push_back(XLATensor::CommonDeviceForTensors(replica_inputs));
    CHECK(unique_devices.insert(devices.back()).second)
        << "Duplicated device in different replicas: "
        << devices.back().ToString();
  }
  return devices;
}

}  // namespace jit
}  // namespace torch
