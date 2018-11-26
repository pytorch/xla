#include "module.h"
#include "helpers.h"

#include <algorithm>
#include <set>
#include "c10/util/Exception.h"
#include "cross_replica_reduces.h"
#include "passes/eval_static_size.h"
#include "passes/insert_explicit_expand.h"
#include "passes/remove_unused_forward_outputs.h"
#include "passes/replace_untraced_operators.h"
#include "passes/set_mat_mul_output_shape.h"
#include "passes/threshold_backward_peephole.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "torch/csrc/jit/passes/canonicalize_ops.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/lower_tuples.h"
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
    const xla::Shape& result_shape) {
  std::vector<xla::Shape> shapes = GetComponentShapes(result_shape);
  XlaModule::TensorBatchVector batch_tensors;
  if (shapes.size() > 1) {
    auto result_components = XlaGetClient()->DeconstructTuple(results);
    for (auto& replica_result_components : result_components) {
      XlaModule::TensorBatchVector::value_type replica_tensors;
      for (auto& replica_data : replica_result_components) {
        replica_tensors.push_back(XLATensor::Create(std::move(replica_data),
                                                    /*requires_grad=*/false));
      }
      batch_tensors.push_back(std::move(replica_tensors));
    }
  } else {
    for (auto& replica_data : results) {
      XlaModule::TensorBatchVector::value_type replica_tensors;
      replica_tensors.push_back(
          XLATensor::Create(replica_data, /*requires_grad=*/false));
      batch_tensors.push_back(std::move(replica_tensors));
    }
  }
  return batch_tensors;
}

}  // namespace

XlaModule::XlaModule(const std::shared_ptr<script::Module> module,
                     bool use_full_conv_precision, bool differentiate)
    : use_full_conv_precision_(use_full_conv_precision),
      differentiate_(differentiate),
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
  SetMatMulOutputShape(forward_graph);
  InsertExplicitExpand(forward_graph);
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
  devices_ = CommonDevicesForReplicas(inputs);
  for (const auto& device : devices_) {
    TensorBatchVector::value_type replica_params;
    TensorBatchVector::value_type optimizable_replica_params;
    for (size_t j = 0; j < params_buffers_regather.size(); ++j) {
      replica_params.push_back(XLATensor::Create(
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
        XLA_CHECK(inputs_require_grad_[j] == replica_inputs[j]->RequiresGrad())
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
  InsertExplicitExpand(gradient.f);
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
  LowerAllTuples(gradient.df);
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
  VLOG(4) << "Gradient F:\n" << f_->toString();
  VLOG(4) << "Gradient DF:\n" << df_->toString();
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
  if (!backward_input_gradients_.empty()) {
    const auto return_node = df_->return_node();
    const auto node_inputs = return_node->inputs();
    if (!node_inputs.empty()) {
      return RunFusedTrain(inputs);
    }
  }
  return RunUnfusedForward(inputs);
}

void XlaModule::SetInputGradientsForFusion(std::vector<at::Tensor> gradients) {
  backward_input_gradients_ = std::move(gradients);
}

void XlaModule::backward(const TensorBatchVector& grad_outputs) {
  JIT_ASSERTM(differentiate_,
              "Calling backward() on a module with differentiate not set");
  CheckInitialized();
  // Tensors could have pending in-place operations, apply them first to reset
  // their parent module and thus invalidate the gradients we set aside from the
  // fused computation.
  FlushTensorsOperations();

  if (!backward_input_gradients_.empty()) {
    // We already have the gradients from the fused computation, just set the
    // gradients for input and parameters.
    ApplyGradients(grad_inputs_, inputs_, optimizable_params_,
                   inputs_require_grad_, *df_);
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
      XlaTranslator::ParameterKind kind =
          zero_input[j] ? XlaTranslator::ParameterKind::kZeroInput
                        : XlaTranslator::ParameterKind::kGraphInput;
      backward_shapes.push_back(XlaTranslator::ParameterShape(
          replica_raw_grad_outputs[j]->shape(), kind));
    }

    XlaTranslator xla_bwd_impl(df_, GetPrecisionConfig());
    backward_computation_ = xla_bwd_impl.BuildComputation(
        "XlaBackward", backward_shapes,
        GetBackwardBuildOptions(0, inputs_.size()));
    backward_shape_.reset();
  }
  // Collect the computation client data vector.
  DataBatchVector raw_grad_outputs_data =
      GetDataBatchVector(raw_grad_outputs, &zero_input);
  if (!backward_shape_) {
    backward_shape_ = GetResultShape(*backward_computation_, grad_outputs);
  }

  TensorBatchVector grad_inputs =
      Execute(*backward_computation_, raw_grad_outputs_data, devices_,
              *backward_shape_);

  ApplyGradients(grad_inputs, inputs_, optimizable_params_,
                 inputs_require_grad_, *df_);
  // Release handles to saved / captured inputs and outputs.
  inputs_.clear();
  captured_outputs_.clear();
  captured_inputs_outputs_.clear();
}

void XlaModule::ApplyGradients(const TensorBatchVector& grad_inputs,
                               const TensorBatchVector& inputs,
                               const TensorBatchVector& optimizable_params,
                               const std::vector<bool>& inputs_require_grad,
                               const Graph& df) {
  size_t inputs_require_grad_count =
      std::count(inputs_require_grad.begin(), inputs_require_grad.end(), true);
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto& replica_grad_inputs = grad_inputs[i];
    auto& replica_inputs = inputs[i];
    auto& replica_optimizable_params = optimizable_params[i];
    XLA_CHECK_EQ(replica_grad_inputs.size(), inputs_require_grad_count)
        << "Graph:\n"
        << df.toString();
    size_t grad_index = 0;
    for (size_t j = 0; j < replica_inputs.size(); j++) {
      if (inputs_require_grad[j]) {
        replica_inputs[j]->setGrad(replica_grad_inputs[grad_index]);
        ++grad_index;
      }
    }
    for (size_t j = 0; j < replica_optimizable_params.size(); j++) {
      replica_optimizable_params[j]->setGrad(replica_grad_inputs[grad_index]);
      ++grad_index;
    }
  }
}

XlaModule::TensorBatchVector XlaModule::RunFusedTrain(
    const TensorBatchVector& inputs) {
  Initialize(inputs);

  TensorBatchVector inputs_params_buffers = PrepareForwardInput(inputs);
  if (!forward_computation_) {
    // Shapes are going to be the same for all replicas, so use the ones of the
    // first replica here.
    const TensorBatchVector::value_type& replica_inputs =
        inputs_params_buffers.front();
    std::vector<XlaTranslator::ParameterShape> forward_shapes;
    for (size_t i = 0; i < replica_inputs.size(); ++i) {
      forward_shapes.push_back(XlaTranslator::ParameterShape(
          replica_inputs[i]->shape(),
          XlaTranslator::ParameterKind::kGraphInput));
    }
    BuildFusedTrainComputation(forward_shapes);
  }
  DataBatchVector inputs_params_buffers_data =
      GetDataBatchVector(inputs_params_buffers, /*zero_input=*/nullptr);
  if (!forward_shape_) {
    forward_shape_ = GetResultShape(*forward_computation_, inputs);
  }

  TensorBatchVector result_components =
      Execute(*forward_computation_, inputs_params_buffers_data, devices_,
              *forward_shape_);

  // First f_real_outputs_ are the forward outputs returned to user code.
  XLA_CHECK_LE(f_real_outputs_, result_components.front().size());
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
  XLA_CHECK(!computation_in_outs.outputs.empty());
  XLA_CHECK_EQ(f_real_outputs_, backward_input_gradients_.size());
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
  // the sequence of the graph->inputs() vector. Before the gradients passed in
  // by the user, then then zeroed virtual inputs, and then the captured
  // inputs/outputs.
  std::vector<XlaTranslator::ParameterShape> backward_shapes;
  std::vector<xla::XlaOp> backward_operands;
  for (size_t i = 0; i < backward_input_gradients_.size(); ++i) {
    xla::Literal literal =
        GetTensorLiteral(backward_input_gradients_[i], /*shape=*/nullptr);
    xla::XlaOp gradient_op = xla::ConstantLiteral(&b, literal);
    backward_shapes.push_back(XlaTranslator::ParameterShape(
        XlaHelpers::ShapeOfXlaOp(gradient_op),
        XlaTranslator::ParameterKind::kGraphInput));
    backward_operands.push_back(gradient_op);
  }
  for (auto p : captured_outputs) {
    backward_shapes.push_back(XlaTranslator::ParameterShape(
        XlaHelpers::ShapeOfXlaOp(p), XlaTranslator::ParameterKind::kZeroInput));
  }
  for (auto p : captured_inputs_outputs) {
    backward_shapes.push_back(XlaTranslator::ParameterShape(
        XlaHelpers::ShapeOfXlaOp(p),
        XlaTranslator::ParameterKind::kGraphInput));
    backward_operands.push_back(p);
  }
  // The arguments are set up correctly, call into the backward computation.
  XlaTranslator xla_bwd_impl(df_, GetPrecisionConfig());
  auto backward_computation =
      xla_bwd_impl.BuildComputation("XlaBackward", backward_shapes,
                                    GetBackwardBuildOptions(0, inputs_.size()));
  xla::XlaOp backward_op =
      xla::Call(&b, backward_computation, backward_operands);

  // Return the real outputs of the forward, followed by the outputs of the
  // backward.
  std::vector<xla::XlaOp> returned_outputs;
  for (size_t i = 0; i < f_real_outputs_; ++i) {
    returned_outputs.push_back(computation_in_outs.outputs[i]);
  }
  xla::Shape backward_shape = XlaHelpers::ShapeOfXlaOp(backward_op);
  if (xla::ShapeUtil::IsTuple(backward_shape)) {
    for (xla::int64 i = 0;
         i < xla::ShapeUtil::TupleElementCount(backward_shape); ++i) {
      returned_outputs.push_back(xla::GetTupleElement(backward_op, i));
    }
  } else if (!xla::ShapeUtil::IsEmptyTuple(backward_shape)) {
    returned_outputs.push_back(backward_op);
  }
  XlaHelpers::CreateReturnValue(&b, returned_outputs);

  forward_computation_ = b.Build().ValueOrDie();
  forward_shape_.reset();
  VLOG(5) << "Fused computation:\n"
          << xla::xrt_util::GetComputationHloText(*forward_computation_)
                 .ValueOrDie();
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
      forward_shapes.push_back(XlaTranslator::ParameterShape(
          p->shape(), XlaTranslator::ParameterKind::kGraphInput));
    }

    XlaTranslator xla_fwd_impl(f_, GetPrecisionConfig());
    forward_computation_ =
        xla_fwd_impl.BuildComputation("XlaForward", forward_shapes);
    forward_shape_.reset();
  }
  DataBatchVector inputs_params_buffers_data =
      GetDataBatchVector(inputs_params_buffers, /*zero_input=*/nullptr);
  if (!forward_shape_) {
    forward_shape_ = GetResultShape(*forward_computation_, inputs);
  }

  TensorBatchVector raw_outputs =
      Execute(*forward_computation_, inputs_params_buffers_data, devices_,
              *forward_shape_);

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
  FlushTensorsOperations();
  // Clear the previous forward's captured vectors.
  // This is needed in case backward is not yet run, but two forward calls were
  // made.
  captured_outputs_.clear();
  captured_inputs_outputs_.clear();
  // Needed so that in backward, we can set .grad attributes correctly.
  inputs_ = inputs;

  TensorBatchVector inputs_params_buffers;
  XLA_CHECK_EQ(inputs_.size(), all_params_.size());
  for (size_t i = 0; i < inputs_.size(); ++i) {
    TensorBatchVector::value_type replica_inputs_params_buffers;
    for (auto& p : inputs_[i]) {
      replica_inputs_params_buffers.push_back(p);
    }
    for (auto& p : all_params_[i]) {
      replica_inputs_params_buffers.push_back(p);
    }
    inputs_params_buffers.push_back(std::move(replica_inputs_params_buffers));
  }
  return inputs_params_buffers;
}

XlaModule::TensorBatchVector XlaModule::Execute(
    const xla::XlaComputation& computation, const DataBatchVector& inputs,
    const std::vector<XLATensor::Device>& devices,
    const xla::Shape& result_shape) {
  std::vector<std::string> device_strings(devices.size());
  for (size_t i = 0; i < devices.size(); ++i) {
    device_strings[i] = devices[i].ToString();
  }
  auto client = XlaGetClient();
  std::vector<std::shared_ptr<xla::ComputationClient::Data>> exec_results;
  if (inputs.size() == 1) {
    exec_results.push_back(client->ExecuteComputation(
        computation, inputs.front(), device_strings[0], &result_shape));
  } else {
    exec_results = client->ExecuteReplicated(computation, inputs,
                                             device_strings, &result_shape);
  }
  return DecomposeComputationResult(std::move(exec_results), result_shape);
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

void XlaModule::FlushTensorsOperations() {
  // We might have to do something smarter here, as we are syncing even tensors
  // which are not part of the traning loop. Nothing happens, but if we want to
  // fuse the sync operation with the forward+backward+optimizer, we need to
  // have a path leading to the same XLA computation.
  std::vector<std::shared_ptr<XLATensor>> tensors = XLATensor::GetLiveTensors();
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
    XLA_CHECK(unique_devices.insert(devices.back()).second)
        << "Duplicated device in different replicas: "
        << devices.back().ToString();
  }
  return devices;
}

xla::Shape XlaModule::GetResultShape(const xla::XlaComputation& computation,
                                     const TensorBatchVector& input_tensors) {
  auto devices = CommonDevicesForReplicas(input_tensors);
  const auto program_shape = computation.GetProgramShape().ValueOrDie();
  const auto result_shape = program_shape.result();
  return MakeShapeWithDeviceLayout(result_shape, devices.front().hw_type);
}

}  // namespace jit
}  // namespace torch
