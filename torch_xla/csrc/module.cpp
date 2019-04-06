#include "torch_xla/csrc/module.h"
#include "torch_xla/csrc/helpers.h"

#include <algorithm>
#include <set>
#include "c10/util/Exception.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "torch/csrc/jit/passes/canonicalize_ops.h"
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/lower_tuples.h"
#include "torch/csrc/jit/passes/shape_analysis.h"
#include "torch/csrc/jit/passes/specialize_autogradzero.h"
#include "torch_xla/csrc/cross_replica_reduces.h"
#include "torch_xla/csrc/passes/eval_static_size.h"
#include "torch_xla/csrc/passes/remove_in_place_out_param_ops.h"
#include "torch_xla/csrc/passes/remove_unused_forward_outputs.h"
#include "torch_xla/csrc/passes/replace_in_place_ops.h"
#include "torch_xla/csrc/passes/replace_untraced_operators.h"
#include "torch_xla/csrc/passes/threshold_backward_peephole.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace {

// Extract the slots from a named IValue dictionary.
std::unordered_set<torch::jit::script::Slot> ToSlotSet(
    const c10::ArrayRef<torch::jit::script::Slot>& slots) {
  std::unordered_set<torch::jit::script::Slot> slots_set;
  for (const auto& slot : slots) {
    slots_set.insert(slot);
  }
  XLA_CHECK_EQ(slots_set.size(), slots.size())
      << "Found duplicated values in the Slot list";
  return slots_set;
}

void GatherParameters(std::vector<at::Tensor>* values,
                      std::vector<bool>* requires_grad,
                      const torch::jit::script::Module& m,
                      const torch::jit::script::Method* forward) {
  const auto parameter_set = ToSlotSet(m.get_parameters());
  const auto attribute_set = ToSlotSet(m.get_attributes());
  for (auto& initial_ivalue : forward->initial_ivalues()) {
    if (parameter_set.find(initial_ivalue) != parameter_set.end()) {
      values->push_back(initial_ivalue.value().toTensor());
      requires_grad->push_back(true);
    } else if (attribute_set.find(initial_ivalue) != attribute_set.end() &&
               initial_ivalue.value().isTensor()) {
      values->push_back(initial_ivalue.value().toTensor());
      requires_grad->push_back(false);
    }
  }
  for (const auto& submodule : m.get_modules()) {
    GatherParameters(values, requires_grad, *submodule, forward);
  }
}

XlaModule::TensorBatchVector CreateResultBatchVector(
    std::vector<std::vector<xla::ComputationClient::DataPtr>> results) {
  XlaModule::TensorBatchVector batch_tensors;
  for (auto& replica_result_components : results) {
    XlaModule::TensorBatchVector::value_type replica_tensors;
    for (auto& replica_data : replica_result_components) {
      replica_tensors.push_back(XLATensor::Create(std::move(replica_data),
                                                  /*requires_grad=*/false));
    }
    batch_tensors.push_back(std::move(replica_tensors));
  }
  return batch_tensors;
}

// Returns the number of real outputs from the forward graph pointed to by
// df_input_vjps. df_input_vjps contains an ordered subset of the full real
// outputs set, followed by an ordered subset of the additional outputs.
size_t InputVjpsRealOutputCount(const torch::jit::Gradient& gradient) {
  size_t real_output_count = 0;
  for (; real_output_count < gradient.df_input_vjps.size();
       ++real_output_count) {
    if (gradient.df_input_vjps[real_output_count] >= gradient.f_real_outputs) {
      break;
    }
  }
  return real_output_count;
}

}  // namespace

XlaModule::XlaModule(const std::shared_ptr<torch::jit::script::Module> module,
                     bool differentiate)
    : differentiate_(differentiate), script_module_(module) {}

void XlaModule::Initialize(const TensorBatchVector& inputs) {
  if (script_module_ == nullptr) {
    return;
  }

  // Get forward graph.
  const auto forward = script_module_->find_method("forward");
  XLA_CHECK(forward != nullptr) << "Forward method not found in the module";
  std::shared_ptr<torch::jit::Graph> forward_graph = forward->graph()->copy();
  RunForwardPasses(&forward_graph);

  // Convert model parameters to vector of XLATensors.
  std::vector<at::Tensor> params_buffers_regather;
  std::vector<bool> param_requires_grad;
  GatherParameters(&params_buffers_regather, &param_requires_grad,
                   *script_module_, forward);
  // The loop below is going to send individual parameters to the different
  // cores. We might need to do something smarter here.
  devices_ = CommonDevicesForReplicas(inputs);
  for (const auto& device : devices_) {
    TensorBatchVector::value_type replica_params;
    TensorBatchVector::value_type optimizable_replica_params;
    for (size_t j = 0; j < params_buffers_regather.size(); ++j) {
      const torch::autograd::Variable& var_ref =
          torch::autograd::as_variable_ref(params_buffers_regather[j]);
      replica_params.push_back(
          XLATensor::Create(var_ref, device, var_ref.requires_grad()));
      if (param_requires_grad[j]) {
        optimizable_replica_params.push_back(replica_params.back());
      }
    }
    all_params_.push_back(std::move(replica_params));
    optimizable_params_.push_back(std::move(optimizable_replica_params));
  }
  if (!differentiate_) {
    gradient_.f = forward_graph;
    gradient_.f_real_outputs = forward_graph->outputs().size();
    return;
  }
  // Collect the requires-gradient property making sure all the replica inputs
  // agree on it.
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& replica_inputs = inputs[i];
    if (i == 0) {
      for (const auto& p : replica_inputs) {
        inputs_require_grad_.push_back(p.RequiresGrad());
      }
    } else {
      for (size_t j = 0; j < replica_inputs.size(); ++j) {
        XLA_CHECK(inputs_require_grad_[j] == replica_inputs[j].RequiresGrad())
            << "Input " << j << " of replica " << i
            << " does not match the requires-grad property";
      }
    }
  }
  inputs_require_grad_.insert(inputs_require_grad_.end(),
                              param_requires_grad.begin(),
                              param_requires_grad.end());

  gradient_ = ComputeGradient(forward_graph);

  TF_VLOG(4) << "Gradient F:\n" << gradient_.f->toString();
  TF_VLOG(4) << "Gradient DF:\n" << gradient_.df->toString();
  // Release the reference to the script module to mark initialization as done.
  script_module_ = nullptr;
}

void XlaModule::RunForwardPasses(std::shared_ptr<torch::jit::Graph>* graph) {
  // Run forward passes.
  CanonicalizeOps(*graph);
  EvalStaticSize(*graph);
  ConstantPropagation(*graph);
  ReplaceUntracedOperators(*graph);
  RemoveInPlaceOutParamOps(*graph);
  ReplaceInPlaceOps(*graph);
  EliminateDeadCode(*graph);
  LowerAllTuples(*graph);
}

torch::jit::Gradient XlaModule::ComputeGradient(
    const std::shared_ptr<torch::jit::Graph>& graph) {
  // Automatically differentiate the forward graph to get the backward graph.
  // Since differentiation is mutating the graph, do it on a copy.
  std::shared_ptr<torch::jit::Graph> graph_copy = graph->copy();
  torch::jit::Gradient gradient = differentiate(graph_copy);
  // Run the forward passes.
  CanonicalizeOps(gradient.f);
  ConstantPropagation(gradient.f);
  ReplaceUntracedOperators(gradient.f);
  EliminateDeadCode(gradient.f);
  // Run the backward passes.
  specializeAutogradZero(*(gradient.df.get()));
  ConstantPropagation(gradient.df);
  ThresholdBackwardPeephole(gradient.df);
  EliminateDeadCode(gradient.df);
  LowerAllTuples(gradient.df);
  // Run pass on forward and backward graphs that drops outputs that XLA doesn't
  // need.
  RemoveUnusedForwardOutputs(&gradient);
  return gradient;
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
    const auto return_node = gradient_.df->return_node();
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
  XLA_CHECK(differentiate_)
      << "Calling backward() on a module with differentiate not set";
  CheckInitialized();

  if (!backward_input_gradients_.empty()) {
    // We already have the gradients from the fused computation, just set the
    // gradients for input and parameters.
    ApplyGradients(grad_inputs_, &inputs_, &optimizable_params_,
                   inputs_require_grad_, *gradient_.df);
    return;
  }
  // Tensors could have pending in-place operations, apply them first to reset
  // their parent module and thus invalidate the gradients we set aside from the
  // fused computation.
  FlushTensorsOperations();

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
    const auto& replica_captured_outputs = captured_outputs_[i];
    const auto input_vjps_real_outputs = InputVjpsRealOutputCount(gradient_);
    XLA_CHECK_EQ(input_vjps_real_outputs, replica_raw_grad_outputs.size());
    for (size_t input_vjp_idx = input_vjps_real_outputs;
         input_vjp_idx < gradient_.df_input_vjps.size(); ++input_vjp_idx) {
      const auto raw_output_index = gradient_.df_input_vjps[input_vjp_idx];
      // The index in gradient_.df_input_vjps points inside all outputs list,
      // both real and captured. Skip the real output count to get the captured
      // output index.
      XLA_CHECK_GE(raw_output_index, input_vjps_real_outputs);
      XLA_CHECK_LT(raw_output_index - input_vjps_real_outputs,
                   replica_captured_outputs.size());
      XLATensor p =
          replica_captured_outputs[raw_output_index - input_vjps_real_outputs];
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
  if (backward_computation_ == nullptr) {
    // The shape for all the replicas are the same, so use replica[0] for
    // building the shapes vector for the BuildComputation() call.
    const auto& replica_raw_grad_outputs = raw_grad_outputs.front();
    std::vector<XlaTranslator::ParameterShape> backward_shapes;
    for (size_t j = 0; j < replica_raw_grad_outputs.size(); ++j) {
      XlaTranslator::ParameterKind kind =
          zero_input[j] ? XlaTranslator::ParameterKind::kZeroInput
                        : XlaTranslator::ParameterKind::kGraphInput;
      backward_shapes.push_back(XlaTranslator::ParameterShape(
          replica_raw_grad_outputs[j].shape(), kind));
    }

    XlaTranslator xla_bwd_impl(gradient_.df);
    xla::XlaComputation computation =
        xla_bwd_impl
            .BuildComputation("XlaBackward", backward_shapes,
                              backward_size_op_values_,
                              GetBackwardBuildOptions(inputs_.size()))
            .computation;
    xla::Shape result_shape = GetResultShape(computation, grad_outputs);
    backward_computation_ = xla::ComputationClient::Get()->Compile(
        std::move(computation), GetStringDevices(), &result_shape);
  }
  // Collect the computation client data vector.
  DataBatchVector raw_grad_outputs_data =
      GetDataBatchVector(&raw_grad_outputs, &zero_input);

  TensorBatchVector grad_inputs =
      Execute(*backward_computation_, raw_grad_outputs_data);

  ApplyGradients(grad_inputs, &inputs_, &optimizable_params_,
                 inputs_require_grad_, *gradient_.df);
  // Release handles to saved / captured inputs and outputs.
  captured_outputs_.clear();
  captured_inputs_outputs_.clear();
}

void XlaModule::ApplyGradients(const TensorBatchVector& grad_inputs,
                               TensorBatchVector* inputs,
                               TensorBatchVector* optimizable_params,
                               const std::vector<bool>& inputs_require_grad,
                               const torch::jit::Graph& df) {
  size_t inputs_require_grad_count =
      std::count(inputs_require_grad.begin(), inputs_require_grad.end(), true);
  for (size_t i = 0; i < inputs->size(); ++i) {
    auto& replica_grad_inputs = grad_inputs[i];
    auto& replica_inputs = (*inputs)[i];
    auto& replica_optimizable_params = (*optimizable_params)[i];
    XLA_CHECK_EQ(replica_grad_inputs.size(), inputs_require_grad_count)
        << "Graph:\n"
        << df.toString();
    size_t grad_index = 0;
    for (size_t j = 0; j < replica_inputs.size(); j++) {
      if (inputs_require_grad[j]) {
        replica_inputs[j].SetGradient(replica_grad_inputs[grad_index]);
        ++grad_index;
      }
    }
    for (size_t j = 0; j < replica_optimizable_params.size(); j++) {
      replica_optimizable_params[j].SetGradient(
          replica_grad_inputs[grad_index]);
      ++grad_index;
    }
  }
}

XlaModule::TensorBatchVector XlaModule::RunFusedTrain(
    const TensorBatchVector& inputs) {
  Initialize(inputs);

  TensorBatchVector inputs_params_buffers = PrepareForwardInput(inputs);
  DataBatchVector inputs_params_buffers_data =
      GetDataBatchVector(&inputs_params_buffers, /*zero_input=*/nullptr);
  if (forward_computation_ == nullptr) {
    // Shapes are going to be the same for all replicas, so use the ones of the
    // first replica here.
    const TensorBatchVector::value_type& replica_inputs =
        inputs_params_buffers.front();
    std::vector<XlaTranslator::ParameterShape> forward_shapes;
    for (size_t i = 0; i < replica_inputs.size(); ++i) {
      forward_shapes.push_back(XlaTranslator::ParameterShape(
          replica_inputs[i].shape(),
          XlaTranslator::ParameterKind::kGraphInput));
    }
    xla::XlaComputation computation =
        BuildFusedTrainComputation(forward_shapes);
    xla::Shape result_shape = GetResultShape(computation, inputs);
    forward_computation_ = xla::ComputationClient::Get()->Compile(
        std::move(computation), GetStringDevices(), &result_shape);
  }

  TensorBatchVector result_components =
      Execute(*forward_computation_, inputs_params_buffers_data);

  // First gradient_.f_real_outputs are the forward outputs returned to user
  // code.
  XLA_CHECK_LE(gradient_.f_real_outputs, result_components.front().size());
  grad_inputs_.clear();
  TensorBatchVector forward_result;
  for (auto& replica_result_components : result_components) {
    TensorBatchVector::value_type replica_forward_result;
    TensorBatchVector::value_type replica_grad_inputs;
    for (size_t j = 0; j < gradient_.f_real_outputs; ++j) {
      replica_forward_result.push_back(replica_result_components[j]);
    }
    for (size_t j = gradient_.f_real_outputs;
         j < replica_result_components.size(); ++j) {
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

xla::XlaComputation XlaModule::BuildFusedTrainComputation(
    const std::vector<XlaTranslator::ParameterShape>& forward_shapes) {
  XlaTranslator xla_fwd_impl(gradient_.f);
  xla::XlaBuilder b("XlaFusedComputation");
  // Build the forward pass program without compiling it, the backward pass
  // needs to be called before finalizing it.
  XlaComputationInOut computation_in_outs =
      xla_fwd_impl.BuildComputationProgram(forward_shapes,
                                           backward_size_op_values_, &b);
  // Take the XLA outputs from the forward pass and set them for the backward
  // call in the same order the standalone, unfused version takes its arguments.
  XLA_CHECK(!computation_in_outs.outputs.empty());
  XLA_CHECK_EQ(gradient_.f_real_outputs, backward_input_gradients_.size());
  std::vector<xla::XlaOp> captured_inputs_outputs;
  for (auto i : gradient_.df_input_captured_inputs) {
    captured_inputs_outputs.push_back(computation_in_outs.inputs[i]);
  }
  for (auto i : gradient_.df_input_captured_outputs) {
    captured_inputs_outputs.push_back(computation_in_outs.outputs[i]);
  }
  backward_size_op_values_ = SetBackwardSizeOpValues(
      computation_in_outs.ret_size_op_values, gradient_);
  // NOTE: The order of the input parameters passed to the BuildComputation()
  // call to build the backward computation is critical, as they have to match
  // the sequence of the graph->inputs() vector. Before the gradients passed in
  // by the user, then then zeroed virtual inputs, and then the captured
  // inputs/outputs.
  std::vector<XlaTranslator::ParameterShape> backward_shapes;
  std::vector<xla::XlaOp> backward_operands;
  for (size_t i = 0; i < backward_input_gradients_.size(); ++i) {
    xla::Shape shape =
        CreateComputationShapeFromTensor(backward_input_gradients_[i],
                                         /*device=*/nullptr);
    xla::Literal literal = GetTensorLiteral(backward_input_gradients_[i],
                                            &shape, /*device=*/nullptr);
    xla::XlaOp gradient_op = xla::ConstantLiteral(&b, literal);
    backward_shapes.push_back(XlaTranslator::ParameterShape(
        XlaHelpers::ShapeOfXlaOp(gradient_op),
        XlaTranslator::ParameterKind::kGraphInput));
    backward_operands.push_back(gradient_op);
  }
  for (size_t input_vjp_idx = backward_input_gradients_.size();
       input_vjp_idx < gradient_.df_input_vjps.size(); ++input_vjp_idx) {
    const auto raw_output_index = gradient_.df_input_vjps[input_vjp_idx];
    XLA_CHECK_LT(raw_output_index, computation_in_outs.outputs.size());
    backward_shapes.push_back(XlaTranslator::ParameterShape(
        XlaHelpers::ShapeOfXlaOp(computation_in_outs.outputs[raw_output_index]),
        XlaTranslator::ParameterKind::kZeroInput));
  }
  for (auto p : captured_inputs_outputs) {
    backward_shapes.push_back(XlaTranslator::ParameterShape(
        XlaHelpers::ShapeOfXlaOp(p),
        XlaTranslator::ParameterKind::kGraphInput));
    backward_operands.push_back(p);
  }
  // The arguments are set up correctly, call into the backward computation.
  XlaTranslator xla_bwd_impl(gradient_.df);
  xla::XlaComputation backward_computation =
      xla_bwd_impl
          .BuildComputation("XlaBackward", backward_shapes,
                            backward_size_op_values_,
                            GetBackwardBuildOptions(inputs_.size()))
          .computation;
  xla::XlaOp backward_op =
      xla::Call(&b, backward_computation, backward_operands);

  // Return the real outputs of the forward, followed by the outputs of the
  // backward.
  std::vector<xla::XlaOp> returned_outputs;
  for (size_t i = 0; i < gradient_.f_real_outputs; ++i) {
    returned_outputs.push_back(computation_in_outs.outputs[i]);
  }
  xla::Shape backward_shape = XlaHelpers::ShapeOfXlaOp(backward_op);
  if (backward_shape.IsTuple()) {
    for (xla::int64 i = 0;
         i < xla::ShapeUtil::TupleElementCount(backward_shape); ++i) {
      returned_outputs.push_back(xla::GetTupleElement(backward_op, i));
    }
  } else if (!xla::ShapeUtil::IsEmptyTuple(backward_shape)) {
    returned_outputs.push_back(backward_op);
  }
  XlaHelpers::CreateReturnValue(&b, returned_outputs);

  xla::XlaComputation computation = b.Build().ValueOrDie();
  TF_VLOG(5) << "Fused computation:\n"
             << ConsumeValue(xla::xrt_util::GetComputationHloText(computation));
  return computation;
}

XlaModule::TensorBatchVector XlaModule::RunUnfusedForward(
    const TensorBatchVector& inputs) {
  TensorBatchVector inputs_params_buffers = PrepareForwardInput(inputs);
  DataBatchVector inputs_params_buffers_data =
      GetDataBatchVector(&inputs_params_buffers, /*zero_input=*/nullptr);

  // Lazy-convert forward graph to XlaComputation.
  if (forward_computation_ == nullptr) {
    // Shapes are going to be the same for all replicas, so use the ones of the
    // first replica here.
    std::vector<XlaTranslator::ParameterShape> forward_shapes;
    for (auto& p : inputs_params_buffers.front()) {
      forward_shapes.push_back(XlaTranslator::ParameterShape(
          p.shape(), XlaTranslator::ParameterKind::kGraphInput));
    }

    XlaTranslator xla_fwd_impl(gradient_.f);
    XlaTranslationResult forward_translation_result =
        xla_fwd_impl.BuildComputation("XlaForward", forward_shapes,
                                      backward_size_op_values_);
    backward_size_op_values_ = SetBackwardSizeOpValues(
        forward_translation_result.ret_size_op_values, gradient_);

    xla::Shape result_shape =
        GetResultShape(forward_translation_result.computation, inputs);
    forward_computation_ = xla::ComputationClient::Get()->Compile(
        std::move(forward_translation_result.computation), GetStringDevices(),
        &result_shape);
  }

  TensorBatchVector raw_outputs =
      Execute(*forward_computation_, inputs_params_buffers_data);

  TensorBatchVector outputs;
  for (size_t i = 0; i < raw_outputs.size(); ++i) {
    auto& replica_raw_outputs = raw_outputs[i];
    TensorBatchVector::value_type replica_outputs;
    for (size_t j = 0; j < gradient_.f_real_outputs; j++) {
      replica_outputs.push_back(replica_raw_outputs[j]);
    }
    outputs.push_back(std::move(replica_outputs));

    TensorBatchVector::value_type replica_captured_outputs;
    for (size_t j = gradient_.f_real_outputs; j < replica_raw_outputs.size();
         j++) {
      replica_captured_outputs.push_back(replica_raw_outputs[j]);
    }
    captured_outputs_.push_back(std::move(replica_captured_outputs));

    auto& replica_inputs_params_buffers = inputs_params_buffers[i];
    TensorBatchVector::value_type replica_captured_inputs_outputs;
    for (auto j : gradient_.df_input_captured_inputs) {
      replica_captured_inputs_outputs.push_back(
          replica_inputs_params_buffers[j]);
    }
    for (auto j : gradient_.df_input_captured_outputs) {
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

  if (inputs_.empty()) {
    inputs_ = inputs;
  } else {
    ReferenceNewTensorData(inputs, &inputs_);
  }

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

std::vector<std::string> XlaModule::GetStringDevices() const {
  std::vector<std::string> devices(devices_.size());
  for (size_t i = 0; i < devices_.size(); ++i) {
    devices[i] = devices_[i].ToString();
  }
  return devices;
}

XlaModule::TensorBatchVector XlaModule::Execute(
    const xla::ComputationClient::Computation& computation,
    const DataBatchVector& inputs) {
  std::vector<std::vector<xla::ComputationClient::DataPtr>> exec_results;
  if (inputs.size() == 1) {
    xla::ComputationClient::ExecuteComputationOptions options;
    exec_results.push_back(xla::ComputationClient::Get()->ExecuteComputation(
        computation, inputs.front(), computation.devices()[0], options));
  } else {
    xla::ComputationClient::ExecuteReplicatedOptions options;
    exec_results = xla::ComputationClient::Get()->ExecuteReplicated(
        computation, inputs, computation.devices(), options);
  }
  return CreateResultBatchVector(std::move(exec_results));
}

XlaTranslator::BuildOptions XlaModule::GetBackwardBuildOptions(
    size_t num_replicas) {
  XlaTranslator::BuildOptions options;
  if (num_replicas > 1) {
    options.output_transform = [num_replicas](const xla::XlaOp& op, size_t) {
      double scale = 1.0 / num_replicas;
      return BuildCrossReplicaSum(op, scale, {});
    };
  }
  return options;
}

void XlaModule::FlushTensorsOperations() {
  // We might have to do something smarter here, as we are syncing even tensors
  // which are not part of the traning loop. Nothing happens, but if we want to
  // fuse the sync operation with the forward+backward+optimizer, we need to
  // have a path leading to the same XLA computation.
  std::vector<XLATensor> tensors = XLATensor::GetLiveTensors();
  XLATensor::ApplyPendingGraph(&tensors);
}

void XlaModule::ReferenceNewTensorData(const TensorBatchVector& source,
                                       TensorBatchVector* dest) {
  XLA_CHECK_EQ(source.size(), dest->size());
  for (size_t i = 0; i < source.size(); ++i) {
    const TensorBatchVector::value_type& replica_source = source[i];
    TensorBatchVector::value_type* replica_dest = &(*dest)[i];
    XLA_CHECK_EQ(replica_source.size(), replica_dest->size());
    for (size_t j = 0; j < replica_source.size(); ++j) {
      (*replica_dest)[j].ReferenceDataFrom(replica_source[j]);
    }
  }
}

XlaComputationInOut::SizeOpValues XlaModule::SetBackwardSizeOpValues(
    const XlaComputationInOut::SizeOpValues& ret_size_op_values,
    const torch::jit::Gradient& gradient) {
  size_t backward_input_idx = 0;
  XlaComputationInOut::SizeOpValues backward_size_op_values;
  for (const auto out_idx : gradient.df_input_vjps) {
    const auto ret_size_op_value_it = ret_size_op_values.find(out_idx);
    if (ret_size_op_value_it != ret_size_op_values.end()) {
      const auto it_ok = backward_size_op_values.emplace(
          backward_input_idx, ret_size_op_value_it->second);
      XLA_CHECK(it_ok.second)
          << "Duplicated backward_input_idx: " << backward_input_idx;
    }
    ++backward_input_idx;
  }
  backward_input_idx += gradient.df_input_captured_inputs.size();
  for (const auto out_idx : gradient.df_input_captured_outputs) {
    const auto ret_size_op_value_it = ret_size_op_values.find(out_idx);
    if (ret_size_op_value_it != ret_size_op_values.end()) {
      const auto it_ok = backward_size_op_values.emplace(
          backward_input_idx, ret_size_op_value_it->second);
      XLA_CHECK(it_ok.second)
          << "Duplicated backward_input_idx: " << backward_input_idx;
    }
    ++backward_input_idx;
  }
  return backward_size_op_values;
}

XlaModule::DataBatchVector XlaModule::GetDataBatchVector(
    TensorBatchVector* inputs, const std::vector<bool>* zero_input) {
  DataBatchVector inputs_data;
  for (auto& replica_inputs : *inputs) {
    DataBatchVector::value_type replica_inputs_data;
    for (size_t j = 0; j < replica_inputs.size(); ++j) {
      if (zero_input == nullptr || !zero_input->at(j)) {
        replica_inputs_data.push_back(replica_inputs[j].GetXlaData());
      }
    }
    inputs_data.push_back(std::move(replica_inputs_data));
  }
  return inputs_data;
}

std::vector<Device> XlaModule::CommonDevicesForReplicas(
    const TensorBatchVector& inputs) {
  std::vector<Device> devices;
  std::set<Device> unique_devices;
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
  xla::ProgramShape program_shape = computation.GetProgramShape().ValueOrDie();
  return MakeShapeWithDeviceLayout(program_shape.result(),
                                   devices.front().hw_type);
}

}  // namespace torch_xla
