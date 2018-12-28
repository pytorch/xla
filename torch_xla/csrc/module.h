#pragma once

#include <initializer_list>

#include "tensor.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/script/module.h"
#include "torch/csrc/utils/disallow_copy.h"
#include "translator.h"

#include <atomic>
#include <map>
#include <memory>

namespace torch {
namespace jit {

struct XlaModule : public std::enable_shared_from_this<XlaModule> {
  TH_DISALLOW_COPY_AND_ASSIGN(XlaModule);

  // The i-th entry in this vector, is a vector of XLA tensors which belong the
  // i-th replica.
  using TensorBatchVector =
      std::vector<std::vector<std::shared_ptr<XLATensor>>>;

  // Creates a new XlaModule from a PyTorch script module "module".
  // "use_full_conv_precision" controls whether to use maximum precision
  // available in hardware for convolutions.
  XlaModule(const std::shared_ptr<script::Module> module,
            bool use_full_conv_precision, bool differentiate);

  TensorBatchVector forward(const TensorBatchVector& inputs);
  // For the given gradient outputs, compute the gradient of input and
  // parameters and set it as their grad field.
  void backward(const TensorBatchVector& grad_outputs);

  const TensorBatchVector& parameters();
  const TensorBatchVector& parameters_buffers();

  // When we want to enable forward+backward fusion, we need to feed the
  // backward with the gradient of the output.
  // For example, say that we have:
  //
  //   def wrapped_model(model, loss_fn, input, target):
  //     output = model(input)
  //     return loss_fn(output, target)
  //
  // The output of the forward is the loss, but the input of the backward must
  // be the derivative of the loss WRT itself (which is 1s).
  // The following API allows to set the gradients which are the input of the
  // backward, when we first create the fused computation.
  void SetInputGradientsForFusion(std::vector<at::Tensor> gradients);

 private:
  // The i-th entry in this vector, is a vector of XLA computation data which
  // belong the i-th replica.
  using DataBatchVector =
      std::vector<std::vector<xla::ComputationClient::Data*>>;

  void Initialize(const TensorBatchVector& inputs);

  void CheckInitialized() const;

  xla::PrecisionConfig::Precision GetPrecisionConfig() const;

  // Retrieves the module devices as vector of strings representations, so that
  // it can be passed to the computation client API.
  std::vector<std::string> GetStringDevices() const;

  // Builds the fused forward and backward computation for RunFusedTrain.
  xla::XlaComputation BuildFusedTrainComputation(
      const std::vector<XlaTranslator::ParameterShape>& forward_shapes);

  // Runs the original, unfused forward computation on the given inputs.
  TensorBatchVector RunUnfusedForward(const TensorBatchVector& inputs);

  // Runs a fused forward and backward computation. Takes the same input as the
  // forward computation, returns the outputs for the forward computation and
  // the gradients for model parameters.
  TensorBatchVector RunFusedTrain(const TensorBatchVector& inputs);

  // Collect the inputs and model parameters and clear the captured inputs /
  // outputs state.
  TensorBatchVector PrepareForwardInput(const TensorBatchVector& inputs);

  // Executes the provided XLA computation. The execution will be replicated in
  // as many replicas as the size of the inputs first dimension.
  // The result_shape is the shape+layout which we want the computation to
  // return.
  TensorBatchVector Execute(
      const xla::ComputationClient::Computation& computation,
      const DataBatchVector& inputs);

  // Creates the build options to be used to create a backward pass computation.
  XlaTranslator::BuildOptions GetBackwardBuildOptions(size_t num_replicas);

  // Makes sure the XLA tensors partecipating to the forward/backward
  // computation have their accumulated operations sync to device memory.
  void FlushTensorsOperations();

  static void RunForwardPasses(std::shared_ptr<Graph>* graph);

  // Computes the gradient structure of the given graph, and runs all the
  // appropriate passes over the resulting forward and backward graphs.
  static Gradient ComputeGradient(const std::shared_ptr<Graph>& graph);

  // Propagate ret_size_op_values storing the aten::size values collected during
  // forward pass translation to the backward pass. Uses the capture information
  // from the gradient descriptor.
  static XlaComputationInOut::SizeOpValues SetBackwardSizeOpValues(
      const XlaComputationInOut::SizeOpValues& ret_size_op_values,
      const Gradient& gradient);

  // Sets the gradients of the optimizeable inputs and the optimizable
  // parameters, according to the grad_inputs values. The inputs_require_grad
  // vector tell which inputs requires the gradient to be updated.
  static void ApplyGradients(const TensorBatchVector& grad_inputs,
                             const TensorBatchVector& inputs,
                             const TensorBatchVector& optimizable_params,
                             const std::vector<bool>& inputs_require_grad,
                             const Graph& df);

  // Makes the data references in dest point to the ones in source.
  static void ReferenceNewTensorData(const TensorBatchVector& source,
                                     TensorBatchVector* dest);

  // Extracts the XLA computation data from the inputs, and returns a matching
  // batch vector where data[i][j] holds the data beind the XLA tensor
  // inputs[i][j].
  // Elements in the return vector are populated only if zero_input is nullptr,
  // or if zero_input[j] is false.
  static DataBatchVector GetDataBatchVector(
      const TensorBatchVector& inputs, const std::vector<bool>* zero_input);

  // Returns the common device for every replica copy of the inputs.
  // All common devices must be different in different replicas.
  static std::vector<XLATensor::Device> CommonDevicesForReplicas(
      const TensorBatchVector& inputs);

  // Computes the optimal result shape for a given computation and inputs.
  static xla::Shape GetResultShape(const xla::XlaComputation& computation,
                                   const TensorBatchVector& input_tensors);

  // The devices where the replicas should be running. Replica 'i' on
  // devices_[i].
  std::vector<XLATensor::Device> devices_;
  // The module parameters which are marked for being subject to optimization.
  TensorBatchVector optimizable_params_;
  // All the module parameters (which include the optimizable_params_ ones).
  TensorBatchVector all_params_;

  std::shared_ptr<xla::ComputationClient::Computation> forward_computation_;
  std::shared_ptr<xla::ComputationClient::Computation> backward_computation_;
  XlaComputationInOut::SizeOpValues backward_size_op_values_;

  // Information needed to connect the forward and backward graphs.
  Gradient gradient_;

  TensorBatchVector inputs_;
  std::vector<bool> inputs_require_grad_;
  TensorBatchVector captured_outputs_;
  TensorBatchVector captured_inputs_outputs_;

  // The optional input gradients for a fused backward, set by the
  // SetInputGradientsForFusion() API.
  std::vector<at::Tensor> backward_input_gradients_;

  // The context used in FlushTensorsOperations() to be passed to the
  // XLATensor::ApplyPendingGraph() API, to register the computation and tensor
  // information of the last apply operation. The XLATensor::ApplyPendingGraph()
  // API will use that to avoid re-building and re-compiling the XLA computation
  // required for the apply.
  XLATensor::ApplyContext apply_context_;

  // Specifies whether to use the highest precision available for convolutions.
  // Currently it only makes a difference for TPUs.
  const bool use_full_conv_precision_;
  // Gradients set aside by the fused train computation, to be consumed by the
  // backward call if we receive an unmodified tensor from the forward pass.
  TensorBatchVector grad_inputs_;
  // Whether to differentiate the graph or not.
  bool differentiate_;

  // Keep the script module alive for lazy initialization of this XlaModule.
  // Once this XlaModule is initialized, script_module_ will be set to null.
  std::shared_ptr<script::Module> script_module_;
};

}  // namespace jit
}  // namespace torch
