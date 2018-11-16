#pragma once

#include <initializer_list>

#include "tensor.h"
#include "torch/csrc/jit/script/module.h"
#include "torch/csrc/utils/disallow_copy.h"
#include "translator.h"

#include <atomic>
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

  static constexpr uint64_t kInvalidModuleId = 0;

 private:
  // The i-th entry in this vector, is a vector of XLA computation data which
  // belong the i-th replica.
  using DataBatchVector =
      std::vector<std::vector<xla::ComputationClient::Data*>>;

  void Initialize(const TensorBatchVector& inputs);

  void CheckInitialized() const;

  xla::PrecisionConfig::Precision GetPrecisionConfig() const;

  // Builds the fused forward and backward computation for RunFusedTrain.
  void BuildFusedTrainComputation(
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
  // return. The module_id is used to track changes in the tensors taking place
  // of the fused computation, and will be assigned to the output tensors.
  TensorBatchVector Execute(const xla::XlaComputation& computation,
                            const DataBatchVector& inputs,
                            const xla::Shape& result_shape, uint64_t module_id);

  // Creates the build options to be used to create a backward pass computation.
  XlaTranslator::BuildOptions GetBackwardBuildOptions(
      size_t param_to_return_count, size_t num_replicas);

  // Sets the gradients of the optimizeable inputs and the optimizable
  // parameters, according to the grad_inputs values. The inputs_require_grad
  // vector tell which inputs requires the gradient to be updated.
  static void ApplyGradients(const TensorBatchVector& grad_inputs,
                             const TensorBatchVector& inputs,
                             const TensorBatchVector& optimizable_params,
                             const std::vector<bool>& inputs_require_grad,
                             const Graph& df);

  static void FlushTensorsOperations(
      std::initializer_list<const TensorBatchVector*> batch_tensors);

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

  // The module parameters which are marked for being subject to optimization.
  TensorBatchVector optimizable_params_;
  // All the module parameters (which include the optimizable_params_ ones).
  TensorBatchVector all_params_;
  c10::optional<xla::XlaComputation> forward_computation_;
  c10::optional<xla::Shape> forward_shape_;
  c10::optional<xla::XlaComputation> backward_computation_;
  c10::optional<xla::Shape> backward_shape_;

  std::shared_ptr<Graph> f_;
  std::shared_ptr<Graph> df_;

  // info for backwrd captures
  size_t f_real_outputs_;
  std::vector<size_t> df_input_captured_inputs_;
  std::vector<size_t> df_input_captured_outputs_;

  // TODO: captured_outputs only needs shape, no need for holding onto full
  // Tensor
  TensorBatchVector inputs_;
  std::vector<bool> inputs_require_grad_;
  TensorBatchVector captured_outputs_;
  TensorBatchVector captured_inputs_outputs_;

  // Specifies whether to use the highest precision available for convolutions.
  // Currently it only makes a difference for TPUs.
  const bool use_full_conv_precision_;
  // Gradients set aside by the fused train computation, to be consumed by the
  // backward call if we receive an unmodified tensor from the forward pass.
  TensorBatchVector grad_inputs_;
  // Keeps track whether an attempt to fuse the forward and backward
  // computations failed. Starts on true (we attempt to fuse), permanently goes
  // to false on failure. Mitigates doing redundant work (compute gradients we
  // can't use) after the first training step, if the fusion fails.
  bool enable_trace_fusion_;
  // Whether to differentiate the graph or not.
  bool differentiate_;
  // Unique identifier for the module, used to keep track of tensors originating
  // from its forward method.
  uint64_t module_id_;

  // Keep the script module alive for lazy initialization of this XlaModule.
  // Once this XlaModule is initialized, script_module_ will be set to null.
  std::shared_ptr<script::Module> script_module_;

  static std::atomic<uint64_t> s_module_id_;
};

}  // namespace jit
}  // namespace torch
