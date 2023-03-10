#ifndef XLA_TORCH_XLA_CSRC_LOWERING_CONTEXT_H_
#define XLA_TORCH_XLA_CSRC_LOWERING_CONTEXT_H_

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/backend/lowering_context.h>
#include <torch/csrc/lazy/core/ir_util.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/tsl/platform/macros.h"
#include "third_party/xla_client/computation_client.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ir_util.h"

namespace torch_xla {

class LoweringContext : public torch::lazy::LoweringContext {
 public:
  explicit LoweringContext(const std::string& name,
                           torch::lazy::BackendDevice device);
  LoweringContext(const std::string& name, torch::lazy::BackendDevice device,
                  c10::ArrayRef<const torch::lazy::Node*> post_order,
                  torch::lazy::Util::EmissionMap emit_status);

  xla::XlaBuilder* builder() { return &builder_; }

  const torch::lazy::BackendDevice& device() const { return device_; };

  // If a parameter associated with data has already been declared, it will be
  // returned. Otherwise a new one will be created, associated with the tensor
  // held in data.
  xla::XlaOp GetParameter(
      const std::shared_ptr<torch::lazy::BackendData>& data);

  // Retrieves the vector holding all the tensors associated with the parameter
  // instructions which have been created.
  const std::vector<torch::lazy::BackendDataPtr>& GetParametersData() const;

  const std::vector<size_t>& GetParameterSequence() const;

  xla::XlaOp GetResult(size_t index) const;

  void SetResult(size_t index, xla::XlaOp op);

  // Assigns the given XLA operation to the specified output. As outputs are
  // lowered in a post-order fashion, later nodes should always find their
  // operands among the emitted outputs.
  void AssignOutputOp(const torch::lazy::Output& output, xla::XlaOp op);

  // Retrieves the lowered operation for a output. If the requested output is
  // not available yet, the graph behind the output's XlaNode is lowered, and
  // the corresponding XLA operation returned.
  xla::XlaOp GetOutputOp(const torch::lazy::Output& output);

  // Build the XLA computation capturing all the operations created with the
  // embedded XLA builder (returned by the builder() API).
  xla::StatusOr<xla::XlaComputation> BuildXla();

  // Build the XLA computation capturing all the operations created with the
  // embedded XLA builder (returned by the builder() API).
  // Uses root as return value for the computation. It is an error to use this
  // API after having called the AddResult() API.
  xla::StatusOr<xla::XlaComputation> BuildXla(xla::XlaOp root);

  // Lowers a single IR node. All the inputs to the node must have a lowering
  // before calling this API. Returns the generated XLA operations.
  XlaOpVector LowerNode(const torch::lazy::Node* node);

  void SetUpAlias(const std::vector<int64_t>& output_index,
                  int64_t param_number, const std::vector<int64_t>& param_index,
                  bool must_alias = false) override;

  bool CheckResultShape(const torch::lazy::BackendDataPtr& parameter_data,
                        size_t result_idx) override;

  size_t AddResult(const torch::lazy::Output& output) override;

  size_t AddResult(xla::XlaOp op);

  void AddParameter(const torch::lazy::Output& output, size_t index,
                    const torch::lazy::Shape& shape,
                    const std::string& name) override;

  torch::lazy::ComputationPtr Build() override;

  const OutputMap<xla::XlaOp> GetEmittedOutputs() const {
    return emitted_outputs_;
  }

 private:
  struct Parameter {
    xla::XlaOp param;
    size_t index = 0;
  };

  // Reports an XLA builder error for the given node.
  TF_ATTRIBUTE_NORETURN void ReportBuilderError(const torch::lazy::Node* node,
                                                const char* error_msg);

  xla::XlaBuilder builder_;
  std::unordered_map<torch::lazy::BackendData::Handle, Parameter>
      parameters_map_;
  std::vector<xla::XlaOp> root_tuple_;
  OutputMap<xla::XlaOp> emitted_outputs_;
};  // namespace torch_xla

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_LOWERING_CONTEXT_H_