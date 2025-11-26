#ifndef XLA_TORCH_XLA_CSRC_LOWERING_CONTEXT_H_
#define XLA_TORCH_XLA_CSRC_LOWERING_CONTEXT_H_

#include <c10/util/ArrayRef.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/backend/lowering_context.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_metadata.h>
#include <torch/csrc/lazy/core/ir_util.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class StackFrameIndexBuilder;

class LoweringContext : public torch::lazy::LoweringContext {
 public:
  explicit LoweringContext(const std::string& name,
                           torch::lazy::BackendDevice device);
  LoweringContext(const std::string& name, torch::lazy::BackendDevice device,
                  c10::ArrayRef<const torch::lazy::Node*> post_order,
                  torch::lazy::Util::EmissionMap emit_status);

  xla::XlaBuilder* builder() { return &builder_; }

  void set_name_string(const std::string& name) { name_ = name; }

  const std::string& get_name_string() { return name_; }

  StackFrameIndexBuilder* stack_frame_index_builder() {
    return stack_frame_index_builder_.get();
  }

  const torch::lazy::BackendDevice& device() const { return device_; };

  // If a parameter associated with data has already been declared, it will be
  // returned. Otherwise a new one will be created, associated with the tensor
  // held in data.
  absl::StatusOr<xla::XlaOp> GetParameter(
      const torch::lazy::BackendDataPtr& backend_data,
      const std::unordered_set<uint32_t>& unbounded_dynamic_dims = {});

  // If a parameter associated with data has already been declared, returns its
  // ID. Otherwise, returns `std::nullopt`.
  std::optional<size_t> GetParameterId(
      const std::shared_ptr<torch::lazy::BackendData>& backend_data) const;

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

  // Retrieves the lowered operation for a output.
  //
  // If the requested output is not available yet, the graph behind the output's
  // XlaNode is lowered, and the corresponding XLA operation returned.
  [[deprecated("Use SafeGetOutputOp for better error handling.")]] xla::XlaOp
  GetOutputOp(const torch::lazy::Output& output);
  // Retrieves the lowered operation for a output.
  //
  // If the requested output is not available yet, the graph behind the output's
  // XlaNode is lowered, and the corresponding XLA operation returned.
  //
  // This function shall return an error status if the lowering the underlying
  // `output`, or any other dependent nodes, returns an error status.
  // Additionally, it might abort if after the lowering of `output` and its
  // dependent nodes, the lowered node for `output` is not available, i.e. not
  // in `emitted_outputs_`.
  absl::StatusOr<xla::XlaOp> SafeGetOutputOp(const torch::lazy::Output& output);

  // Build the XLA computation capturing all the operations created with the
  // embedded XLA builder (returned by the builder() API).
  absl::StatusOr<xla::XlaComputation> BuildXla();

  // Build the XLA computation capturing all the operations created with the
  // embedded XLA builder (returned by the builder() API).
  // Uses root as return value for the computation. It is an error to use this
  // API after having called the AddResult() API.
  absl::StatusOr<xla::XlaComputation> BuildXla(xla::XlaOp root);

  // Lowers a single IR node. All the inputs to the node must have a lowering
  // before calling this API. Returns the generated XLA operations.
  absl::StatusOr<XlaOpVector> LowerNode(const torch::lazy::Node& node);

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

  const torch::lazy::OutputMap<xla::XlaOp> GetEmittedOutputs() const {
    return emitted_outputs_;
  }

  // Return stack frame id
  int64_t AddStackFrameLocation(const torch::lazy::SourceLocation& source,
                                int64_t parent_id);

 private:
  struct Parameter {
    xla::XlaOp param;
    size_t index = 0;
  };

  // Checks whether the given output is already emitted. In other words, whether
  // we can find it inside `emitted_outputs_`.
  absl::Status CheckOutputIsEmitted(const torch::lazy::Output& output) const;

  xla::XlaBuilder builder_;
  std::unordered_map<torch::lazy::BackendData::Handle, Parameter>
      parameters_map_;
  std::vector<xla::XlaOp> root_tuple_;
  torch::lazy::OutputMap<xla::XlaOp> emitted_outputs_;
  std::string name_;

  std::shared_ptr<StackFrameIndexBuilder> stack_frame_index_builder_;
};  // namespace torch_xla

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_LOWERING_CONTEXT_H_
