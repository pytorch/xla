#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/core/platform/macros.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ir_util.h"

namespace torch_xla {
namespace ir {

class LoweringContext {
 public:
  explicit LoweringContext(const std::string& name) : builder_(name) {}

  xla::XlaBuilder* builder() { return &builder_; }

  // If a parameter associated with data has already been declared, it will be
  // returned. Otherwise a new one will be created, associated with the tensor
  // held in data.
  xla::XlaOp GetParameter(
      const std::shared_ptr<xla::ComputationClient::Data>& data);

  // Retrieves the vector holding all the tensors associated with the parameter
  // instructions which have been created.
  const std::vector<xla::ComputationClient::DataPtr>& GetParametersData() const;

  // Adds the output of a given operation to the result tuple. Returns the index
  // of the output within the tuple.
  size_t AddResult(xla::XlaOp op);

  xla::XlaOp GetResult(size_t index) const;

  void SetResult(size_t index, xla::XlaOp op);

  // Assigns the given XLA operation to the specified output. As outputs are
  // lowered in a post-order fashion, later nodes should always find their
  // operands among the emitted outputs.
  void AssignOutputOp(const Output& output, xla::XlaOp op);

  // Retrieves the lowered operation for a output. If the requested output is
  // not available yet, the graph behind the output's Node is lowered, and the
  // corresponding XLA operation returned.
  xla::XlaOp GetOutputOp(const Output& output);

  // Build the XLA computation capturing all the operations created with the
  // embedded XLA builder (returned by the builder() API).
  xla::StatusOr<xla::XlaComputation> Build();

  // Build the XLA computation capturing all the operations created with the
  // embedded XLA builder (returned by the builder() API).
  // Uses root as return value forthe computation. It is an error to use this
  // API after having called the AddResult() API.
  xla::StatusOr<xla::XlaComputation> Build(xla::XlaOp root);

  // Lowers a single IR node. All the inputs to the node must have a lowering
  // before calling this API. Returns the generated XLA operations.
  XlaOpVector LowerNode(const Node* node);

  size_t GetEmittedNodeCount() const { return emit_status_.size(); }

 private:
  // Reports an XLA builder error for the given node.
  TF_ATTRIBUTE_NORETURN void ReportBuilderError(const Node* node,
                                                const char* error_msg);

  xla::XlaBuilder builder_;
  std::vector<xla::ComputationClient::DataPtr> parameters_;
  std::unordered_map<xla::ComputationClient::Data::OpaqueHandle, xla::XlaOp>
      parameters_map_;
  std::vector<xla::XlaOp> root_tuple_;
  OutputMap<xla::XlaOp> emitted_outputs_;
  Util::EmissionMap emit_status_;
};

}  // namespace ir
}  // namespace torch_xla
