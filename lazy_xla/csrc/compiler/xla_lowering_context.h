#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "torch_xla/csrc/compiler/node_lowering.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_lazy_tensors {
namespace compiler {

using XlaOpVector = tensorflow::gtl::InlinedVector<xla::XlaOp, 1>;

namespace xla_backend {

class XlaLoweringContext : public ir::LoweringContext {
 public:
  XlaLoweringContext(const std::string& name, Device device)
      : ir::LoweringContext(name, device), builder_(name) {}

  XlaLoweringContext(const std::string& name, Device device,
                     absl::Span<const ir::Node* const> post_order,
                     ir::Util::EmissionMap emit_status)
      : ir::LoweringContext(name, device, post_order, emit_status),
        builder_(name) {
    auto lowering = NodeLowering::Create(this);
    for (auto node : post_order) {
      bool ok = lowering->Lower(node);
      XLA_CHECK(ok) << "Failed to lower: " << *node;
    }
  }

  lazy_tensors::Shape GetResultShape(size_t index) const override;

  size_t AddResult(const ir::Output& output) override;

  void LowerNodeToResult(const ir::Node* node) override;

  void AddParameter(const ir::Output& output, size_t index,
                    const lazy_tensors::Shape& shape,
                    const std::string& name) override;

  xla::XlaBuilder* builder() { return &builder_; }

  lazy_tensors::StatusOr<std::shared_ptr<lazy_tensors::GenericComputation>>
  Build() override;

  void SetUpAlias(const lazy_tensors::ShapeIndex& output_index,
                  lazy_tensors::int64 param_number,
                  const lazy_tensors::ShapeIndex& param_index) override;

  // Retrieves the lowered operation for a output. If the requested output is
  // not available yet, the graph behind the output's Node is lowered, and the
  // corresponding XLA operation returned.
  xla::XlaOp GetOutputOp(const ir::Output& output);

  // Assigns the given XLA operation to the specified output. As outputs are
  // lowered in a post-order fashion, later nodes should always find their
  // operands among the emitted outputs.
  void AssignOutputOp(const ir::Output& output, xla::XlaOp op);

  // If a parameter associated with data has already been declared, it will be
  // returned. Otherwise a new one will be created, associated with the tensor
  // held in data.
  xla::XlaOp GetParameter(
      const std::shared_ptr<lazy_tensors::ComputationClient::Data>& data);

  // Reports a builder error for the given node.
  TF_ATTRIBUTE_NORETURN void ReportBuilderError(const ir::Node* node,
                                                const char* error_msg);

 private:
  struct Parameter {
    xla::XlaOp param;
    size_t index = 0;
  };

  // Adds the output of a given operation to the result tuple. Returns the index
  // of the output within the tuple.
  size_t AddResult(xla::XlaOp op);

  xla::XlaOp GetResult(size_t index) const;

  // Lowers a single IR node. All the inputs to the node must have a lowering
  // before calling this API. Returns the generated XLA operations.
  XlaOpVector LowerNode(const ir::Node* node);

  xla::XlaBuilder builder_;
  std::unordered_map<lazy_tensors::ComputationClient::Data::OpaqueHandle,
                     Parameter>
      parameters_map_;
  std::vector<xla::XlaOp> root_tuple_;
  ir::OutputMap<xla::XlaOp> emitted_outputs_;
};

XlaOpVector LowerNodeToXla(const ir::Node* node, XlaLoweringContext* loctx);

}  // namespace xla_backend
}  // namespace compiler
}  // namespace torch_lazy_tensors
