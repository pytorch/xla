#pragma once

#include <map>
#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace torch {
namespace jit {

// Tracks an evolving XLA computation.
class XlaGraphContext {
 public:
  XlaGraphContext() : builder_("XlaGraphContext") {}

  xla::XlaBuilder* builder() { return &builder_; }

  // If a parameter associated with data has already been declared, it will be
  // returned. Otherwise a new one will be created, associated with the tensor
  // held in data.
  xla::XlaOp GetParameter(
      const std::shared_ptr<xla::ComputationClient::Data>& data);

  // Retrieves the vector holding all the tensors associated with the parameter
  // instructions which have been created.
  std::vector<xla::ComputationClient::Data*> GetParametersData() const;

  // Adds the output of a given operation to the result tuple.
  xla::int64 AddResult(xla::XlaOp op);

  // Build the XLA computation capturing all the operations created with the
  // embedded XLA builder (returned by the builder() API).
  xla::StatusOr<xla::XlaComputation> Build();

  // Build the XLA computation capturing all the operations created with the
  // embedded XLA builder (returned by the builder() API).
  // Uses root as return value forthe computation. It is an error to use this
  // API after having called the AddResult() API.
  xla::StatusOr<xla::XlaComputation> Build(const xla::XlaOp& root);

 private:
  xla::XlaBuilder builder_;
  std::vector<std::shared_ptr<xla::ComputationClient::Data>> parameters_;
  std::map<xla::ComputationClient::Data*, xla::XlaOp> parameters_map_;
  std::vector<xla::XlaOp> root_tuple_;
};

// A class whose task is to encapsulate the generation of an XLA operation.
class XlaGraphNode {
 public:
  // The generation function used by the XLA tensors to create xla::XlaOp nodes.
  using Generator = std::function<xla::StatusOr<xla::XlaOp>(
      XlaGraphContext*, const XlaGraphNode&)>;

  static std::shared_ptr<XlaGraphNode> New(
      Generator generator, xla::Shape shape,
      tensorflow::gtl::ArraySlice<const std::shared_ptr<XlaGraphNode>> inputs) {
    return std::make_shared<XlaGraphNode>(std::move(generator),
                                          std::move(shape), std::move(inputs));
  }

  XlaGraphNode(
      Generator generator, xla::Shape shape,
      tensorflow::gtl::ArraySlice<const std::shared_ptr<XlaGraphNode>> inputs);

  // Runs the generator function using the ctx argument, and returns the XLA
  // operation which is the end result of the generation.
  xla::StatusOr<xla::XlaOp> Generate(XlaGraphContext* ctx) const {
    return generator_(ctx, *this);
  }

  const xla::Shape& shape() const { return shape_; }

  const std::shared_ptr<XlaGraphNode>& input(xla::int64 ordinal) const {
    return inputs_[ordinal];
  }

  xla::int64 graph_size() const { return graph_size_; }

 private:
  Generator generator_;
  xla::Shape shape_;
  std::vector<std::shared_ptr<XlaGraphNode>> inputs_;
  xla::int64 graph_size_ = 1;
};

}  // namespace jit
}  // namespace torch
