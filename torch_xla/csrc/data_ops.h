#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "torch/csrc/jit/ir.h"

// Collection of XLA lowerings for operations which only involve some form of
// data movement and no computation.

namespace torch {
namespace jit {

// Creates a new tensor with the same data as the input tensor and the size
// specified by the "size" attribute of the given node.
xla::XlaOp BuildView(const Node* node, const xla::XlaOp& input);

// Creates a new tensor with the singleton dimensions expanded to the sizes
// specified by the "size" attribute of the given node.
xla::XlaOp BuildExpand(const Node* node, const xla::XlaOp& input);

// Concatenates a list of tensors along a new dimension specified by the "dim"
// attribute of the given node.
xla::XlaOp BuildStack(const Node* node,
                      const std::function<xla::XlaOp(const Value*)>& node_op,
                      xla::XlaBuilder* b);

// Concatenates a list of tensors along an existing dimension specified by the
// "dim" attribute of the given node.
xla::XlaOp BuildCat(const Node* node,
                    const std::function<xla::XlaOp(const Value*)>& node_op,
                    xla::XlaBuilder* b);

// Splits a tensor into a specific number of chunks specified by the "chunks"
// attribute of the given node, along an existing dimension specified by the
// "dim" attribute of the given node.
std::vector<xla::XlaOp> BuildChunk(const Node* node, const xla::XlaOp& input);

}  // namespace jit
}  // namespace torch
