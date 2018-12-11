#include "replace_in_place_ops.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch {
namespace jit {

namespace {

bool IsInPlaceOp(const Node* node) {
  switch (node->kind()) {
    case aten::add_:
    case aten::mul_: {
      return true;
    }
    default:
      break;
  }
  // TODO(asuhan): no interned string for aten::threshold_. Should patch PyTorch
  // core instead.
  return std::string(node->kind().toQualString()) == "aten::threshold_";
}

NodeKind ReplacementForInPlaceOp(const Node* node) {
  XLA_CHECK(IsInPlaceOp(node));
  switch (node->kind()) {
    case aten::add_: {
      return aten::add;
    }
    case aten::mul_: {
      return aten::mul;
    }
    default: { break; }
  }
  XLA_CHECK_EQ(std::string(node->kind().toQualString()), "aten::threshold_")
      << "Invalid node kind: " << node->kind().toQualString();
  return aten::threshold;
}

}  // namespace

void ReplaceInPlaceOps(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub : it->blocks()) {
      ReplaceInPlaceOps(sub);
    }
    if (IsInPlaceOp(*it)) {
      WithInsertPoint guard(*it);
      auto graph = block->owningGraph();
      auto node = *it;
      const auto node_inputs = node->inputs();
      XLA_CHECK(!node_inputs.empty());
      const auto dest = node_inputs[0];
      // The destination definition must be the same block and only have this
      // use. TODO(asuhan): this isn't enough because of aliasing, PyTorch core
      // will handle de-inplace optimization. Until then, use this to unblock
      // our work.
      if (dest->uses().size() != 1 ||
          dest->node()->owningBlock() != node->owningBlock()) {
        continue;
      }
      const auto output_count = node->outputs().size();
      auto replacement_node =
          graph->create(ReplacementForInPlaceOp(*it), output_count);
      graph->insertNode(replacement_node);
      for (const auto node_input : node_inputs) {
        replacement_node->addInput(node_input);
      }
      XLA_CHECK_EQ(replacement_node->outputs().size(), output_count);
      for (size_t i = 0; i < output_count; ++i) {
        it->output(i)->replaceAllUsesWith(replacement_node->output(i));
      }
      it.destroyCurrent();
    }
  }
}

void ReplaceInPlaceOps(const std::shared_ptr<Graph>& graph) {
  ReplaceInPlaceOps(graph->block());
}

}  // namespace jit
}  // namespace torch
