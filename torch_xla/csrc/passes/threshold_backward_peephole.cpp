#include "threshold_backward_peephole.h"

namespace torch {
namespace jit {

namespace {

void ThresholdBackwardPeephole(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub_block : it->blocks()) {
      ThresholdBackwardPeephole(sub_block);
    }
    if (it->kind() == aten::mul) {
      const auto type_as_cand = it->input(1)->node();
      if (type_as_cand->kind() == aten::type_as) {
        const auto gt_cand = type_as_cand->input(0)->node();
        if (gt_cand->kind() == aten::gt) {
          WithInsertPoint guard(*it);
          auto graph = block->owningGraph();
          auto replacement_node = graph->create(aten::threshold_backward);
          graph->insertNode(replacement_node);
          replacement_node->addInput(it->input(0));
          replacement_node->addInput(gt_cand->input(0));
          replacement_node->addInput(gt_cand->input(1));
          it->output()->replaceAllUsesWith(replacement_node->outputs()[0]);
          it.destroyCurrent();
        }
      }
    }
  }
}

}  // namespace

void ThresholdBackwardPeephole(const std::shared_ptr<Graph>& graph) {
  ThresholdBackwardPeephole(graph->block());
}

}  // namespace jit
}  // namespace torch
