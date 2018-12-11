#include "remove_in_place_out_param_ops.h"

namespace torch {
namespace jit {

namespace {

void RemoveInPlaceOutParamOps(Block* block,
                              const at::ArrayRef<Value*> graph_inputs) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub_block : it->blocks()) {
      RemoveInPlaceOutParamOps(sub_block, graph_inputs);
    }
    if (it->kind() == aten::add_ && !it->hasUses()) {
      const auto graph_inputs_it =
          std::find(graph_inputs.begin(), graph_inputs.end(), it->input(0));
      if (graph_inputs_it != graph_inputs.end()) {
        it.destroyCurrent();
      }
    }
  }
}

}  // namespace

void RemoveInPlaceOutParamOps(const std::shared_ptr<Graph>& graph) {
  const auto graph_inputs = graph->inputs();
  RemoveInPlaceOutParamOps(graph->block(), graph_inputs);
}

}  // namespace jit
}  // namespace torch
