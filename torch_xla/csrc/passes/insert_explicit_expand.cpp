#include "insert_explicit_expand.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch {
namespace jit {

namespace {

void InsertExplicitExpand(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub_block : it->blocks()) {
      InsertExplicitExpand(sub_block);
    }
    if (it->kind() == aten::add || it->kind() == aten::mul) {
      const auto lhs_type = it->input(0)->type()->cast<CompleteTensorType>();
      const auto rhs_type = it->input(1)->type()->cast<CompleteTensorType>();
      const auto output_type =
          it->output(0)->type()->cast<CompleteTensorType>();
      if (lhs_type && rhs_type && output_type &&
          lhs_type->sizes() != rhs_type->sizes()) {
        WithInsertPoint insert_point_guard(*it);
        auto graph = block->owningGraph();
        const auto tensor_sizes = graph->insertConstant(rhs_type->sizes());
        const auto expand =
            graph->insert(aten::expand, {it->input(0), tensor_sizes});
        TF_VLOG(3) << "Explicitly expand the first operand of " << **it;
        it->replaceInput(0, expand);
        it->output(0)->setType(output_type->withRequiresGrad(true));
      }
    }
  }
}

}  // namespace

void InsertExplicitExpand(const std::shared_ptr<Graph>& graph) {
  XLA_VLOG_LINES(4, "Before InsertExplicitExpand:\n" + graph->toString());
  InsertExplicitExpand(graph->block());
  XLA_VLOG_LINES(4, "After InsertExplicitExpand:\n" + graph->toString());
}

}  // namespace jit
}  // namespace torch
