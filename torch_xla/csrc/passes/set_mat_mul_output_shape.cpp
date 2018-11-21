#include "set_mat_mul_output_shape.h"

namespace torch {
namespace jit {

namespace {

void SetMatMulOutputShape(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub_block : it->blocks()) {
      SetMatMulOutputShape(sub_block);
    }
    if (it->kind() == aten::mm) {
      const auto lhs_type = it->input(0)->type()->cast<CompleteTensorType>();
      const auto rhs_type = it->input(1)->type()->cast<CompleteTensorType>();
      JIT_ASSERT(lhs_type->sizes().size() == 2 &&
                 rhs_type->sizes().size() == 2);
      it->output()->setType(CompleteTensorType::create(
          lhs_type->scalarType(), lhs_type->device(),
          at::IntList{lhs_type->sizes().at(0), rhs_type->sizes().at(1)}));
    }
  }
}

}  // namespace

void SetMatMulOutputShape(const std::shared_ptr<Graph>& graph) {
  SetMatMulOutputShape(graph->block());
}

}  // namespace jit
}  // namespace torch
