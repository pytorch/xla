#include "eval_static_size.h"

namespace torch {
namespace jit {

namespace {

// Evaluates aten::size on a statically known input.
int64_t RunSizeQuery(Node* node) {
  const auto tensor_type = node->input(0)->type()->cast<CompleteTensorType>();
  JIT_ASSERT(tensor_type);
  const auto tensor_sizes = tensor_type->sizes();
  const auto dim = node->get<int64_t>(attr::dim).value();
  JIT_ASSERT(dim >= 0);
  JIT_ASSERT(static_cast<size_t>(dim) < tensor_sizes.size());
  return tensor_sizes[dim];
}

// Returns true if the size can be evaluated during trace optimization.
bool IsStaticSizeQuery(Node* node) {
  return node->kind() == aten::size &&
         node->inputs().size() == 2 &&
         node->input(0)->type()->cast<CompleteTensorType>() &&
         node->get<int64_t>(attr::dim) &&
         node->get<int64_t>(attr::dim).value() >= 0;
}

}  // namespace

void EvalStaticSize(const std::shared_ptr<Graph>& graph) {
  auto nodes = graph->block()->nodes();
  for (auto node : nodes) {
    if (IsStaticSizeQuery(node)) {
      WithInsertPoint insert_point_guard(node);
      auto new_output = graph->insertConstant(RunSizeQuery(node));
      node->outputs()[0]->replaceAllUsesWith(new_output);
    }
  }
}

}  // namespace jit
}  // namespace torch
