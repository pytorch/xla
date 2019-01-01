#include "eval_static_size.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch {
namespace jit {

namespace {

// Evaluates aten::size on a statically known input.
int64_t RunSizeQuery(Node* node) {
  const auto tensor_type = node->input(0)->type()->cast<CompleteTensorType>();
  XLA_CHECK(tensor_type != nullptr);
  const auto tensor_sizes = tensor_type->sizes();
  const auto dim = node->get<int64_t>(attr::dim).value();
  XLA_CHECK_GE(dim, 0);
  XLA_CHECK_LT(static_cast<size_t>(dim), tensor_sizes.size());
  return tensor_sizes[dim];
}

// Returns true if the size can be evaluated during trace optimization.
bool IsStaticSizeQuery(Node* node) {
  return node->kind() == aten::size && node->inputs().size() == 2 &&
         node->input(0)->type()->cast<CompleteTensorType>() &&
         node->get<int64_t>(attr::dim) &&
         node->get<int64_t>(attr::dim).value() >= 0;
}

}  // namespace

void EvalStaticSize(const std::shared_ptr<Graph>& graph) {
  XLA_VLOG_LINES(4, "Before EvalStaticSize:\n" + graph->toString());
  auto nodes = graph->block()->nodes();
  for (auto node : nodes) {
    if (IsStaticSizeQuery(node)) {
      WithInsertPoint insert_point_guard(node);
      TF_VLOG(3) << "Evaluated " << *node << " to a constant size";
      auto new_output = graph->insertConstant(RunSizeQuery(node));
      node->outputs()[0]->replaceAllUsesWith(new_output);
    }
  }
  XLA_VLOG_LINES(4, "After EvalStaticSize:\n" + graph->toString());
}

}  // namespace jit
}  // namespace torch
