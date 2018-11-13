#include "replace_untraced_operators.h"

namespace torch {
namespace jit {

namespace {

// Returns true if the node contains an attribute and has the expected value.
template <class T>
bool NodeHasExpectedAttribute(const Node* node, const Symbol attribute_name,
                              const T& expected) {
  const auto maybe_attribute = node->get<T>(attribute_name);
  return maybe_attribute && *maybe_attribute == expected;
}

// Only allow certain aten::_convolution operators to be replaced.
bool CanTraceConvolution(const Node* node) {
  return NodeHasExpectedAttribute(node, attr::dilation,
                                  std::vector<int64_t>{1, 1}) &&
         NodeHasExpectedAttribute(node, attr::output_padding,
                                  std::vector<int64_t>{0, 0}) &&
         NodeHasExpectedAttribute(node, attr::transposed, false) &&
         NodeHasExpectedAttribute(node, attr::groups, int64_t(1)) &&
         NodeHasExpectedAttribute(node, attr::benchmark, false) &&
         NodeHasExpectedAttribute(node, attr::deterministic, false);
}

// When possible, replace aten::{_convolution, batch_norm} operators with
// equivalent ones which are part of the operator schema and differentiable.
void ReplaceUntracedOperators(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub : it->blocks()) {
      ReplaceUntracedOperators(sub);
    }
    switch (it->kind()) {
      case aten::_convolution: {
        WithInsertPoint guard(*it);
        auto graph = block->owningGraph();
        auto node = *it;
        if (!CanTraceConvolution(node)) {
          break;
        }
        const auto weight = node->namedInput(attr::weight);
        const auto weight_type = weight->type()->expect<CompleteTensorType>();
        const auto& weight_size = weight_type->sizes();
        const auto kernel_size = graph->insertConstant(
            std::vector<int64_t>{weight_size[2], weight_size[3]});
        const auto stride = graph->insertConstant(
            node->get<std::vector<int64_t>>(attr::stride).value());
        const auto padding = graph->insertConstant(
            node->get<std::vector<int64_t>>(attr::padding).value());

        auto replacement_node = graph->create(aten::thnn_conv2d_forward, 3);

        graph->insertNode(replacement_node);
        replacement_node->addInput(node->namedInput(attr::input));
        replacement_node->addInput(weight);
        replacement_node->addInput(kernel_size);
        replacement_node->addInput(node->namedInput(attr::bias));
        replacement_node->addInput(stride);
        replacement_node->addInput(padding);

        replacement_node->outputs()[0]->setType(it->outputs()[0]->type());
        it->output()->replaceAllUsesWith(replacement_node->outputs()[0]);
        it.destroyCurrent();
        break;
      }
      case aten::batch_norm: {
        WithInsertPoint guard(*it);
        auto graph = block->owningGraph();
        auto node = *it;
        auto replacement_node = graph->create(aten::native_batch_norm, 3);

        graph->insertNode(replacement_node);
        const auto node_inputs = node->inputs();
        JIT_ASSERT(node_inputs.size() == 9);
        for (size_t i = 0; i < node_inputs.size() - 1; ++i) {
          replacement_node->addInput(node_inputs[i]);
        }
        replacement_node->outputs()[0]->setType(it->outputs()[0]->type());
        it->output()->replaceAllUsesWith(replacement_node->outputs()[0]);
        it.destroyCurrent();
        break;
      }
      default: { break; }
    }
  }
}

}  // namespace

void ReplaceUntracedOperators(const std::shared_ptr<Graph>& graph) {
  ReplaceUntracedOperators(graph->block());
}

}  // namespace jit
}  // namespace torch
