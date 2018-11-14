#include "remove_unused_forward_outputs.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"

namespace torch {
namespace jit {

namespace {

// Remove an unused input from the backward graph from both the outputs and
// captured outputs sections of its input.
void RemoveInputFromBackwardGraph(Gradient& gradient,
                                  const at::optional<size_t> output_idx_maybe,
                                  const size_t captured_output_idx) {
  const auto backward_inputs = gradient.df->inputs();
  const Value* grad_output =
      output_idx_maybe ? backward_inputs[*output_idx_maybe] : nullptr;
  const Value* captured_output = backward_inputs[captured_output_idx];
  // Remove grad_output and captured_output from the inputs of the backward
  // graph.
  for (auto it = gradient.df->nodes().begin(), end = gradient.df->nodes().end();
       it != end; ++it) {
    const auto node_inputs = it->inputs();
    if (grad_output) {
      const auto grad_output_it =
          std::find(node_inputs.begin(), node_inputs.end(), grad_output);
      // Assert that grad_output doesn't have remaining uses.
      JIT_ASSERT(grad_output_it == node_inputs.end());
    }
    const auto captured_output_it =
        std::find(node_inputs.begin(), node_inputs.end(), captured_output);
    if (captured_output_it != node_inputs.end()) {
      WithInsertPoint guard(*it);
      Node* undef = gradient.df->insertNode(gradient.df->createUndefined());
      it->replaceInput(captured_output_it - node_inputs.begin(),
                       undef->output());
    }
  }
  // captured_output_idx points inside the captured outputs section, output_idx
  // points inside the outputs section. We thus have captured_output_idx >
  // output_idx because outputs come before captured outputs. Remove the
  // captured_output_idx first to avoid invalidation of indices.
  JIT_ASSERT(!output_idx_maybe || captured_output_idx > *output_idx_maybe);
  gradient.df->eraseInput(captured_output_idx);
  if (output_idx_maybe) {
    gradient.df->eraseInput(*output_idx_maybe);
  }
}

namespace {

// Counts the number of additional outputs after the "real" outputs used by
// differentiation the same way autodiff does.
size_t IntermediateRequireGradOutputsCount(const Gradient& gradient) {
  const auto forward_outputs = gradient.f->outputs();
  return std::count_if(forward_outputs.begin() + gradient.f_real_outputs,
                       forward_outputs.end(),
                       [](const Value* intermediate_output) {
                         return intermediate_output->requires_grad();
                       });
}

}  // namespace

// Remove the unused output specified by node_output_idx from the given node,
// with subsequent removal from the backward graph input as well.
void RemoveNodeOutputFromGradient(Node* node, const size_t node_output_idx,
                                  Gradient& gradient) {
  const Value* output = node->outputs()[node_output_idx];
  // Find index of this output in forward graph outputs.
  const auto forward_outputs = gradient.f->outputs();
  const auto output_it =
      std::find(forward_outputs.begin(), forward_outputs.end(), output);
  // This output isn't returned from the forward graph, nothing to do.
  if (output_it == forward_outputs.end()) {
    return;
  }
  const size_t output_idx = output_it - forward_outputs.begin();
  const auto output_idx_maybe = (*output_it)->requires_grad()
                                    ? at::optional<size_t>(output_idx)
                                    : at::nullopt;

  // Find the captured_output_idx absolute index of the backward graph input to
  // remove. First, position it at the beginning of the captured outputs, right
  // after the outputs of the forward graph and the captureed inputs.
  size_t captured_output_idx = gradient.f_real_outputs +
                               IntermediateRequireGradOutputsCount(gradient) +
                               gradient.df_input_captured_inputs.size();

  // Remove the given output from the graph outputs.
  gradient.f->eraseOutput(output_idx);
  // Remove the given output from the node outputs.
  node->eraseOutput(node_output_idx);

  // Next, find the index and value in df_input_captured_outputs of the node to
  // remove. Use it to adjust captured_output_idx and update
  // df_input_captured_outputs.
  int df_input_captured_outputs_idx = -1;
  for (size_t i = 0; i < gradient.df_input_captured_outputs.size(); i++) {
    if (static_cast<size_t>(output_idx) ==
        gradient.df_input_captured_outputs[i]) {
      captured_output_idx += i;
      df_input_captured_outputs_idx = i;
      break;
    }
  }
  JIT_ASSERT(df_input_captured_outputs_idx != -1);
  const size_t df_input_captured_outputs_val =
      gradient.df_input_captured_outputs[df_input_captured_outputs_idx];
  // Remove the node from df_input_captured_outputs and adjust references to
  // nodes with higher indices in df_input_captured_outputs.
  gradient.df_input_captured_outputs.erase(
      gradient.df_input_captured_outputs.begin() +
      df_input_captured_outputs_idx);
  for (size_t i = 0; i < gradient.df_input_captured_outputs.size(); i++) {
    if (gradient.df_input_captured_outputs[i] > df_input_captured_outputs_val) {
      --gradient.df_input_captured_outputs[i];
    }
  }

  // Finally, remove the node from all inputs of the backward graph.
  RemoveInputFromBackwardGraph(
      /*gradient=*/gradient,
      /*output_idx_maybe=*/output_idx_maybe,
      /*captured_output_idx=*/captured_output_idx);
}

}  // namespace

void RemoveUnusedForwardOutputs(Gradient& gradient) {
  for (auto it = gradient.f->nodes().begin(), end = gradient.f->nodes().end();
       it != end; ++it) {
    JIT_ASSERT(it->blocks().size() == 0);
    switch (it->kind()) {
      case aten::thnn_conv2d_forward: {
        JIT_ASSERT(it->outputs().size() == 3);
        RemoveNodeOutputFromGradient(*it, 2, gradient);
        RemoveNodeOutputFromGradient(*it, 1, gradient);
        break;
      }
      case aten::max_pool2d_with_indices: {
        JIT_ASSERT(it->outputs().size() == 2);
        RemoveNodeOutputFromGradient(*it, 1, gradient);
        break;
      }
      default:
        break;
    }
  }
}

}  // namespace jit
}  // namespace torch
