#include "remove_unused_forward_outputs.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"

namespace torch {
namespace jit {

namespace {

// Remove an unused input from the backward graph from both the outputs and
// captured outputs sections of its input.
void RemoveInputFromBackwardGraph(Gradient* gradient, size_t output_idx,
                                  size_t captured_output_idx) {
  const auto backward_inputs = gradient->df->inputs();
  XLA_CHECK_LT(output_idx, backward_inputs.size());
  const Value* grad_output = backward_inputs[output_idx];
  const Value* captured_output = backward_inputs[captured_output_idx];
  // Remove grad_output and captured_output from the inputs of the backward
  // graph.
  for (auto it = gradient->df->nodes().begin(),
            end = gradient->df->nodes().end();
       it != end; ++it) {
    const auto node_inputs = it->inputs();
    if (grad_output) {
      const auto grad_output_it =
          std::find(node_inputs.begin(), node_inputs.end(), grad_output);
      // Assert that grad_output doesn't have remaining uses.
      XLA_CHECK(grad_output_it == node_inputs.end())
          << "Gradient output has remaining uses";
    }
    const auto captured_output_it =
        std::find(node_inputs.begin(), node_inputs.end(), captured_output);
    if (captured_output_it != node_inputs.end()) {
      WithInsertPoint guard(*it);
      Node* undef = gradient->df->insertNode(gradient->df->createUndefined());
      TF_VLOG(3) << "Replacing input at index "
                 << captured_output_it - node_inputs.begin() << " of " << **it
                 << " with undefined";
      it->replaceInput(captured_output_it - node_inputs.begin(),
                       undef->output());
    }
  }
  // captured_output_idx points inside the captured outputs section, output_idx
  // points inside the outputs section. We thus have captured_output_idx >
  // output_idx because outputs come before captured outputs. Remove the
  // captured_output_idx first to avoid invalidation of indices.
  XLA_CHECK_GT(captured_output_idx, output_idx)
      << "Captured output must follow output on removal";
  TF_VLOG(3) << "Removing inputs at indices " << captured_output_idx << " and "
             << output_idx << " from the backward graph";
  gradient->df->eraseInput(captured_output_idx);
  gradient->df->eraseInput(output_idx);
}

// Remove the unused output specified by node_output_idx from the given node,
// with subsequent removal from the backward graph input as well.
void RemoveNodeOutputFromGradient(Node* node, size_t node_output_idx,
                                  Gradient* gradient) {
  const Value* output = node->outputs()[node_output_idx];
  // Find index of this output in forward graph outputs.
  const auto forward_outputs = gradient->f->outputs();
  const auto output_it =
      std::find(forward_outputs.begin(), forward_outputs.end(), output);
  // This output isn't returned from the forward graph, nothing to do.
  if (output_it == forward_outputs.end()) {
    return;
  }
  const size_t forward_output_idx = output_it - forward_outputs.begin();

  // Find the captured_output_idx absolute index of the backward graph input to
  // remove. First, position it at the beginning of the captured outputs, right
  // after the outputs of the forward graph and the captureed inputs.
  size_t captured_output_idx = gradient->df_input_vjps.size() +
                               gradient->df_input_captured_inputs.size();

  // Remove the given output from the graph outputs.
  TF_VLOG(3) << "Removing output at index " << forward_output_idx
             << " from the forward graph";
  gradient->f->eraseOutput(forward_output_idx);
  // Remove the given output from the node outputs.
  TF_VLOG(3) << "Removing output at index " << node_output_idx << " from "
             << *node;
  node->eraseOutput(node_output_idx);

  // Next, find the index and value in df_input_captured_outputs of the node to
  // remove. Use it to adjust captured_output_idx and update
  // df_input_captured_outputs.
  int df_input_captured_outputs_idx = -1;
  for (size_t i = 0; i < gradient->df_input_captured_outputs.size(); i++) {
    if (static_cast<size_t>(forward_output_idx) ==
        gradient->df_input_captured_outputs[i]) {
      captured_output_idx += i;
      df_input_captured_outputs_idx = i;
      break;
    }
  }
  XLA_CHECK_NE(df_input_captured_outputs_idx, -1)
      << "Outputs to remove not found among backward graph inputs";
  const size_t df_input_captured_outputs_val =
      gradient->df_input_captured_outputs[df_input_captured_outputs_idx];
  // Remove the node from df_input_captured_outputs and adjust references to
  // nodes with higher indices in df_input_captured_outputs.
  gradient->df_input_captured_outputs.erase(
      gradient->df_input_captured_outputs.begin() +
      df_input_captured_outputs_idx);
  for (size_t i = 0; i < gradient->df_input_captured_outputs.size(); i++) {
    if (gradient->df_input_captured_outputs[i] >
        df_input_captured_outputs_val) {
      --gradient->df_input_captured_outputs[i];
    }
  }

  // df_input_vjps must be adjusted similarly to df_input_captured_outputs since
  // it also contains output indices which can shift when outputs are removed.
  const auto df_input_vjps_it =
      std::find(gradient->df_input_vjps.begin(), gradient->df_input_vjps.end(),
                forward_output_idx);
  XLA_CHECK(df_input_vjps_it != gradient->df_input_vjps.end())
      << "Forward output index not found in df_input_vjps";
  const size_t output_idx = df_input_vjps_it - gradient->df_input_vjps.begin();
  const auto df_input_vjps_val = *df_input_vjps_it;
  gradient->df_input_vjps.erase(df_input_vjps_it);
  for (auto& df_input_vjps_elem : gradient->df_input_vjps) {
    if (df_input_vjps_elem > df_input_vjps_val) {
      --df_input_vjps_elem;
    }
  }

  // Finally, remove the node from all inputs of the backward graph.
  RemoveInputFromBackwardGraph(
      /*gradient=*/gradient,
      /*output_idx=*/output_idx,
      /*captured_output_idx=*/captured_output_idx);
}

}  // namespace

void RemoveUnusedForwardOutputs(Gradient* gradient) {
  XLA_VLOG_LINES(4, "Before RemoveUnusedForwardOutputs:\n");
  XLA_VLOG_LINES(4, "Forward:\n" + gradient->f->toString() + "\n");
  XLA_VLOG_LINES(4, "Backward:\n" + gradient->df->toString() + "\n");
  for (auto it = gradient->f->nodes().begin(), end = gradient->f->nodes().end();
       it != end; ++it) {
    XLA_CHECK_EQ(it->blocks().size(), 0) << "Graph not flattened";
    switch (it->kind()) {
      case aten::thnn_conv2d_forward: {
        XLA_CHECK_EQ(it->outputs().size(), 3)
            << "Invalid number of outputs for thnn_conv2d_forward";
        RemoveNodeOutputFromGradient(*it, 2, gradient);
        RemoveNodeOutputFromGradient(*it, 1, gradient);
        break;
      }
      case aten::max_pool2d_with_indices: {
        XLA_CHECK_EQ(it->outputs().size(), 2)
            << "Invalid number of outputs for max_pool2d_with_indices";
        RemoveNodeOutputFromGradient(*it, 1, gradient);
        break;
      }
      default:
        break;
    }
  }
  XLA_VLOG_LINES(4, "After RemoveUnusedForwardOutputs:\n");
  XLA_VLOG_LINES(4, "Forward:\n" + gradient->f->toString() + "\n");
  XLA_VLOG_LINES(4, "Backward:\n" + gradient->df->toString() + "\n");
}

}  // namespace jit
}  // namespace torch
