#include "lowering_context.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch_xla {
namespace ir {

xla::XlaOp LoweringContext::GetParameter(
    const std::shared_ptr<xla::ComputationClient::Data>& data) {
  auto it = parameters_map_.find(data.get());
  if (it == parameters_map_.end()) {
    xla::XlaOp param =
        xla::Parameter(builder(), parameters_.size(), data->shape(),
                       absl::StrCat("param_", parameters_.size()));
    parameters_.push_back(data);
    it = parameters_map_.emplace(data.get(), param).first;
  }
  return it->second;
}

std::vector<xla::ComputationClient::Data*> LoweringContext::GetParametersData()
    const {
  std::vector<xla::ComputationClient::Data*> parameters;
  for (auto& param : parameters_) {
    parameters.push_back(param.get());
  }
  return parameters;
}

xla::int64 LoweringContext::AddResult(xla::XlaOp op) {
  root_tuple_.push_back(std::move(op));
  return root_tuple_.size() - 1;
}

xla::StatusOr<xla::XlaComputation> LoweringContext::Build() {
  if (!root_tuple_.empty()) {
    auto root = xla::Tuple(builder(), root_tuple_);
    return builder()->Build(root);
  }
  return builder()->Build();
}

xla::StatusOr<xla::XlaComputation> LoweringContext::Build(
    const xla::XlaOp& root) {
  XLA_CHECK(root_tuple_.empty());
  return builder()->Build(root);
}

void LoweringContext::AssignOutputOp(const Output& output, xla::XlaOp op) {
  emitted_outputs_[output] = op;
}

xla::XlaOp LoweringContext::GetOutputOp(const Output& output) {
  auto it = emitted_outputs_.find(output);
  if (it == emitted_outputs_.end()) {
    for (auto node : GetEmissionPostOrder(output.node)) {
      node->Lower(this);
    }
    // At this point the outpout better be present, otherwise there is an issue
    // with the lowering code.
    it = emitted_outputs_.find(output);
    XLA_CHECK(it != emitted_outputs_.end())
        << "No XLA operation emitted for output: " << output;
  }
  return it->second;
}

std::vector<Node*> LoweringContext::GetEmissionPostOrder(Node* node) {
  std::vector<Node*> post_order;
  std::vector<Node*> queue;
  queue.push_back(node);
  while (!queue.empty()) {
    node = queue.back();
    auto it = emit_status_.find(node);
    if (it == emit_status_.end()) {
      emit_status_[node] = kEmitting;

      for (auto& output : node->operands()) {
        auto oit = emit_status_.find(output.node);
        if (oit == emit_status_.end()) {
          queue.push_back(output.node);
        } else if (oit->second == kEmitting) {
          XLA_ERROR() << "Graph loop found at " << *output.node;
        }
      }
    } else if (it->second == kEmitting) {
      for (auto& output : node->operands()) {
        auto oit = emit_status_.find(output.node);
        XLA_CHECK(oit != emit_status_.end() && oit->second == kEmitted)
            << "Graph loop found at " << *output.node;
      }
      emit_status_[node] = kEmitted;
      post_order.push_back(node);
      queue.pop_back();
    } else {
      XLA_CHECK_EQ(it->second, kEmitted);
      queue.pop_back();
    }
  }
  return post_order;
}

}  // namespace ir
}  // namespace torch_xla
