#include "ir.h"

#include <sstream>

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch_xla {
namespace ir {

bool Use::operator<(const Use& rhs) const {
  int cmp = node->op().compare(rhs.node->op());
  if (cmp != 0) {
    return cmp < 0;
  }
  if (operand_index != rhs.operand_index) {
    return operand_index < rhs.operand_index;
  }
  return index < rhs.index;
}

std::string Use::ToString() const {
  std::stringstream ss;
  ss << node->ToString() << ";o=" << operand_index << ";i=" << index;
  return ss.str();
}

std::string Output::ToString() const {
  std::stringstream ss;
  ss << node->ToString() << ";i=" << index;
  return ss.str();
}

Node::Node(std::string op,
           tensorflow::gtl::ArraySlice<const NodeOperand> operands,
           size_t num_outputs)
    : op_(std::move(op)), num_outputs_(num_outputs) {
  for (auto& operand : operands) {
    AddOperand(operand.node, operand.index);
  }
}

Node::~Node() {
  for (size_t i = 0; i < operands_as_outputs_.size(); ++i) {
    operands_as_outputs_[i].node->RemoveUse(
        Use(this, i, operands_as_outputs_[i].index));
  }
}

void Node::AddOperand(NodePtr node, size_t index) {
  XLA_CHECK_LT(index, node->num_outputs());
  operands_.push_back(std::move(node));
  operands_as_outputs_.push_back(Output(operands_.back().get(), index));
  operands_.back()->AddUse(Use(this, operands_.size() - 1, index));
}

void Node::ReplaceOperand(size_t operand_no, NodePtr node, size_t index) {
  XLA_CHECK_LT(index, node->num_outputs());
  Output* output = &operands_as_outputs_.at(operand_no);
  output->node->RemoveUse(Use(this, operand_no, output->index));
  node->AddUse(Use(this, operand_no, index));
  *output = Output(node.get(), index);
  operands_[operand_no] = std::move(node);
}

void Node::ReplaceAllUsesWith(NodePtr node, size_t index) {
  // A call to ReplaceOperand() will end up calling RemoveUse() into the
  // current node, so snapshot the current uses and iterate over them.
  std::vector<Use> current_uses(uses_.begin(), uses_.end());
  for (auto& use : current_uses) {
    use.node->ReplaceOperand(use.operand_index, node, index);
  }
}

std::string Node::ToString() const {
  std::stringstream ss;
  ss << op();
  if (num_outputs() > 1) {
    ss << ";n=" << num_outputs();
  }
  return ss.str();
}

XlaOpVector Node::Lower(LoweringContext* loctx) const {
  XLA_ERROR() << "Lowering not implemented for node: " << *this;
}

}  // namespace ir
}  // namespace torch_xla
