#pragma once

#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace torch_xla {
namespace ir {

class Node;
class LoweringContext;

using NodePtr = std::shared_ptr<Node>;

using XlaOpVector = tensorflow::gtl::InlinedVector<xla::XlaOp, 1>;

// Represents a use of the output of a given node.
// If use U is within node N, it means that node U.node is using the output
// U.index of the node N.
struct Use {
  Use() = default;
  Use(Node* node, size_t operand_index, size_t index)
      : node(node), operand_index(operand_index), index(index) {}

  bool operator<(const Use& rhs) const;

  std::string ToString() const;

  // The node using the output of the node this use belongs to.
  Node* node = nullptr;
  // The operand index, within node's operands, which this use refers to.
  size_t operand_index = 0;
  // The index within output the user node refers to.
  size_t index = 0;
};

inline std::ostream& operator<<(std::ostream& stream, const Use& use) {
  stream << use.ToString();
  return stream;
}

// Represents a specific output produced by a node. Since the output of a node
// can be composed by multiple outputs, the node+index coordinates fully qualify
// each single output.
struct Output {
  struct Hasher {
    size_t operator()(const Output& output) const {
      size_t h = reinterpret_cast<std::ptrdiff_t>(output.node);
      return h ^ (h >> 11) ^ output.index;
    }
  };

  Output() = default;
  explicit Output(Node* node, size_t index = 0) : node(node), index(index) {}

  bool operator==(const Output& rhs) const {
    return node == rhs.node && index == rhs.index;
  }
  bool operator!=(const Output& rhs) const { return !operator==(rhs); }

  std::string ToString() const;

  // The node providing the output.
  Node* node = nullptr;
  // The index in the node's output this output refers to.
  size_t index = 0;
};

inline std::ostream& operator<<(std::ostream& stream, const Output& output) {
  stream << output.ToString();
  return stream;
}

using OutputSet = std::unordered_set<Output, Output::Hasher>;

template <typename T>
using OutputMap = std::unordered_map<Output, T, Output::Hasher>;

// Represents an input/operand for a Node object.
struct NodeOperand {
  NodeOperand() = default;
  explicit NodeOperand(NodePtr node, size_t index = 0)
      : node(std::move(node)), index(index) {}

  NodePtr node;
  size_t index = 0;
};

// A node in the graph. Nodes for operations which requires extra data to be
// stored for lowering, should inherit from this class and add operation
// specific member there. For example, a constant might create a new
// NodeConstant class (inheriting from Node) with an extra xla::Literal field,
// or a tensor value might create a new NodeTensor with computation client data
// handle in it.
class Node {
 public:
  // Creates a new node with the given op name. The op name is a unique
  // identifier for the operation, which in the PyTorch case will be the full
  // operation signature which is currently used throughout the code base.
  // The num_outputs tells how many outputs a given operation generates.
  Node(std::string op, tensorflow::gtl::ArraySlice<const NodeOperand> operands,
       size_t num_outputs = 1);

  virtual ~Node();

  const std::string& op() const { return op_; }

  size_t num_outputs() const { return num_outputs_; }

  const std::vector<Output>& operands() const { return operands_as_outputs_; }

  const std::set<Use>& uses() const { return uses_; }

  void ReplaceOperand(size_t operand_no, NodePtr node, size_t index = 0);

  void ReplaceAllUsesWith(NodePtr node, size_t index = 0);

  virtual std::string ToString() const;

  virtual XlaOpVector Lower(LoweringContext* loctx) const;

 private:
  // Adds node's index output number as operand.
  void AddOperand(NodePtr node, size_t index = 0);

  void AddUse(Use use) { uses_.insert(std::move(use)); }

  void RemoveUse(const Use& use) { uses_.erase(use); }

  // The name/ID of the operation captured by this node.
  const std::string op_;
  const size_t num_outputs_ = 1;
  // A node holds a real reference to its operands.
  std::vector<NodePtr> operands_;
  // Outputs do not hold references on the nodes, and neither do the uses, since
  // otherwise we get into circular reference counting.
  std::vector<Output> operands_as_outputs_;
  // We use a set for uses, as we want deterministic use sequencing.
  std::set<Use> uses_;
};

inline std::ostream& operator<<(std::ostream& stream, const Node& node) {
  stream << node.ToString();
  return stream;
}

}  // namespace ir
}  // namespace torch_xla
