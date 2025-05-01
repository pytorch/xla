#ifndef XLA_TORCH_XLA_CSRC_DYNAMIC_SHAPE_DETECTOR_H_
#define XLA_TORCH_XLA_CSRC_DYNAMIC_SHAPE_DETECTOR_H_

#include <torch/csrc/lazy/core/hash.h>

#include <map>

#include "absl/types/span.h"

namespace torch_xla {

struct TrieNode;

// Unit of information stored inside the trie.
struct TrieValue {
  // Unique value that identifies an IR node.
  torch::lazy::hash_t hash;

  // String representation of the IR node.
  std::string str;
};

// Helper struct for iteratively building the trie.
//
// This structure keeps track of what's the current state (i.e. TrieNode) and
// how much of its common_sequence_ have we matched up to this point (i.e.
// matched).
//
// Upon calling AddValue, we might change the current state of this builder by
// increasing the number of matched elements, or jumping to another TrieNode.
//
// You can think of it as if we were jumping around in a DFA, and this struct
// allow us to walk through and modify said DFA.
//
// Main assumption for every TrieBuilder
// =====================================
//
// 1. matched_ will be up to the size of the node's common_sequence_.
// 2. matched_ will be 0 only in the beginning for the root node.
struct TrieBuilder {
  // Wrappers to the currently pointed to TrieNode methods.
  void AddValue(TrieValue value, bool allow_new_graph);
  bool MarkGraphBoundary(bool allow_new_graph);

  // Current TrieNode.
  TrieNode* node_;

  // Number of matched elements in the current node.
  std::size_t matched_;
};

// Implementation of a compressed trie.
//
// Main idea
// =========
//
// The main interface to interact with TrieNode is TrieBuilder. We start from
// the root, incrementally accepting new TrieValue by calling AddValue. Said
// function will incrementally build the trie. Finally, MarkGraphBoundary will
// set is_graph_boundary_ and maybe split the current node (if we haven't
// matched everything in this node's common_sequence_).
//
// Main assumption for every TrieNode
// ==================================
//
// Except for the root, common_sequence_ will always have size, at least, 1.The
// first element always corresponds to the TrieValue that was used to go from
// the parent node to this one.
//
// Examples
// ========
//
// 1. In an empty trie, we started with TrieBuilder {root, 0}, where root
// corresponds to the empty trie's only node. Upon calling AddValue, the given
// value will be appended to common_sequence_. Finally, we return a new
// TrieBuilder {root, 1} (we have one match!).
//
// 2. Consider the TrieBuilder {root, 20}. If AddValue is called, and
// common_sequence_'s size is greater-than 20, we check if the given value is
// the same as common_sequence_[20]. If so, we do nothing but return an updated
// TrieBuilder {root, 21} (we have matched one extra value).
//
// 3. Consider the TrieBuilder {root, 20}. If AddValue is called, and
// common_sequence_'s size is exactly 20, we check if the given value is one of
// this node's children. If not, we create a new TrieNode, and add it to the
// children's map. Finally, we return a new TrieBuilder {newnode, 1} (the 1st
// element of newnode is the value that was responsible for creating it).
//
// 4. Consider the TrieBuilder {root, 20}. If AddValue is called, but the node's
// common_sequence_ has size 35, root will be split. As a result, 2 nodes will
// be created: (i) a node containing the remaining unmatched 15 elements; and
// (ii) a node containing the given TrieValue. The returned TrieBuilder will be
// {node (ii), 1}.
//
// 5. Consider the TrieBuilder {root, 20}. If MarkGraphBoundary is called, and
// root is a leaf (i.e. no children), then root.is_graph_boundary_ is set to
// true.
struct TrieNode {
  using ChildrenMap = std::map<torch::lazy::hash_t, std::unique_ptr<TrieNode>>;

  TrieNode(absl::Span<const TrieValue> common_sequence = {},
           bool is_graph_boundary = false);

  // May add TrieValue to this TrieNode.
  //
  // This function is used to iteratively construct the graph. It does 2 things.
  //
  // First, it checks whether the given value actually matches the values
  // already inside this node, i.e. this graph was seen before. For example, the
  // given value may match the value inside common_sequence_ (after `matched`
  // elements) or one of children (if `matched` equals the size of
  // common_sequence_).
  //
  // Then, if the given value is not inside this node, we have to add it by
  // either:
  //   1. adding it to the common_sequence_
  //   2. adding it to the children_
  //   3. splitting this node, creating 2 new nodes containing: (i) rest of the
  //      unmatched common_sequence_; and (ii) the given value.
  TrieBuilder AddValue(TrieValue value, std::size_t matched,
                       bool allow_new_graph);

  // Marks this node as graph boundary.
  //
  // Given the number of `matched` elements in the common_sequence_, this
  // function sets `is_graph_boundary_` and possibly moves the rest of the
  // unmatched common_sequence_ to a new node.
  //
  // Returns whether a new graph was created.
  bool MarkGraphBoundary(std::size_t matched, bool allow_new_graph);

  // Issue an error indicating a new graph is not allowed.
  //
  // This function will correctly inspect the TrieNode, building an informative
  // error message.
  void NewGraphNotAllowedError(std::optional<TrieValue> value,
                               std::size_t matched);

  // Maybe split this node into 2, containing, respectively: (i)
  // common_sequence_ before `matched`; and (ii) common_sequence_ after
  // `matched`.
  //
  // If `matched` is 0, it means that we have a divergence from the start. The
  // created suffix node will be a copy of this node, while this node will have
  // a 0-sized common_sequence_ with populated children.
  //
  // If `matched` is the size of common_sequence_, it means that we don't need a
  // suffix node (since there's no suffix).
  //
  // Return whether we did split it or not.
  bool MaybeSplitAt(std::size_t matched);

  // Returns true if this node is a leaf, i.e. no children.
  bool IsLeaf() const;

  // Sequence of values all children_ in this node share.
  std::vector<TrieValue> common_sequence_;

  // Flag indicating whether the current node is a graph boundary. i.e.
  // whether there is a graph that ends with common_sequence_.
  bool is_graph_boundary_;

  // Children, i.e. forking points, of this node.
  ChildrenMap children_;
};

struct SessionInfo {
  // Instantiates a new TrieBuilder located at its root.
  TrieBuilder NewBuilder();

  // Name of this session.
  std::string name_;

  // Root of the trie that stores graph information for this session.
  std::unique_ptr<TrieNode> root_;

  // Number of recorded graphs for this session.
  std::size_t graphs_;
};

// Surface class for detecting dynamic shapes.
//
// Manages the information related to each session as well as the active
// session, i.e. the one that we are recording graphs for.
class DynamicShapeDetector {
 public:
  static DynamicShapeDetector* Get();

  // Starts recording the created IR nodes into the trie whose root is
  // associated with the session named: `name`.
  void StartSession(const std::string& name);

  // Stops recording the created IR nodes for the active session.
  //
  // Before doing that, we commit the current graph, turning the current
  // TrieNode being visited into a graph boundary.
  //
  // This function may raise an exception if we aren't allowed to create
  // more graphs.
  void EndSession();

  // Records a newly created IR node (its metadata).
  //
  // This function may raise an exception if:
  //   1. we aren't allowed to create more graphs; and
  //   2. we have to create a new TrieNode because this IR node wasn't expected
  //      in the trie.
  void AddNodeInfo(torch::lazy::hash_t hash, const std::string& str);

  // Checks whether there's any session active.
  bool IsSessionActive();

  // Maybe removes the session entry.
  void RemoveSessionIfExists(const std::string& name);

  // API for setting the maximum number of graphs allowed to be recorded.
  static void SetMaxDifferentGraphs(std::size_t value);
  static std::size_t GetMaxDifferentGraphs();

 private:
  // Whether the current session allows new graphs, i.e. new graph compilations.
  bool AllowNewGraph();

  // Move the TrieBuilder to the root node of this session.
  void RootBuilder();

  // Resets the data related to the current session.
  //
  // Specifically, this function:
  //   1. resets the builder
  //   2. assigns current_session_ to nullptr
  void ResetSession();

  // Stores the information related to each session.
  std::unordered_map<std::string, SessionInfo> session_infos_;

  // Pointer to the current active session.
  SessionInfo* current_session_;

  // Iterative builder for the current active session.
  TrieBuilder builder_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_DYNAMIC_SHAPE_DETECTOR_H_
