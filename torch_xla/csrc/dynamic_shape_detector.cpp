#include "torch_xla/csrc/dynamic_shape_detector.h"

#include <sstream>

#include "torch_xla/csrc/runtime/debug_macros.h"

namespace torch_xla {

// Maximum number of allowed traces per function (i.e. session).
static std::size_t max_allowed_traces_per_function = 1;

TrieNode::TrieNode(absl::Span<const TrieValue> common_sequence,
                   bool is_trace_boundary)
    : common_sequence_(common_sequence.begin(), common_sequence.end()),
      is_trace_boundary_(is_trace_boundary) {}

bool TrieNode::IsLeaf() const { return children_.empty(); }

void TrieNode::NewTraceNotAllowedError(std::optional<TrieValue> value,
                                       std::size_t matched) {
  std::ostringstream ostr;
  ostr << "Maximum number of different traces allowed per function exceeded: "
       << max_allowed_traces_per_function << std::endl;

  if (value.has_value()) {
    ostr << "Got: " << value->str << std::endl;
  } else {
    ostr << "Reached the end of the function at: "
         << common_sequence_[matched - 1].str << std::endl;
  }

  if (common_sequence_.size() > matched) {
    ostr << "Expected: " << common_sequence_[matched].str << std::endl;
  } else {
    ostr << "Expected either of:" << std::endl;
    for (auto& pair : children_) {
      ostr << "  - " << pair.second->common_sequence_.front().str << std::endl;
    }
  }

  XLA_ERROR() << ostr.str();
}

bool TrieNode::MarkTraceBoundary(std::size_t matched, bool allow_new_trace) {
  // No need to do anything here, iff:
  //
  //   1. nothing was matched, yet
  //
  //   2. we matched everything in this node, and this node is already marked as
  //   a trace boundary.
  if (matched == 0 ||
      (common_sequence_.size() == matched && is_trace_boundary_)) {
    return false;
  }

  // From this point, we will create a new trace.
  if (!allow_new_trace) {
    // Raise an error if we have reached the maximum number of traces.
    NewTraceNotAllowedError(std::nullopt, matched);
  }

  // If we haven't matched everything in this node, we will have to split this
  // node. The newly created node will contain the suffix (common_sequence_
  // after matched), and this node (i.e. the existing one) will contain the
  // prefix.
  if (common_sequence_.size() != matched) {
    MaybeSplitAt(matched);
  }

  // Finally, mark this node as a trace boundary.
  is_trace_boundary_ = true;

  return true;
}

TrieBuilder TrieNode::AddValue(TrieValue value, std::size_t matched,
                               bool allow_new_trace) {
  TF_VLOG(5) << "Adding value: " << value.str << " (" << value.hash << ")";

  // If this node has no children and is not marked as a trace boundary, it
  // means that TrieBuilder created this node and is incrementally adding
  // TrieValue to it. Therefore, we just need to keep doing it.
  if (IsLeaf() && !is_trace_boundary_) {
    common_sequence_.push_back(value);
    return {this, matched + 1};
  }

  // If common_sequence_ still has more elements to be matched, try to match
  // it with value. If we succeed, we simply increment the number of matched
  // elements.
  if (common_sequence_.size() > matched &&
      common_sequence_[matched].hash == value.hash) {
    return {this, matched + 1};
  }

  // If we have matched every element in this node, try to find a child that
  // corresponds to the given value. If we find it, return a TrieBuilder with
  // the node found, and set matched_ to 1.
  if (common_sequence_.size() == matched &&
      children_.find(value.hash) != children_.end()) {
    return {children_[value.hash].get(), 1};
  }

  // Otherwise, we will have to create a new trace. So, first, check whether we
  // are allowed to do so.
  if (!allow_new_trace) {
    NewTraceNotAllowedError(value, matched);
  }

  // Maybe split the current node into: prefix (before matched) and suffix
  // (after matched).
  bool did_split = MaybeSplitAt(matched);

  // Create a new node that contains only the given value.
  std::unique_ptr<TrieNode> node =
      std::make_unique<TrieNode>(absl::Span<const TrieValue>{value});

  // Associate the given value with the created node in the children's map.
  children_[value.hash] = std::move(node);

  TF_VLOG(5) << "Created new node " << children_[value.hash].get()
             << " for value: " << value.str << " (" << value.hash << ")";

  // Unmark this node as trace boundary iff we actually split this node (i.e.
  // suffix actually had something). Otherwise, this should still be a trace
  // boundary.
  if (did_split) {
    is_trace_boundary_ = false;
  }

  return {children_[value.hash].get(), 1};
}

bool TrieNode::MaybeSplitAt(std::size_t matched) {
  // Split common_sequence_ into prefix (before matched) and suffix (after
  // matched). Note that these variables are spans, i.e. they don't own their
  // contents.
  absl::Span<const TrieValue> common_sequence(common_sequence_);
  absl::Span<const TrieValue> prefix =
      common_sequence.subspan(0, /*len=*/matched);
  absl::Span<const TrieValue> suffix = common_sequence.subspan(matched);

  // A split only occurs if suffix is not empty.
  if (!suffix.empty()) {
    std::unique_ptr<TrieNode> suffix_node =
        std::make_unique<TrieNode>(suffix, is_trace_boundary_);

    // The suffix node's children should be what this node's children was before
    // the split. Therefore, we swap those.
    std::swap(children_, suffix_node->children_);

    // Create the children_ map entry for the newly created suffix node.
    children_[suffix.front().hash] = std::move(suffix_node);

    TF_VLOG(5) << "Split node " << children_[suffix.front().hash].get()
               << " at position " << matched << ": " << suffix.front().str
               << " (" << suffix.front().hash << ")";
  }

  // This node's common_sequence_ will be whatever the prefix was.
  common_sequence_ = std::vector<TrieValue>{prefix.begin(), prefix.end()};
}

DynamicShapeDetector* DynamicShapeDetector::Get() {
  static DynamicShapeDetector ds_detector = DynamicShapeDetector();
  return &ds_detector;
}

void DynamicShapeDetector::StartSession(std::string session) {
  if (session_infos_.find(session) == session_infos_.end()) {
    // Create a new session, with a fresh TrieNode.
    session_infos_[session] = {session, std::make_unique<TrieNode>(), 0};
    TF_VLOG(5) << "Created new session: " << session;
  }
  current_session_ = &session_infos_[session];
  TF_VLOG(5) << "Started session: " << session;
  RootBuilder();
}

void DynamicShapeDetector::SetMaxAllowedTraces(std::size_t value) {
  max_allowed_traces_per_function = value;
}

std::size_t DynamicShapeDetector::GetMaxAllowedTraces() {
  return max_allowed_traces_per_function;
}

bool DynamicShapeDetector::IsSessionActive() {
  return current_session_ != nullptr;
}

bool DynamicShapeDetector::AllowNewTrace() {
  XLA_CHECK(IsSessionActive());
  return current_session_->traces_ < max_allowed_traces_per_function;
}

void DynamicShapeDetector::EndSession() {
  XLA_CHECK(IsSessionActive());

  try {
    // Mark the current builder_ node as trace boundary.
    // If we did create a new trace, increment the session's trace number.
    if (builder_.MarkTraceBoundary(AllowNewTrace())) {
      current_session_->traces_++;
      TF_VLOG(5) << "Created new trace.";
    }

    ResetSession();
    TF_VLOG(5) << "Ended session: " << current_session_->name_;
  } catch (const std::exception& e) {
    // MarkTraceBoundary might raise an exception if AllowNewTrace() is false.
    // Catch it here, so that we can correctly end the session.
    ResetSession();
    throw;
  }
}

void DynamicShapeDetector::ResetSession() {
  current_session_ = nullptr;
  builder_ = {};
}

void DynamicShapeDetector::RootBuilder() {
  builder_ = current_session_->NewBuilder();
}

void DynamicShapeDetector::AddNodeInfo(torch::lazy::hash_t hash,
                                       const std::string& str) {
  XLA_CHECK(current_session_ != nullptr);

  try {
    builder_.AddValue({hash, str}, AllowNewTrace());
  } catch (const std::exception& e) {
    // AddValue might raise an exception if AllowNewTrace() is false. Catch it
    // here, so that we can correctly return the builder to the root of the
    // trie.
    //
    // TODO(ysiraichi): we should actually rollback this trace.
    RootBuilder();
    throw;
  }
}

TrieBuilder SessionInfo::NewBuilder() { return {root_.get(), 0}; }

void TrieBuilder::AddValue(TrieValue value, bool allow_new_trace) {
  *this = node_->AddValue(value, matched_, allow_new_trace);
}

bool TrieBuilder::MarkTraceBoundary(bool allow_new_trace) {
  return node_->MarkTraceBoundary(matched_, allow_new_trace);
}

}  // namespace torch_xla
