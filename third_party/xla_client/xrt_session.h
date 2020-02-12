#ifndef XLA_CLIENT_XRT_SESSION_H_
#define XLA_CLIENT_XRT_SESSION_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/types/span.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace xla {

// Encapsulates an XRT session and its associated node cache. XrtSession are not
// thread safe, but are always accessed by one thread at a time. The
// XrtSessionCache will keep creating new sessions if not enough are available
// to satisfy the threads requests.
class XrtSession {
 public:
  // A cached node captures that single node, or the mini-graph root node,
  // together with the place-holders necessary to feed the node/sub-graph.
  // The end-point node can be either a tensorflow Operation or an Output.
  struct CachedNode {
    CachedNode(tensorflow::Output output,
               std::vector<tensorflow::ops::Placeholder> holders)
        : holders(std::move(holders)) {
      outputs.push_back(std::move(output));
    }
    CachedNode(tensorflow::Operation operation,
               std::vector<tensorflow::ops::Placeholder> holders)
        : holders(std::move(holders)) {
      operations.push_back(std::move(operation));
    }
    CachedNode(std::vector<tensorflow::Output> outputs,
               std::vector<tensorflow::ops::Placeholder> holders)
        : outputs(std::move(outputs)), holders(std::move(holders)) {}
    CachedNode(std::vector<tensorflow::Operation> operations,
               std::vector<tensorflow::ops::Placeholder> holders)
        : operations(std::move(operations)), holders(std::move(holders)) {}

    std::vector<tensorflow::Output> outputs;
    std::vector<tensorflow::Operation> operations;
    std::vector<tensorflow::ops::Placeholder> holders;
  };

  // The node cache holds a set of CachedNode of the same kind (by the means of
  // the NodeTypes entries).
  // The NodeCache access is not thread safe, but so is XrtSession.
  class NodeCache {
   public:
    bool Empty() const { return position_ >= nodes_.size(); }

    const CachedNode& Get() {
      XLA_CHECK_LT(position_, nodes_.size());
      ++position_;
      return *nodes_[position_ - 1];
    }

    void Add(std::shared_ptr<CachedNode> node) {
      nodes_.push_back(std::move(node));
    }

    void Rewind() { position_ = 0; }

   private:
    std::vector<std::shared_ptr<CachedNode>> nodes_;
    size_t position_ = 0;
  };

  explicit XrtSession(const tensorflow::SessionOptions& session_options);

  const std::string& target() const { return target_; }

  tensorflow::Scope* root() { return &root_; }

  tensorflow::ClientSession* session() { return &session_; }

  NodeCache* GetNodeCache(const std::string& key) { return &node_cache_[key]; }

  void Reset();

  static std::string GetCacheKey(const std::string& op_name,
                                 const std::string& device);

 private:
  std::string target_;
  tensorflow::Scope root_;
  tensorflow::ClientSession session_;
  std::map<std::string, NodeCache> node_cache_;
};

}  // namespace xla

#endif  // XLA_CLIENT_XRT_SESSION_H_
