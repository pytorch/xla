#pragma once

#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/async_task.h"
#include "tensorflow/compiler/xla/xla_client/cache.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

// The OpByOpExecutor class is a singleton accessible via its Get() API that
// allows to run an IR graph is per-IR-node isolation mode. Instead of lowering
// the whole IR graph in a single XLA computation, the single IR nodes are
// lowered and executed independently.
class OpByOpExecutor {
 public:
  using AsyncResult = std::vector<xla::ComputationClient::DataPtr>;
  using AsyncTask = xla::util::AsyncTask<AsyncResult>;

  static OpByOpExecutor* Get();

  std::vector<xla::ComputationClient::ExecuteChainedOp> BuildOps(
      absl::Span<const ir::Value> roots, const std::string& device,
      absl::Span<const std::string> devices);

  std::vector<xla::ComputationClient::DataPtr> Execute(
      absl::Span<const ir::Value> roots, const std::string& device,
      absl::Span<const std::string> devices);

  AsyncTask ExecuteAsync(absl::Span<const ir::Value> roots,
                         const std::string& device,
                         absl::Span<const std::string> devices);

 private:
  using CompileCache =
      xla::util::Cache<xla::hash_t, xla::ComputationClient::Computation,
                       xla::util::HashReducer>;

  explicit OpByOpExecutor(size_t compile_cache_size);

  CompileCache compile_cache_;
};

}  // namespace torch_xla
