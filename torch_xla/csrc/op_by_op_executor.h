#ifndef XLA_TORCH_XLA_CSRC_OP_BY_OP_EXECUTOR_H_
#define XLA_TORCH_XLA_CSRC_OP_BY_OP_EXECUTOR_H_

#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/types.h"
#include "third_party/xla_client/async_task.h"
#include "third_party/xla_client/cache.h"
#include "third_party/xla_client/computation_client.h"
#include "third_party/xla_client/util.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

// The OpByOpExecutor class is a singleton accessible via its Get() API that
// allows to run an IR graph is per-IR-node isolation mode. Instead of lowering
// the whole IR graph in a single XLA computation, the single IR nodes are
// lowered and executed independently.
class OpByOpExecutor {
 public:
  using AsyncResult = std::vector<torch::lazy::BackendDataPtr>;
  using AsyncTask = xla::util::AsyncTask<AsyncResult>;

  static OpByOpExecutor* Get();

  std::vector<xla::ComputationClient::ExecuteChainedOp> BuildOps(
      c10::ArrayRef<torch::lazy::Value> roots, const std::string& device,
      absl::Span<const std::string> devices);

  std::vector<torch::lazy::BackendDataPtr> Execute(
      c10::ArrayRef<torch::lazy::Value> roots, const std::string& device,
      absl::Span<const std::string> devices);

  AsyncTask ExecuteAsync(c10::ArrayRef<torch::lazy::Value> roots,
                         const std::string& device,
                         absl::Span<const std::string> devices);

 private:
  using CompileCache =
      xla::util::Cache<torch::lazy::hash_t, xla::ComputationClient::Computation,
                       torch::lazy::HashReducer>;

  explicit OpByOpExecutor(size_t compile_cache_size);

  CompileCache compile_cache_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OP_BY_OP_EXECUTOR_H_