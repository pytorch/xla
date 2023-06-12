#ifndef XLA_TORCH_XLA_CSRC_OP_BY_OP_EXECUTOR_H_
#define XLA_TORCH_XLA_CSRC_OP_BY_OP_EXECUTOR_H_

#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/types.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/runtime/async_task.h"
#include "torch_xla/csrc/runtime/cache.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/util.h"

namespace torch_xla {

// The OpByOpExecutor class is a singleton accessible via its Get() API that
// allows to run an IR graph is per-IR-node isolation mode. Instead of lowering
// the whole IR graph in a single XLA computation, the single IR nodes are
// lowered and executed independently.
class OpByOpExecutor {
 public:
  using AsyncResult = std::vector<torch::lazy::BackendDataPtr>;
  using AsyncTask = runtime::util::AsyncTask<AsyncResult>;

  static OpByOpExecutor* Get();

  std::vector<runtime::ComputationClient::ExecuteChainedOp> BuildOps(
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
      runtime::util::Cache<torch::lazy::hash_t,
                           runtime::ComputationClient::Computation,
                           torch::lazy::HashReducer>;

  explicit OpByOpExecutor(size_t compile_cache_size);

  CompileCache compile_cache_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OP_BY_OP_EXECUTOR_H_
