#include <torch/csrc/lazy/core/hash.h>

namespace torch_xla {
namespace runtime {
namespace hash {

// Take a hash of XLA flags which impact the compilation result.
// TODO(jonbolin): We should move away from manually hashing the env vars and
// instead hash the compilation environment directly when the functionality is
// supported in the runtime.
torch::lazy::hash_t HashXlaEnvVars(bool force_rehash = false);

}  // namespace hash
}  // namespace runtime
}  // namespace torch_xla
