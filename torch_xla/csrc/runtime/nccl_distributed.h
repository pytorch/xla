#ifndef XLA_CLIENT_NCCL_DISTRIBUTED_H_
#define XLA_CLIENT_NCCL_DISTRIBUTED_H_

#include <string>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/types.h"

namespace torch_xla {
namespace runtime {
namespace nccl_detail {

std::string GetNcclUniqueUid(absl::Span<const int64_t> replicas);

}  // namespace nccl_detail
}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_CLIENT_NCCL_DISTRIBUTED_H_
