#ifndef XLA_CLIENT_NCCL_DISTRIBUTED_H_
#define XLA_CLIENT_NCCL_DISTRIBUTED_H_

#include <string>

#include "absl/types/span.h"
#include "xla/types.h"

namespace xla {
namespace nccl_detail {

std::string GetNcclUniqueUid(absl::Span<const int64_t> replicas);

}  // namespace nccl_detail
}  // namespace xla

#endif  // XLA_CLIENT_NCCL_DISTRIBUTED_H_
