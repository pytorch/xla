#include "xla/xla_client/nccl_distributed.h"

#include <map>
#include <mutex>

#include "absl/strings/str_join.h"
#include "xla/xla_client/debug_macros.h"
#if XLA_CUDA
#include "third_party/nccl/nccl.h"
#endif

namespace xla {
namespace nccl_detail {

#if XLA_CUDA

namespace {

class NcclUidManager {
 public:
  static NcclUidManager* Get();

  std::string GetNcclUniqueUid(absl::Span<const int64_t> replicas);

 private:
  std::mutex mutex_;
  std::map<std::string, std::string> replicas_uid_map_;
};

NcclUidManager* NcclUidManager::Get() {
  static NcclUidManager* nccl_mgr = new NcclUidManager();
  return nccl_mgr;
}

std::string NcclUidManager::GetNcclUniqueUid(
    absl::Span<const int64_t> replicas) {
  std::string replicas_str = absl::StrJoin(replicas, ",");
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = replicas_uid_map_.find(replicas_str);
  if (it == replicas_uid_map_.end()) {
    ncclUniqueId id;
    ncclResult_t r = ncclGetUniqueId(&id);
    XLA_CHECK_EQ(r, ncclSuccess)
        << "NCCL UID generation failed: replicas=(" << replicas_str
        << "), error: " << ncclGetErrorString(r);
    it = replicas_uid_map_
             .emplace(std::move(replicas_str),
                      std::string(id.internal, NCCL_UNIQUE_ID_BYTES))
             .first;
  }
  return it->second;
}

}  // namespace

std::string GetNcclUniqueUid(absl::Span<const int64_t> replicas) {
  return NcclUidManager::Get()->GetNcclUniqueUid(replicas);
}

#else  // XLA_CUDA

std::string GetNcclUniqueUid(absl::Span<const int64_t> replicas) {
  XLA_ERROR() << "Calling GetNcclUniqueUid() without NCCL configuration";
}

#endif  // XLA_CUDA

}  // namespace nccl_detail
}  // namespace xla
