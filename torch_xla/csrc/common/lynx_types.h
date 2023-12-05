#ifndef XLA_TORCH_XLA_CSRC_COMMON_LYNX_TYPES_H_
#define XLA_TORCH_XLA_CSRC_COMMON_LYNX_TYPES_H_
#include <string>
#include <unordered_map>
#include <utility>

#include "torch_xla/csrc/common/singleton.h"

namespace lynx {

struct P2PChannelsManager : public Singleton<P2PChannelsManager> {
  typedef long long int int64_t;

 public:
  std::unordered_map<int64_t, std::pair<int64_t, int64_t>>* GetChannelsMap() {
    return &map_;
  }

 private:
  std::unordered_map<int64_t, std::pair<int64_t, int64_t>> map_;
  friend class Singleton<P2PChannelsManager>;
};

}  // namespace lynx

#endif