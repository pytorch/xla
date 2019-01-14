#ifndef TENSORFLOW_COMPILER_XLA_RPC_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_RPC_UTIL_H_

#include <memory>
#include <vector>

#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace xla {
namespace util {

struct MidPolicy {
  size_t operator()(size_t size) const { return size / 2; }
};

// Hasher for string-like objects which hashes only a partial window of the data
// of size N. The P (policy) type is a functor which returns the position of the
// window.
template <typename T, size_t N = 1024, typename P = MidPolicy>
struct PartialHasher {
  size_t operator()(const T& data) const {
    size_t pos = policy(data.size());
    size_t end = pos + N;
    if (end > data.size()) {
      end = data.size();
      if (N > data.size()) {
        pos = 0;
      } else {
        pos = end - N;
      }
    }
    return tensorflow::Hash64(data.data() + pos, end - pos, 17);
  }

  P policy;
};

template <typename C>
std::vector<const typename C::value_type::element_type*> GetConstSharedPointers(
    const C& shared_pointers) {
  std::vector<const typename C::value_type::element_type*> pointers;
  pointers.reserve(shared_pointers.size());
  for (auto& shared_pointer : shared_pointers) {
    pointers.push_back(shared_pointer.get());
  }
  return pointers;
}

template <typename C>
std::vector<typename C::value_type::element_type*> GetSharedPointers(
    const C& shared_pointers) {
  std::vector<typename C::value_type::element_type*> pointers;
  pointers.reserve(shared_pointers.size());
  for (auto& shared_pointer : shared_pointers) {
    pointers.push_back(shared_pointer.get());
  }
  return pointers;
}

}  // namespace util
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RPC_UTIL_H_
