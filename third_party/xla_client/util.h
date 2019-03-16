#ifndef TENSORFLOW_COMPILER_XLA_RPC_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_RPC_UTIL_H_

#include <functional>
#include <memory>
#include <numeric>
#include <set>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace xla {
namespace util {

class Cleanup {
 public:
  explicit Cleanup(std::function<void(Status)> func) : func_(std::move(func)) {}
  Cleanup(Cleanup&& ref) : func_(std::move(ref.func_)) {}
  Cleanup(const Cleanup&) = delete;

  ~Cleanup() {
    if (func_ != nullptr) {
      func_(std::move(status_));
    }
  }

  Cleanup& operator=(const Cleanup&) = delete;

  Cleanup& operator=(Cleanup&& ref) {
    if (this != &ref) {
      func_ = std::move(ref.func_);
    }
    return *this;
  }

  void Release() { func_ = nullptr; }

  void SetStatus(Status status) { status_ = std::move(status); }

 private:
  std::function<void(Status)> func_;
  Status status_;
};

// Allows APIs which might return const references and values, to not be forced
// to return values in the signature.
template <typename T>
class MaybeRef {
 public:
  MaybeRef(const T& ref) : ref_(ref) {}
  MaybeRef(T&& value) : storage_(std::move(value)), ref_(*storage_) {}

  const T& get() const { return ref_; }

  const T& operator*() const { return get(); }

  operator const T&() const { return get(); }

  bool is_stored() const { return storage_.has_value(); }

 private:
  absl::optional<T> storage_;
  const T& ref_;
};

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

template <typename T>
std::vector<T> Iota(size_t size, T init = 0) {
  std::vector<T> result(size);
  std::iota(result.begin(), result.end(), init);
  return result;
}

template <typename T>
std::vector<T> Range(T start, T end, T step = 1) {
  std::vector<T> result;
  result.reserve(static_cast<size_t>((end - start) / step));
  for (; start < end; start += step) {
    result.push_back(start);
  }
  return result;
}

template <typename T, typename S>
std::vector<T> ToVector(const S& input) {
  return std::vector<T>(input.begin(), input.end());
}

template <typename T, typename S>
T Multiply(const S& input) {
  return std::accumulate(input.begin(), input.end(), T(1),
                         std::multiplies<T>());
}

static inline size_t HashCombine(size_t a, size_t b) {
  return a ^ (b + 0x9e3779b97f4a7c15 + (a << 6) + (a >> 2));
}

template <typename T>
size_t Hash(const T& value) {
  return std::hash<T>()(value);
}

// Forward declare to allow hashes of vectors of vectors to work.
template <typename T>
size_t ContainerHash(const T& values);

template <typename T>
size_t Hash(tensorflow::gtl::ArraySlice<const T> values) {
  return ContainerHash(values);
}

template <typename T>
size_t Hash(const std::vector<T>& values) {
  return ContainerHash(values);
}

template <typename T>
size_t Hash(const std::set<T>& values) {
  return ContainerHash(values);
}

template <typename T>
size_t ContainerHash(const T& values) {
  size_t h = 0x5a2d296e9;
  for (auto& value : values) {
    h = HashCombine(h, Hash(value));
  }
  return h;
}

template <typename T = void>
size_t MHash() {
  return 0x5a2d296e9;
}

template <typename T, typename... Targs>
size_t MHash(T value, Targs... Fargs) {
  return HashCombine(Hash(value), MHash(Fargs...));
}

}  // namespace util
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RPC_UTIL_H_
