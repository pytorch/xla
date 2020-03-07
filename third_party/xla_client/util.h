#ifndef XLA_CLIENT_UTIL_H_
#define XLA_CLIENT_UTIL_H_

#include <algorithm>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <numeric>
#include <set>
#include <type_traits>
#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace xla {
namespace util {

template <typename F>
Status CheckedCall(const F& fn) {
  try {
    fn();
  } catch (const std::exception& ex) {
    return tensorflow::errors::Internal(ex.what());
  }
  return Status::OK();
}

template <typename T>
class Cleanup {
 public:
  using StatusType = T;

  explicit Cleanup(std::function<void(StatusType)> func)
      : func_(std::move(func)) {}
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

  void SetStatus(StatusType status) { status_ = std::move(status); }

  const StatusType& GetStatus() const { return status_; }

 private:
  std::function<void(StatusType)> func_;
  StatusType status_;
};

using ExceptionCleanup = Cleanup<std::exception_ptr>;
using StatusCleanup = Cleanup<xla::Status>;

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

template <class T>
class MaybePtr {
 public:
  MaybePtr(T* ptr) : ptr_(ptr) {
    if (ptr_ == nullptr) {
      storage_ = T();
      ptr_ = &storage_.value();
    }
  }

  T* get() const { return ptr_; }

  T* operator->() const { return get(); }

  T& operator*() const { return *get(); }

 private:
  T* ptr_ = nullptr;
  absl::optional<T> storage_;
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

template <typename C, typename K, typename T, typename F>
void InsertCombined(C* map, const K& key, const T& value, const F& combiner) {
  auto it = map->find(key);
  if (it == map->end()) {
    map->emplace(key, value);
  } else {
    it->second = combiner(it->second, value);
  }
}

template <typename T>
std::vector<T> Iota(size_t size, T init = 0, T incr = 1) {
  std::vector<T> result(size);
  T value = init;
  for (size_t i = 0; i < size; ++i, value += incr) {
    result[i] = value;
  }
  return result;
}

template <typename T>
std::vector<T> Range(T start, T end, T step = 1) {
  std::vector<T> result;
  result.reserve(static_cast<size_t>((end - start) / step));
  if (start < end) {
    for (; start < end; start += step) {
      result.push_back(start);
    }
  } else {
    for (; start > end; start += step) {
      result.push_back(start);
    }
  }
  return result;
}

template <typename T, typename S>
std::vector<T> ToVector(const S& input) {
  return std::vector<T>(input.begin(), input.end());
}

template <typename T>
std::vector<T> GetValuesVector(
    absl::Span<const T> values,
    absl::Span<const absl::optional<T>* const> opt_values) {
  std::vector<T> result(values.begin(), values.end());
  for (auto opt : opt_values) {
    if (*opt) {
      result.push_back(*(*opt));
    }
  }
  return result;
}

template <typename T, typename S>
bool Equal(const T& v1, const S& v2) {
  return std::equal(v1.begin(), v1.end(), v2.begin());
}

template <typename T>
typename T::mapped_type FindOr(const T& cont, const typename T::key_type& key,
                               const typename T::mapped_type& defval) {
  auto it = cont.find(key);
  return it != cont.end() ? it->second : defval;
}

template <typename T, typename G>
const typename T::mapped_type& MapInsert(T* cont,
                                         const typename T::key_type& key,
                                         const G& gen) {
  auto it = cont->find(key);
  if (it == cont->end()) {
    it = cont->emplace(key, gen()).first;
  }
  return it->second;
}

template <typename T>
typename std::underlying_type<T>::type GetEnumValue(T value) {
  return static_cast<typename std::underlying_type<T>::type>(value);
}

template <typename T, typename S>
T Multiply(const S& input) {
  return std::accumulate(input.begin(), input.end(), T(1),
                         std::multiplies<T>());
}

static inline size_t DataHash(const void* data, size_t size) {
  return tensorflow::Hash64(reinterpret_cast<const char*>(data), size,
                            0xc2b2ae3d27d4eb4f);
}

static inline size_t StringHash(const char* data) {
  return DataHash(data, std::strlen(data));
}

static inline size_t HashCombine(size_t a, size_t b) {
  return a ^
         (b * 0x27d4eb2f165667c5 + 0x9e3779b97f4a7c15 + (a << 6) + (a >> 2));
}

template <typename T, typename std::enable_if<
                          std::is_arithmetic<T>::value>::type* = nullptr>
size_t Hash(const T& value) {
  return DataHash(&value, sizeof(value));
}

static inline size_t Hash(const std::string& value) {
  return DataHash(value.data(), value.size());
}

// Forward declare to allow hashes of vectors of vectors to work.
template <typename T>
size_t ContainerHash(const T& values);

template <typename T>
size_t Hash(absl::Span<const T> values) {
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
  size_t h = 0x85ebca77c2b2ae63;
  for (auto& value : values) {
    h = HashCombine(h, Hash(value));
  }
  return h;
}

template <typename T = void>
size_t MHash() {
  return 0x165667b19e3779f9;
}

template <typename T, typename... Targs>
size_t MHash(T value, Targs... Fargs) {
  return HashCombine(Hash(value), MHash(Fargs...));
}

}  // namespace util
}  // namespace xla

#endif  // XLA_CLIENT_UTIL_H_
