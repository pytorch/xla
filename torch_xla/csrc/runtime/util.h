#ifndef XLA_CLIENT_UTIL_H_
#define XLA_CLIENT_UTIL_H_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "torch_xla/csrc/runtime/types.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/hash.h"

namespace torch_xla {
namespace runtime {
namespace util {

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

template <typename T, typename S>
T Multiply(const S& input) {
  return std::accumulate(input.begin(), input.end(), T(1),
                         std::multiplies<T>());
}

namespace internal {

// ExtractStatusOrValue<U>::type is T if U is absl::StatusOr<T>, and is
// undefined otherwise.
template <typename U>
struct ExtractStatusOrValue;
template <typename T>
struct ExtractStatusOrValue<absl::StatusOr<T>> {
  using type = T;
};

}  // namespace internal

// RaisePythonValueErrorOnFailure(func) requires `func` to be a functor that
// takes no argument and returns an absl::StatusOr<T>. It's a wrapper of
// `func()` that translates any failure in `func()` to a Python ValueError
// exception. In particular:
//
//   - if `func()` returns an error, throws an std::invalid_argument,
//     which is translated to a Python ValueError exception;
//     (https://pybind11.readthedocs.io/en/stable/advanced/exceptions.html).
//   - if `func()` throws any exception, rethrows it as an
//     std::invalid_argument so that we get a Python ValueError;
//   - if `func()` successfully returns a value of type T, returns the value;
//   - however, if `func()` crashes (e.g. due to a CHECK), we cannot
//     catch it; therefore we should ensure that `func()` never
//     crashes (and fix any crash as a bug).
template <typename Func>
typename internal::ExtractStatusOrValue<decltype(std::declval<Func>()())>::type
RaisePythonValueErrorOnFailure(const Func& func) {
  decltype(std::declval<Func>()()) result;
  try {
    result = func();
  } catch (const std::exception& e) {
    throw std::invalid_argument(e.what());
  } catch (...) {
    throw std::invalid_argument(
        "Function threw an unknown exception. Please file a bug at "
        "https://github.com/pytorch/xla/issues with details on how to "
        "reproduce the error.");
  }
  if (result.ok()) {
    return *std::move(result);
  }
  throw std::invalid_argument(std::string(result.status().message()));
}

}  // namespace util
}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_CLIENT_UTIL_H_
