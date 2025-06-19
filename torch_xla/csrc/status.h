#ifndef XLA_TORCH_XLA_CSRC_STATUS_H_
#define XLA_TORCH_XLA_CSRC_STATUS_H_

#include "absl/status/statusor.h"

namespace torch_xla {

#define _XLA_RETURN_IF_ERROR_AND_THEN(REXPR, ...) \
  {                                               \
    auto& _statusor = (REXPR);                    \
    if (!_statusor.ok()) {                        \
      return _statusor.status();                  \
    }                                             \
    __VA_ARGS__                                   \
  }

// Propagates `REXPR`, in case it's a non-ok status.
#define XLA_RETURN_IF_ERROR(REXPR) _XLA_RETURN_IF_ERROR_AND_THEN(REXPR)

// Propagates `REXPR`, in case it's a non-ok status. Otherwise, assign
// its result to `LHS`.
#define XLA_ASSIGN_OR_RETURN(LHS, REXPR) \
  _XLA_RETURN_IF_ERROR_AND_THEN(REXPR, { LHS = std::move(_status.value()); })

// Consumes the `status` and maybe throws an exception if `status` has
// a non-ok code.
//
// Ideally, this function should be used only used in the project's
// boundary, e.g. when we need to throw an exception for the user to see.
void ConsumeAndMaybeThrow(absl::Status status);

// Consumes the `status`, either returning the value it holds (for
// ok status), or throwing an exception.
template <class T>
T ConsumeAndMaybeThrow(absl::StatusOr<T> status) {
  ConsumeAndMaybeThrow(status.status());
  return std::move(status.value());
}

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_STATUS_H_
