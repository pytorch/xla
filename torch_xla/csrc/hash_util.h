// Utilities for hash computation.

#ifndef XLA_TORCH_XLA_CSRC_HASH_UTIL_H_
#define XLA_TORCH_XLA_CSRC_HASH_UTIL_H_

#include <torch/csrc/lazy/core/hash.h>

#include <initializer_list>

namespace torch_xla {

// Merges the new hash value into the old hash value by combining them.
// old_hash must not be null. We pass it by pointer instead of reference,
// so that at the call site it's easy to tell the source from the target.
inline void MergeHash(const torch::lazy::hash_t new_hash,
                      torch::lazy::hash_t* const old_hash) {
  *old_hash = torch::lazy::HashCombine(*old_hash, new_hash);
}

// Like the above, but merges a list of hashes instead of a single hash.
inline void MergeHash(
    const std::initializer_list<torch::lazy::hash_t> new_hashes,
    torch::lazy::hash_t* const old_hash) {
  for (const auto hash : new_hashes) {
    MergeHash(hash, old_hash);
  }
}

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_HASH_UTIL_H_
