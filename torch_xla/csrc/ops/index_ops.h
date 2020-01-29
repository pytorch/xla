// Indexing tensors by by tensors
//
// This corresponds to "advanced indexing" in NumPy. The two operations are:
//
//  index(Tensor self, indices) -> Tensor
//  index_put_(Tensor self, indices, value, accumulate=false)
//
// The index is a TensorList containg kLong or kByte tensors or nulls. Byte
// tensors (boolean masks) are expanded to long tensors via nonzero(). Null
// tensors signify that the dimension is not indexed.
//
// All indexes are broadcast together and iterated as *one*. From NumPy:
//
// result[i_1, ..., i_M] == x[ind_1[i_1, ..., i_M], ind_2[i_1, ..., i_M],
//                           ..., ind_N[i_1, ..., i_M]]
//
// Note 1: ByteTensors expand to index as many dimensions as there are in the
// mask.
//
// Note 2: The behavior is more complicated when the index tensors are not all
// adjacent (e.g. x[[0, 1], :, [2, 3]]). In this case, self and the index
// tensors are transposed to the front: x.transpose(1, 2)[[0, 1], [2, 3]]

#pragma once

#include <ATen/core/Tensor.h>

#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/tensor.h"

namespace torch_xla {

struct CanonicalIndexInfo {
  at::Tensor base;
  std::vector<at::Tensor> indices;
  // The permutation to be applied to the result. This is needed for indexed
  // updates, since a permutation is applied to the base to bring non-null
  // indices to front. This is the inverse of that permutation.
  std::vector<xla::int64> result_permutation;
  // The dimension number at which indexing starts.
  xla::int64 start_dim = 0;
};

// Transform the given base and indices to a form supported by the XLATensor
// index implementation. Input indices are reordered so that non-null indices
// are first and the tail of null indices is dropped. The dimensions of the base
// are reordered to be consistent with this reordering.
CanonicalIndexInfo GetCanonicalIndexInfo(const at::Tensor& base,
                                         at::TensorList orig_indices);

// Expands a rank <= 1 tensor to rank 1, if necessary.
ir::Value EnsureRank1(const ir::Value& index);

// Implements indexing by tensors of long according to the top-level
// description.
XLATensor IndexByTensors(const XLATensor& base,
                         absl::Span<const XLATensor> indices,
                         xla::int64 start_dim);

ir::Value IndexPutByTensors(const XLATensor& base,
                            absl::Span<const XLATensor> indices,
                            xla::int64 start_dim, const XLATensor& updates,
                            bool accumulate,
                            absl::Span<const xla::int64> result_permutation);

ir::NodePtr IndexFill(const XLATensor& base, xla::int64 dim,
                      const XLATensor& index, at::Scalar value);

ir::NodePtr IndexFill(const XLATensor& base, xla::int64 dim,
                      const XLATensor& index, const XLATensor& value);

ir::Value IndexAdd(const XLATensor& base, xla::int64 dim,
                   const XLATensor& index, const XLATensor& source);

ir::Value IndexCopy(const XLATensor& base, xla::int64 dim,
                    const XLATensor& index, const XLATensor& source);

}  // namespace torch_xla
