#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace torch_xla {

std::vector<xla::XlaOp> CreateKthValue(const xla::XlaOp& input, xla::int64 k,
                                       xla::int64 dim, bool keepdim);

std::vector<xla::XlaOp> CreateTopK(const xla::XlaOp& input, xla::int64 k,
                                   xla::int64 dim, bool largest, bool sorted);

xla::XlaOp CreateMatMul(const xla::XlaOp& lhs, const xla::XlaOp& rhs);

xla::XlaOp BuildDropout(const xla::XlaOp& input, float probability);

std::vector<xla::XlaOp> CreateBroadcastTensors(
    tensorflow::gtl::ArraySlice<const xla::XlaOp> operands);

// Similar to tf.gather_nd, used to implement advanced indexing.
xla::XlaOp CreateIndex(const xla::XlaOp& input, const xla::XlaOp& indices);

// Similar to tf.scatter_nd, used to implement advanced indexing updates.
xla::XlaOp CreateIndexUpdate(
    const xla::XlaOp& buffer, const xla::XlaOp& indices,
    const xla::XlaOp& updates,
    const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp, xla::XlaBuilder*)>&
        combiner);

xla::XlaOp CreateIndexAdd(const xla::XlaOp& buffer, xla::int64 dim,
                          const xla::XlaOp& index, const xla::XlaOp& value);

xla::XlaOp CreateIndexCopy(const xla::XlaOp& buffer, xla::int64 dim,
                           const xla::XlaOp& index, const xla::XlaOp& value);

xla::XlaOp CreateIndexFill(const xla::XlaOp& buffer, xla::int64 dim,
                           const xla::XlaOp& index, const xla::XlaOp& values);

}  // namespace torch_xla
