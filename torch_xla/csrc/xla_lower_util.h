#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace torch_xla {

xla::XlaOp PadToSize(const xla::XlaOp& input, const xla::XlaOp& pad_value,
                     tensorflow::gtl::ArraySlice<const xla::int64> size);

std::vector<xla::XlaOp> CreateKthValue(const xla::XlaOp& input, xla::int64 k,
                                       xla::int64 dim, bool keepdim);

std::vector<xla::XlaOp> CreateTopK(const xla::XlaOp& input, xla::int64 k,
                                   xla::int64 dim, bool largest, bool sorted);

xla::XlaOp CreateMatMul(const xla::XlaOp& lhs, const xla::XlaOp& rhs);

xla::XlaOp BuildBernoulli(const xla::XlaOp& probability,
                          const xla::Shape& shape);

xla::XlaOp BuildDropout(const xla::XlaOp& input, float probability);

xla::XlaOp BuildRandperm(xla::int64 n, xla::PrimitiveType element_type,
                         xla::XlaBuilder* builder);

std::vector<xla::XlaOp> CreateBroadcastTensors(
    tensorflow::gtl::ArraySlice<const xla::XlaOp> operands);

// Similar to tf.gather_nd, used to implement advanced indexing.
xla::XlaOp CreateIndex(const xla::XlaOp& input, const xla::XlaOp& indices,
                       xla::int64 start_dim);

// Similar to tf.scatter_nd, used to implement advanced indexing updates.
xla::XlaOp CreateIndexUpdate(
    const xla::XlaOp& buffer, const xla::XlaOp& indices, xla::int64 start_dim,
    const xla::XlaOp& updates,
    const std::function<xla::XlaOp(const xla::XlaOp&, const xla::XlaOp&)>&
        combiner);

xla::XlaOp CreateIndexAdd(const xla::XlaOp& buffer, xla::int64 dim,
                          const xla::XlaOp& index, const xla::XlaOp& value);

xla::XlaOp CreateIndexCopy(const xla::XlaOp& buffer, xla::int64 dim,
                           const xla::XlaOp& index, const xla::XlaOp& value);

xla::XlaOp CreateIndexFill(const xla::XlaOp& buffer, xla::int64 dim,
                           const xla::XlaOp& index, const xla::XlaOp& values);

// Used to lower scatter and scatter_add.
xla::XlaOp CreateScatter(
    const xla::XlaOp& input, const xla::XlaOp& index, const xla::XlaOp& src,
    xla::int64 dim,
    const std::function<xla::XlaOp(const xla::XlaOp&, const xla::XlaOp&)>&
        combiner);

}  // namespace torch_xla
