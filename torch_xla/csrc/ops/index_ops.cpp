#include "torch_xla/csrc/ops/index_ops.h"

#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>

#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/arithmetic_ir_ops.h"
#include "torch_xla/csrc/ops/expand.h"
#include "torch_xla/csrc/ops/index_get.h"
#include "torch_xla/csrc/ops/index_put.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/ops.h"
#include "torch_xla/csrc/ops/permute.h"
#include "torch_xla/csrc/ops/scalar.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace {

void CheckIndexTensorTypes(
    const c10::List<c10::optional<at::Tensor>>& indices) {
  for (const c10::optional<at::Tensor>& tensor : indices) {
    if (tensor.has_value() && tensor->defined()) {
      at::ScalarType scalar_type = tensor->scalar_type();
      if (scalar_type != at::kLong && scalar_type != at::kByte &&
          scalar_type != at::kBool) {
        XLA_ERROR() << "Tensors used as indices must be long, byte or boolean "
                       "tensors, found scalar type: "
                    << scalar_type;
      }
    }
  }
}

// Expands byte tensors (masks) into the equivalent indexing by LongTensors.
// This is a version of at::native::expandByteTensors with style adjustments.
std::vector<at::Tensor> ExpandByteTensors(
    const at::Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices) {
  std::vector<at::Tensor> result;
  for (const c10::optional<at::Tensor>& index : indices) {
    if (index.has_value() && (index->scalar_type() == at::kByte ||
                              index->scalar_type() == at::kBool)) {
      // The sizes of the ByteTensor mask must match the sizes of the
      // corresponding dimensions in self.
      for (int64_t j = 0; j < index->dim(); j++) {
        int64_t src_idx = result.size() + j;
        XLA_CHECK_EQ(index->size(j), self.size(src_idx))
            << "The shape of the mask " << index->sizes() << " at index " << j
            << " does not match the shape of the indexed tensor "
            << self.sizes() << " at index " << src_idx;
      }
      // Replace with nonzeros.
      auto nonzero = index->nonzero();
      for (int64_t j = 0; j < index->dim(); j++) {
        result.emplace_back(nonzero.select(1, j));
      }
    } else {
      result.emplace_back(index.value_or(at::Tensor()));
    }
  }
  return result;
}

struct IndexAdjacencyInfo {
  bool contiguous_non_null = false;
  xla::int64_t start_dim = 0;
};

// Checks whether all the non-null tensors are adjacent, in which case we must
// not permute the base and instead treat the null tensors prefix as a no-op.
// Replicates the behavior of at::native::hasContiguousSubspace and also returns
// the position of the first non-null index.
IndexAdjacencyInfo GetIndexAdjacencyInfo(at::TensorList indices) {
  auto is_defined = [](const at::Tensor& tensor) { return tensor.defined(); };
  auto is_null = [](const at::Tensor& tensor) { return !tensor.defined(); };
  auto start = std::find_if(indices.begin(), indices.end(), is_defined);
  auto stop = std::find_if(indices.rbegin(), indices.rend(), is_defined);
  auto it = std::find_if(start, stop.base(), is_null);
  xla::int64_t start_dim = std::distance(indices.begin(), start);
  return {it == stop.base(), start_dim};
}

// Transposes the tensor and indices together so that all the non-null indices
// index the first k dimensions of the tensor. Returns the transposed tensor and
// the reordered indices. For example:
//  TransposeToFront(tensor, {nullptr, a, nullptr, b})
// returns
//  tensor.permute([1, 3, 0, 2]), {a, b}
//
// This is a simplified version of at::native::transposeToFront which better
// fits our requirements.
CanonicalIndexInfo TransposeToFront(at::Tensor base, at::TensorList indices) {
  std::vector<int64_t> dims;
  std::vector<at::Tensor> transposed_indices;
  size_t base_rank = base.dim();
  dims.reserve(base_rank);
  XLA_CHECK_LE(indices.size(), base_rank);
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i].defined()) {
      dims.push_back(i);
      transposed_indices.emplace_back(indices[i]);
    }
  }
  for (size_t i = 0; i < indices.size(); i++) {
    if (!indices[i].defined()) {
      dims.push_back(i);
    }
  }
  for (size_t i = indices.size(); i < base_rank; ++i) {
    dims.push_back(i);
  }
  IndexAdjacencyInfo adjacency_info = GetIndexAdjacencyInfo(indices);
  if (adjacency_info.contiguous_non_null) {
    return {base, std::move(transposed_indices),
            xla::util::Iota<xla::int64_t>(base_rank), adjacency_info.start_dim};
  }
  return {base.permute(dims), std::move(transposed_indices),
          xla::InversePermutation(XlaHelpers::I64List(dims)), 0};
}

// Wraps index tensors once into the [0, dim_size) interval, where dim_size is
// the size of the current indexed dimension.
std::vector<XLATensor> WrapIndicesOnce(const XLATensor& base,
                                       absl::Span<const XLATensor> indices,
                                       int start_dim) {
  std::vector<XLATensor> canonical_indices;
  auto base_shape_ref = base.shape();
  XLA_CHECK_LE(indices.size(), base_shape_ref.get().rank());
  for (size_t dim_idx = 0; dim_idx < indices.size(); ++dim_idx) {
    const XLATensor& dim_index = indices[dim_idx];
    int64_t dim_size = base_shape_ref.get().dimensions(dim_idx + start_dim);
    XLATensor wrapped_dim_index = XLATensor::Create(
        dim_index.GetIrValue() +
            XLATensor::GetIrValueForScalar(dim_size, dim_index.shape(),
                                           base.GetDevice()),
        base.GetDevice());
    XLATensor wrap_cond =
        XLATensor::lt(indices[dim_idx], at::Scalar(int64_t(0)));
    canonical_indices.push_back(
        XLATensor::where(wrap_cond, wrapped_dim_index, dim_index));
  }
  return canonical_indices;
}

ir::NodePtr IndexFillOp(const ir::Value& buffer, xla::int64_t dim,
                        const ir::Value& index, const ir::Value& value) {
  auto lower_fn = [dim](const ir::Node& node,
                        ir::LoweringContext* loctx) -> ir::XlaOpVector {
    xla::XlaOp xla_base = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_index = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_value = loctx->GetOutputOp(node.operand(2));
    return node.ReturnOp(CreateIndexFill(xla_base, dim, xla_index, xla_value),
                         loctx);
  };
  auto lower_for_shape_fn =
      [dim](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return CreateIndexFill(operands[0], dim, operands[1], operands[2]);
  };
  ir::Value index_rank1 = EnsureRank1(index);
  return ir::ops::GenericOp(
      ir::OpKind(at::aten::index_fill), {buffer, index_rank1, value},
      [&]() {
        return ir::ops::InferOutputShape(
            {buffer.shape(), index_rank1.shape(), value.shape()},
            lower_for_shape_fn);
      },
      std::move(lower_fn), /*num_outputs=*/1, torch::lazy::MHash(dim));
}

ir::NodePtr IndexAddOp(const ir::Value& buffer, xla::int64_t dim,
                       const ir::Value& index, const ir::Value& source) {
  auto lower_fn = [dim](const ir::Node& node,
                        ir::LoweringContext* loctx) -> ir::XlaOpVector {
    xla::XlaOp xla_base = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_index = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_source = loctx->GetOutputOp(node.operand(2));
    return node.ReturnOp(CreateIndexAdd(xla_base, dim, xla_index, xla_source),
                         loctx);
  };
  auto lower_for_shape_fn =
      [dim](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return CreateIndexAdd(operands[0], dim, operands[1], operands[2]);
  };
  ir::Value index_rank1 = EnsureRank1(index);
  return ir::ops::GenericOp(
      ir::OpKind(at::aten::index_add), {buffer, index_rank1, source},
      [&]() {
        return ir::ops::InferOutputShape(
            {buffer.shape(), index_rank1.shape(), source.shape()},
            lower_for_shape_fn);
      },
      std::move(lower_fn));
}

ir::NodePtr IndexCopyOp(const ir::Value& buffer, xla::int64_t dim,
                        const ir::Value& index, const ir::Value& source) {
  auto lower_fn = [dim](const ir::Node& node,
                        ir::LoweringContext* loctx) -> ir::XlaOpVector {
    xla::XlaOp xla_base = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_index = loctx->GetOutputOp(node.operand(1));
    xla::XlaOp xla_source = loctx->GetOutputOp(node.operand(2));
    return node.ReturnOp(CreateIndexCopy(xla_base, dim, xla_index, xla_source),
                         loctx);
  };
  auto lower_for_shape_fn =
      [dim](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return CreateIndexCopy(operands[0], dim, operands[1], operands[2]);
  };
  ir::Value index_rank1 = EnsureRank1(index);
  return ir::ops::GenericOp(
      ir::OpKind(at::aten::index_copy), {buffer, index_rank1, source},
      [&]() {
        return ir::ops::InferOutputShape(
            {buffer.shape(), index_rank1.shape(), source.shape()},
            lower_for_shape_fn);
      },
      std::move(lower_fn));
}

}  // namespace

CanonicalIndexInfo GetCanonicalIndexInfo(
    const at::Tensor& base,
    const c10::List<c10::optional<at::Tensor>>& orig_indices) {
  CheckIndexTensorTypes(orig_indices);
  // First expand ByteTensor (boolean masks) into 1 or more LongTensors, then
  // broadcast all index tensors together.
  auto indices = at::expand_outplace(ExpandByteTensors(base, orig_indices));
  // If the non-null indices are not all adjacent, transpose base and indices
  // together so that they're adjacent at the front.
  CanonicalIndexInfo canonical_index_info = TransposeToFront(base, indices);
  return canonical_index_info;
}

ir::Value EnsureRank1(const ir::Value& index) {
  XLA_CHECK_LE(index->shape().rank(), 1);
  return index->shape().rank() == 0 ? ir::MakeNode<ir::ops::Expand>(
                                          index, std::vector<xla::int64_t>{1})
                                    : index;
}

XLATensor IndexByTensors(const XLATensor& base,
                         absl::Span<const XLATensor> indices,
                         xla::int64_t start_dim) {
  if (indices.empty()) {
    return base;
  }
  auto canonical_indices = WrapIndicesOnce(base, indices, start_dim);
  xla::int64_t indices_rank = canonical_indices.front().shape().get().rank();
  // Stack the indices to allow the whole multi-indexing to be dispatched with a
  // single gather.
  XLATensor indices_nd = XLATensor::stack(canonical_indices, indices_rank);
  return XLATensor::Create(
      ir::MakeNode<ir::ops::IndexGet>(base.GetIrValue(),
                                      indices_nd.GetIrValue(), start_dim),
      base.GetDevice(), base.dtype());
}

ir::Value IndexPutByTensors(const XLATensor& base,
                            absl::Span<const XLATensor> indices,
                            xla::int64_t start_dim, const XLATensor& values,
                            bool accumulate,
                            absl::Span<const xla::int64_t> result_permutation) {
  if (indices.empty()) {
    return base.GetIrValue();
  }
  auto canonical_indices = WrapIndicesOnce(base, indices, start_dim);
  xla::int64_t indices_rank = canonical_indices.front().shape().get().rank();
  // Stack the indices to allow the whole multi-indexing to be dispatched with a
  // single scatter.
  XLATensor indices_nd = XLATensor::stack(canonical_indices, indices_rank);
  return ir::MakeNode<ir::ops::Permute>(
      ir::MakeNode<ir::ops::IndexPut>(base.GetIrValue(),
                                      indices_nd.GetIrValue(), start_dim,
                                      values.GetIrValue(), accumulate),
      xla::util::ToVector<xla::int64_t>(result_permutation));
}

ir::NodePtr IndexFill(const XLATensor& base, xla::int64_t dim,
                      const XLATensor& index, const at::Scalar& value) {
  XLA_CHECK_EQ(index.dtype(), at::ScalarType::Long)
      << "Fill index is expected to be of scalar type Long, but it is "
      << index.dtype();
  XLA_CHECK_LE(index.shape().get().rank(), 1)
      << "Fill index is supposed to be a vector";
  return IndexFillOp(
      base.GetIrValue(), dim, index.GetIrValue(),
      XLATensor::GetIrValueForScalar(value, base.shape().get().element_type(),
                                     base.GetDevice()));
}

ir::NodePtr IndexFill(const XLATensor& base, xla::int64_t dim,
                      const XLATensor& index, const XLATensor& value) {
  XLA_CHECK_EQ(index.dtype(), at::ScalarType::Long)
      << "Fill index is expected to be of scalar type Long, but it is "
      << index.dtype();
  XLA_CHECK_LE(index.shape().get().rank(), 1)
      << "Fill index is supposed to be a vector";
  XLA_CHECK_EQ(value.shape().get().rank(), 0)
      << "Fill only supports a 0-dimensional value tensor";
  return IndexFillOp(base.GetIrValue(), dim, index.GetIrValue(),
                     value.GetIrValue());
}

ir::Value IndexAdd(const XLATensor& base, xla::int64_t dim,
                   const XLATensor& index, const XLATensor& source) {
  XLA_CHECK(index.dtype() == at::ScalarType::Long ||
            index.dtype() == at::ScalarType::Int)
      << "Add index is expected to be of scalar type Long or scalar type Int, "
         "but it is "
      << index.dtype();
  XLA_CHECK_LE(index.shape().get().rank(), 1)
      << "Add index is supposed to be a vector";
  return IndexAddOp(base.GetIrValue(), dim, index.GetIrValue(),
                    source.GetIrValue());
}

ir::Value IndexCopy(const XLATensor& base, xla::int64_t dim,
                    const XLATensor& index, const XLATensor& source) {
  XLA_CHECK_EQ(index.dtype(), at::ScalarType::Long)
      << "Copy index is expected to be of scalar type Long, but it is "
      << index.dtype();
  XLA_CHECK_LE(index.shape().get().rank(), 1)
      << "Copy index is supposed to be a vector";
  return IndexCopyOp(base.GetIrValue(), dim, index.GetIrValue(),
                     source.GetIrValue());
}

}  // namespace torch_xla
