#include "torch_xla/csrc/ops/index_op.h"

#include <ATen/ExpandUtils.h>
#include <ATen/Functions.h>

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/arithmetic_ir_ops.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/ops.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace {

void CheckIndexTensorTypes(at::TensorList indices) {
  for (auto& tensor : indices) {
    if (tensor.defined()) {
      at::ScalarType scalar_type = tensor.scalar_type();
      if (scalar_type != at::kLong && scalar_type != at::kByte) {
        XLA_ERROR() << "Tensors used as indices must be long or byte tensors, "
                       "found scalar type: "
                    << scalar_type;
      }
    }
  }
}

// Expands byte tensors (masks) into the equivalent indexing by LongTensors.
// This is a version of at::native::expandByteTensors with style adjustments.
std::vector<at::Tensor> ExpandByteTensors(const at::Tensor& self,
                                          at::TensorList indices) {
  std::vector<at::Tensor> result;
  for (auto& index : indices) {
    if (index.type().scalarType() == at::kByte) {
      // The sizes of the ByteTensor mask must match the sizes of the
      // corresponding dimensions in self.
      for (int64_t j = 0; j < index.dim(); j++) {
        int64_t src_idx = result.size() + j;
        XLA_CHECK_EQ(index.size(j), self.size(src_idx))
            << "The shape of the mask " << index.sizes() << " at index " << j
            << " does not match the shape of the indexed tensor "
            << self.sizes() << " at index " << src_idx;
      }
      // Replace with nonzeros.
      auto nonzero = index.nonzero();
      for (int64_t j = 0; j < index.dim(); j++) {
        result.emplace_back(nonzero.select(1, j));
      }
    } else {
      result.emplace_back(index);
    }
  }
  return result;
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
  return {base.permute(dims), std::move(transposed_indices)};
}

// Wraps index tensors once into the [0, dim_size) interval, where dim_size is
// the size of the current indexed dimension.
std::vector<XLATensor> WrapIndicesOnce(
    const XLATensor& base,
    tensorflow::gtl::ArraySlice<const XLATensor> indices) {
  std::vector<XLATensor> canonical_indices;
  auto base_shape_ref = base.shape();
  XLA_CHECK_LE(indices.size(), base_shape_ref.get().rank());
  for (size_t dim_idx = 0; dim_idx < indices.size(); ++dim_idx) {
    const XLATensor& dim_index = indices[dim_idx];
    int64_t dim_size = base_shape_ref.get().dimensions(dim_idx);
    XLATensor wrapped_dim_index = XLATensor::Create(
        dim_index.GetIrValue() +
            ir::ops::ScalarOp(at::Scalar(dim_size), dim_index.shape()),
        base.GetDevice());
    XLATensor wrap_cond =
        XLATensor::lt(indices[dim_idx], at::Scalar(int64_t(0)));
    canonical_indices.push_back(
        XLATensor::where(wrap_cond, wrapped_dim_index, dim_index));
  }
  return canonical_indices;
}

ir::NodePtr IndexOp(const ir::Value& base, const ir::Value& indices) {
  auto lower_fn = [](const ir::Node& node,
                     ir::LoweringContext* loctx) -> ir::XlaOpVector {
    xla::XlaOp xla_input = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp xla_indices = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(CreateIndex(xla_input, xla_indices), loctx);
  };
  auto lower_for_shape_fn =
      [&](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp { return CreateIndex(operands[0], operands[1]); };
  return ir::ops::GenericOp(
      ir::OpKind(at::aten::index), {base, indices},
      ir::ops::InferOutputShape({base.shape(), indices.shape()},
                                lower_for_shape_fn),
      std::move(lower_fn));
}

}  // namespace

CanonicalIndexInfo GetCanonicalIndexInfo(const at::Tensor& base,
                                         at::TensorList orig_indices) {
  CheckIndexTensorTypes(orig_indices);
  // First expand ByteTensor (boolean masks) into 1 or more LongTensors, then
  // broadcast all index tensors together.
  auto indices = at::expand_outplace(ExpandByteTensors(base, orig_indices));
  // If the non-null indices are not all adjacent, transpose base and indices
  // together so that they're adjacent at the front.
  CanonicalIndexInfo canonical_index_info = TransposeToFront(base, indices);
  // Ensure indices are on the same device as the base.
  for (size_t i = 0; i < canonical_index_info.indices.size(); i++) {
    if (canonical_index_info.indices[i].device() != base.device()) {
      canonical_index_info.indices[i] =
          canonical_index_info.indices[i].to(base.device());
    }
  }
  return canonical_index_info;
}

XLATensor IndexByTensors(const XLATensor& base,
                         tensorflow::gtl::ArraySlice<const XLATensor> indices) {
  if (indices.empty()) {
    return base;
  }
  auto canonical_indices = WrapIndicesOnce(base, indices);
  xla::int64 indices_rank = canonical_indices.front().shape().get().rank();
  // Stack the indices to allow the whole multi-indexing to be dispatched with a
  // single gather.
  XLATensor indices_nd = XLATensor::stack(canonical_indices, indices_rank);
  return XLATensor::Create(IndexOp(base.GetIrValue(), indices_nd.GetIrValue()),
                           base.GetDevice());
}

}  // namespace torch_xla
