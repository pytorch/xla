#include "torch/csrc/lazy/core/ir.h"
#include "torch/csrc/lazy/core/ir_builder.h"
#include "torch_xla/csrc/ops/as_strided.h"
#include "torch_xla/csrc/ops/cast.h"
#include "torch_xla/csrc/ops/diagonal.h"
#include "torch_xla/csrc/ops/expand.h"
#include "torch_xla/csrc/ops/generic.h"
#include "torch_xla/csrc/ops/ops.h"

namespace torch_xla {

struct XLAIrBuilder : IrBuilder {
  NodePtr MakeDeviceData(
      const std::shared_ptr<BackendData>& data) const override {
    return torch::lazy::MakeNode<DeviceData>(data);
  }

  NodePtr MakeScalar(const at::Scalar& value,
                     const at::ScalarType& type) const override {
    return torch::lazy::MakeNode<Scalar>(
        value, MakeXlaPrimitiveType(type, GetDefaultDevice()));
  }
  NodePtr MakeExpand(const Value& input0, const std::vector<int64_t>& size,
                     const bool& is_scalar_expand) const override {
    // TODO(JackCaoG): handle is_scalar_expand
    return torch::lazy::MakeNode<Expand>(input0, size);
  }
  NodePtr MakeView(const Value& input0,
                   const std::vector<int64_t>& output_size) const override {
    return torch::lazy::MakeNode<ViewOp>(input0, output_size);
  }
  NodePtr MakeCast(const Value& input0, const at::ScalarType& dtype,
                   const c10::optional<at::ScalarType>& stype =
                       c10::nullopt) const override {
    return torch::lazy::MakeNode<Cast>(input0, dtype, stype);
  }
  NodePtr MakeTensorList(const OpList& inputs) const override {
    // TODO(JackCaoG): implement tensorList IR. This is used by codegen.
    return nullptr;
  }
  // Generic needs cleanup
  NodePtr MakeGeneric(const OpKind& op, const OpList& operands,
                      const Shape& shape, const size_t& num_outputs = 1,
                      const hash_t& hash_seed =
                          static_cast<uint32_t>(0x5a2d296e9)) const override {
    return torch::lazy::MakeNode<Generic>(op, operands, shape, num_outputs,
                                          hash_seed);
  }

  // We should use functionization pass for view ops when migrating to the LTC.
  // View op nodes
  NodePtr MakeAsStridedViewUpdate(
      const Value& input0, const Value& input1,
      const std::vector<int64_t>& size, const std::vector<int64_t>& stride,
      const int64_t& storage_offset) const override {
    return nullptr;
  }
  NodePtr MakeAsStrided(const Value& input0, const std::vector<int64_t>& size,
                        const std::vector<int64_t>& stride,
                        const int64_t& storage_offset) const override {
    return nullptr;
  }
  NodePtr MakeDiagonalViewUpdate(const Value& input0, const Value& input1,
                                 const int64_t& offset, const int64_t& dim1,
                                 const int64_t& dim2) const override {
    return nullptr;
  }
  NodePtr MakeDiagonal(const Value& input0, const int64_t& offset,
                       const int64_t& dim1,
                       const int64_t& dim2) const override {
    return nullptr;
  }
  NodePtr MakeNarrowViewUpdate(
      const Value& input0, const Value& input1,
      const std::vector<int64_t>& base_indices) const override {
    return nullptr;
  }
  NodePtr MakeNarrow(const Value& input0,
                     const std::vector<int64_t>& base_indices,
                     const std::vector<int64_t>& sizes) const override {
    return nullptr;
  }
  NodePtr MakePermute(const Value& input0,
                      const std::vector<int64_t>& dims) const override {
    return nullptr;
  }
  NodePtr MakeResize(const Value& input0,
                     const std::vector<int64_t>& size) const override {
    return nullptr;
  }
  NodePtr MakeSelectViewUpdate(const Value& input0, const Value& input1,
                               const int64_t& dim, const int64_t& start,
                               const int64_t& end,
                               const int64_t& stride) const override {
    return nullptr;
  }
  NodePtr MakeSelect(const Value& input0, const int64_t& dim,
                     const int64_t& start, const int64_t& end,
                     const int64_t& stride) const override {
    return nullptr;
  }
  NodePtr MakeSqueeze(const Value& input0, const int& dim) const override {
    return nullptr;
  }
  NodePtr MakeUnsqueeze(const Value& input0, const int& dim) const override {
    return nullptr;
  }

  // dynamic ir nodes
  // TODO(JackCaoG): implement these when dynamic Node Ir merged.
  NodePtr MakeSizeNode(const Value& input, size_t dim) const override {
    return nullptr;
  }
  NodePtr MakeSizeAdd(const Value& a, const Value& b) const override {
    return nullptr;
  }
  NodePtr MakeSizeMul(const Value& a, const Value& b) const override {
    return nullptr;
  }
  NodePtr MakeSizeDiv(const Value& a, const Value& b) const override {
    return nullptr;
  }
};

}  // namespace torch_xla
