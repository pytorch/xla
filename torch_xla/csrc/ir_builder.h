#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch/csrc/lazy/core/ir.h"
#include "torch/csrc/lazy/core/ir_builder.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/ops/as_strided.h"
#include "torch_xla/csrc/ops/cast.h"
#include "torch_xla/csrc/ops/device_data.h"
#include "torch_xla/csrc/ops/diagonal.h"
#include "torch_xla/csrc/ops/dynamic_ir.h"
#include "torch_xla/csrc/ops/expand.h"
#include "torch_xla/csrc/ops/generic.h"
#include "torch_xla/csrc/ops/ops.h"
#include "torch_xla/csrc/ops/dynamic_ir.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {

struct XLAIrBuilder : torch::lazy::IrBuilder {
  torch::lazy::NodePtr MakeDeviceData(
      const std::shared_ptr<torch::lazy::BackendData>& data) const override {
    return torch::lazy::MakeNode<DeviceData>(data);
  }

  torch::lazy::NodePtr MakeScalar(const at::Scalar& value,
                                  const at::ScalarType& type) const override {
    return torch::lazy::MakeNode<Scalar>(
        value, MakeXlaPrimitiveType(type, GetDefaultDevice()));
  }
  torch::lazy::NodePtr MakeExpand(const torch::lazy::Value& input0,
                                  const std::vector<int64_t>& size,
                                  const bool& is_scalar_expand) const override {
    // TODO(JackCaoG): handle is_scalar_expand
    return torch::lazy::MakeNode<Expand>(input0, size);
  }
  torch::lazy::NodePtr MakeView(
      const torch::lazy::Value& input0,
      const std::vector<int64_t>& output_size) const override {
    // TODO(JackCAoG): use functionization pass instead
    return nullptr;
  }
  torch::lazy::NodePtr MakeCast(const torch::lazy::Value& input0,
                                const at::ScalarType& dtype,
                                const c10::optional<at::ScalarType>& stype =
                                    c10::nullopt) const override {
    return torch::lazy::MakeNode<Cast>(input0, dtype, stype);
  }
  torch::lazy::NodePtr MakeTensorList(
      const torch::lazy::OpList& inputs) const override {
    // TODO(JackCaoG): implement tensorList IR. This is used by codegen.
    XLA_ERROR() << "Need to implement";
    return nullptr;
  }
  // Generic needs cleanup
  torch::lazy::NodePtr MakeGeneric(
      const torch::lazy::OpKind& op, const torch::lazy::OpList& operands,
      const torch::lazy::Shape& shape, const size_t& num_outputs = 1,
      const torch::lazy::hash_t& hash_seed =
          static_cast<uint32_t>(0x5a2d296e9)) const override {
    // TODO(JackCaoG): ltc generic op does not take lowering function
    // return torch::lazy::MakeNode<Generic>(
    //     op, operands, MakeXlaShapeFromLazyShape(shape, *GetDefaultDevice()),
    //     num_outputs, hash_seed);
  }

  // We should use functionization pass for view ops when migrating to the LTC.
  // View op nodes
  torch::lazy::NodePtr MakeAsStridedViewUpdate(
      const torch::lazy::Value& input0, const torch::lazy::Value& input1,
      const std::vector<int64_t>& size, const std::vector<int64_t>& stride,
      const int64_t& storage_offset) const override {
    XLA_ERROR() << "Need to implement";
    return nullptr;
  }
  torch::lazy::NodePtr MakeAsStrided(
      const torch::lazy::Value& input0, const std::vector<int64_t>& size,
      const std::vector<int64_t>& stride,
      const int64_t& storage_offset) const override {
    XLA_ERROR() << "Need to implement";
    return nullptr;
  }
  torch::lazy::NodePtr MakeDiagonalViewUpdate(
      const torch::lazy::Value& input0, const torch::lazy::Value& input1,
      const int64_t& offset, const int64_t& dim1,
      const int64_t& dim2) const override {
    XLA_ERROR() << "Need to implement";
    return nullptr;
  }
  torch::lazy::NodePtr MakeDiagonal(const torch::lazy::Value& input0,
                                    const int64_t& offset, const int64_t& dim1,
                                    const int64_t& dim2) const override {
    XLA_ERROR() << "Need to implement";
    return nullptr;
  }
  torch::lazy::NodePtr MakeNarrowViewUpdate(
      const torch::lazy::Value& input0, const torch::lazy::Value& input1,
      const std::vector<int64_t>& base_indices) const override {
    XLA_ERROR() << "Need to implement";
    return nullptr;
  }
  torch::lazy::NodePtr MakeNarrow(
      const torch::lazy::Value& input0,
      const std::vector<int64_t>& base_indices,
      const std::vector<int64_t>& sizes) const override {
    XLA_ERROR() << "Need to implement";
    return nullptr;
  }
  torch::lazy::NodePtr MakePermute(
      const torch::lazy::Value& input0,
      const std::vector<int64_t>& dims) const override {
    XLA_ERROR() << "Need to implement";
    return nullptr;
  }
  torch::lazy::NodePtr MakeResize(
      const torch::lazy::Value& input0,
      const std::vector<int64_t>& size) const override {
    XLA_ERROR() << "Need to implement";
    return nullptr;
  }
  torch::lazy::NodePtr MakeSelectViewUpdate(
      const torch::lazy::Value& input0, const torch::lazy::Value& input1,
      const int64_t& dim, const int64_t& start, const int64_t& end,
      const int64_t& stride) const override {
    XLA_ERROR() << "Need to implement";
    return nullptr;
  }
  torch::lazy::NodePtr MakeSelect(const torch::lazy::Value& input0,
                                  const int64_t& dim, const int64_t& start,
                                  const int64_t& end,
                                  const int64_t& stride) const override {
    XLA_ERROR() << "Need to implement";
    return nullptr;
  }
  torch::lazy::NodePtr MakeSqueeze(const torch::lazy::Value& input0,
                                   const int& dim) const override {
    XLA_ERROR() << "Need to implement";
    return nullptr;
  }
  torch::lazy::NodePtr MakeUnsqueeze(const torch::lazy::Value& input0,
                                     const int& dim) const override {
    XLA_ERROR() << "Need to implement";
    return nullptr;
  }

  torch::lazy::NodePtr MakeSizeNode(const torch::lazy::Value& input,
                                    size_t dim) const override {
    return torch::lazy::MakeNode<SizeNode>(input, dim);
  }
  torch::lazy::NodePtr MakeSizeAdd(const torch::lazy::Value& a,
                                   const torch::lazy::Value& b) const override {
    return torch::lazy::MakeNode<SizeAdd>(a, b);
  }
  torch::lazy::NodePtr MakeSizeMul(const torch::lazy::Value& a,
                                   const torch::lazy::Value& b) const override {
    return torch::lazy::MakeNode<SizeMul>(a, b);
  }
  torch::lazy::NodePtr MakeSizeDiv(const torch::lazy::Value& a,
                                   const torch::lazy::Value& b) const override {
    return torch::lazy::MakeNode<SizeDiv>(a, b);
  }
};

}  // namespace torch_xla
