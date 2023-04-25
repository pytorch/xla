#ifndef XLA_TORCH_XLA_CSRC_TENSOR_H_
#define XLA_TORCH_XLA_CSRC_TENSOR_H_

#include <c10/core/SymNodeImpl.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/lazy/core/ir_util.h>

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/types.h"
#include "third_party/xla_client/async_task.h"
#include "third_party/xla_client/cache.h"
#include "third_party/xla_client/computation_client.h"
#include "third_party/xla_client/multi_wait.h"
#include "third_party/xla_client/util.h"
#include "torch_xla/csrc/computation.h"
#include "torch_xla/csrc/cross_replica_reduces.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ir_util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/torch_util.h"
#include "torch_xla/csrc/view.h"

namespace torch_xla {

enum class PyType {
  INT,    // int64; technically arbitrary precision
  FLOAT,  // double
  BOOL,   // bool
};

class TORCH_API XLASymNodeImpl final : public c10::SymNodeImpl {
 public:
  XLASymNodeImpl(torch::lazy::NodePtr ptr, PyType pytype)
      : node_(std::move(ptr)), pytype_(pytype) {}
  bool is_bool() override;
  bool is_int() override;
  bool is_float() override;
  c10::SymNode add(const c10::SymNode& other) override;
  c10::SymNode sub(const c10::SymNode& other) override;
  c10::SymNode mul(const c10::SymNode& other) override;
  c10::SymNode truediv(const c10::SymNode& other) override;
  c10::SymNode pow(const c10::SymNode& other) override;
  c10::SymNode floordiv(const c10::SymNode& other) override;
  c10::SymNode mod(const c10::SymNode& other) override;
  c10::SymNode eq(const c10::SymNode& other) override;
  c10::SymNode ne(const c10::SymNode& other) override;
  c10::SymNode gt(const c10::SymNode& other) override;
  c10::SymNode lt(const c10::SymNode& other) override;
  c10::SymNode le(const c10::SymNode& other) override;
  c10::SymNode ge(const c10::SymNode& other) override;
  c10::SymNode ceil() override;
  c10::SymNode floor() override;
  c10::SymNode neg() override;
  c10::SymNode sym_min(const c10::SymNode& other) override;
  c10::SymNode sym_max(const c10::SymNode& other) override;
  c10::SymNode sym_or(const c10::SymNode& other) override;
  c10::SymNode sym_and(const c10::SymNode& other) override;
  c10::SymNode sym_not() override;
  // NB: self is ignored here, only the arguments are used
  c10::SymNode is_contiguous(at::ArrayRef<c10::SymNode> sizes,
                             at::ArrayRef<c10::SymNode> strides) override;
  c10::SymNode is_channels_last_contiguous_2d(
      at::ArrayRef<c10::SymNode> sizes,
      at::ArrayRef<c10::SymNode> strides) override;
  c10::SymNode is_channels_last_contiguous_3d(
      at::ArrayRef<c10::SymNode> sizes,
      at::ArrayRef<c10::SymNode> strides) override;
  c10::SymNode is_channels_last_strides_2d(
      at::ArrayRef<c10::SymNode> sizes,
      at::ArrayRef<c10::SymNode> strides) override;
  c10::SymNode is_channels_last_strides_3d(
      at::ArrayRef<c10::SymNode> sizes,
      at::ArrayRef<c10::SymNode> strides) override;
  c10::SymNode is_non_overlapping_and_dense(
      at::ArrayRef<c10::SymNode> sizes,
      at::ArrayRef<c10::SymNode> strides) override;
  c10::SymNode clone() override;
  c10::SymNode sym_float() override;
  c10::SymNode wrap_int(int64_t num) override;
  c10::SymNode wrap_float(double num) override;
  c10::SymNode wrap_bool(bool num) override;
  int64_t guard_int(const char* file, int64_t line) override;
  double guard_float(const char* file, int64_t line) override;
  bool guard_bool(const char* file, int64_t line) override;
  int64_t int_() override;
  bool bool_() override;
  bool has_hint() override;
  std::string str() override;

  torch::lazy::NodePtr node() { return node_; }

 private:
  torch::lazy::NodePtr node_;
  PyType pytype_;
};

class XLATensor;
using XLATensorPtr = c10::intrusive_ptr<XLATensor>;

class XLATensor : public torch::lazy::LazyTensor {
 public:
  struct ShardingSpec;
  using ShardingSpecPtr = std::shared_ptr<ShardingSpec>;

  // This is the core XLA tensor data structure where all the tensor data is
  // held. The XLA tensor is nothing more than a shared pointer to a Data
  // object.
  struct Data : public torch::lazy::LazyTensor::Data {
    Data(torch::lazy::BackendDataPtr handle,
         const torch::lazy::BackendDevice& device,
         c10::optional<at::ScalarType> logical_element_type,
         ShardingSpecPtr sharding = nullptr)
        : torch::lazy::LazyTensor::Data(handle, device),
          logical_element_type(logical_element_type),
          sharding(sharding) {
      alias_id = unique_id;
    }
    Data(torch::lazy::Value ir_value, const torch::lazy::BackendDevice& device,
         c10::optional<at::ScalarType> logical_element_type,
         ShardingSpecPtr sharding = nullptr)
        : torch::lazy::LazyTensor::Data(ir_value, device),
          logical_element_type(logical_element_type),
          sharding(sharding) {
      alias_id = unique_id;
    }
    Data(at::Tensor tensor_data, const torch::lazy::BackendDevice& device,
         ShardingSpecPtr sharding = nullptr)
        : torch::lazy::LazyTensor::Data(tensor_data, device),
          logical_element_type(tensor_data.scalar_type()),
          sharding(sharding) {
      alias_id = unique_id;
    }
    Data(std::shared_ptr<View> view, const torch::lazy::BackendDevice& device,
         c10::optional<at::ScalarType> logical_element_type,
         ShardingSpecPtr sharding = nullptr)
        : torch::lazy::LazyTensor::Data(device),
          view(std::move(view)),
          logical_element_type(logical_element_type),
          sharding(sharding) {
      alias_id = unique_id;
    }

    ~Data();

    std::shared_ptr<View> view;
    // TODO: remove this in favor of torch::lazy::Shape within ir_value.
    c10::optional<at::ScalarType> logical_element_type;
    // The user provided sharding spec is attached to `XLATensor::Data`
    // and all sharding look-up should refer to it as source of truth.
    // A copy of the sharding spec is attached to the IR node via
    // `SetShardingSpec` and also during the sync tensor collection.
    ShardingSpecPtr sharding;
    // This is used to enable XLA's InputOutputAlias. It's inited
    // with unique_id, and then only get updated during the in-place
    // op funtionalize pass to point to the input.
    int64_t alias_id{0};
  };

  static XLATensorPtr Create(const at::Tensor& tensor,
                             const torch::lazy::BackendDevice& device);
  static XLATensorPtr Create(
      torch::lazy::BackendDataPtr handle,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static XLATensorPtr Create(
      torch::lazy::Value ir_value, const torch::lazy::BackendDevice& device,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static XLATensorPtr Create(std::shared_ptr<Data> data);

  // Create a new XLA tensor with the same metadata of the input tensor (with
  // possible overrides), and the new IR value.
  XLATensorPtr CreateFrom(torch::lazy::Value ir_value) const;
  XLATensorPtr CreateFrom(
      torch::lazy::Value ir_value,
      c10::optional<at::ScalarType> logical_element_type_opt) const;
  // TODO: We should remove this one once MaybeCastIrValue is no longer needed.
  XLATensorPtr CreateFrom(torch::lazy::Value ir_value,
                          const torch::lazy::BackendDevice& device,
                          at::ScalarType logical_element_type) const;

  // The default ctor previously created a null LazyTensor (one with no 'data'
  // obj). Creating a null XLATensor is no longer possible, since the same can
  // be achieved by creating a null LazyTensorPtr and it is way too confusing to
  // have to check both lazy_tensor_ptr && *lazy_tensor_ptr, so everywhere that
  // used to rely on a LazyTensor obj with a null Data can now rely on a null
  // LazyTensorPtr instead.
  XLATensor() = delete;

  // Override to use xla::shape.
  int64_t size(int64_t dim) const final;

  // Override to use XLAGraphExecutor.
  at::Tensor ToTensor(bool detached) final;

  // We don't use the upsteram ShallowCopyTo because of logical_element_type.
  void ShallowCopyTo(XLATensorPtr dest) const;

  // Assigns the tensor value to the XLA tensor.
  // TODO(alanwaketan): Reuse the upstream one once Functionalization is done.
  void SetTensor(at::Tensor tensor);

  // TODO(alanwaketan): Reuse the upstream ones once Functionalization is done.
  void UpdateFromTensor(at::Tensor tensor, bool sync);
  void UpdateFromTensorOut(at::Tensor tensor);
  void UpdateFromTensorOut(const XLATensorPtr& tensor);

  // Override to use logical_element_type.
  at::ScalarType dtype() const final;
  c10::optional<at::ScalarType> dtype_optional() const;

  // Set logical_element_type which is visible to upstream PyTorch.
  void SetScalarType(c10::optional<at::ScalarType> logical_element_type);

  // We don't use the upstream shape to provide xla::shape.
  xla::util::MaybeRef<xla::Shape> shape() const;

  // Retrieves an opaque ID of the alias object upon which the tensor's view is
  // rooted, or 0 if this tensor is not a view.
  std::ptrdiff_t GetViewAliasId() const;

  // Fetches the XLA data behind the tensor. If the tensor has a graph defining
  // its current value, executes the graph and fetches the XLA data result.
  // TODO(alanwaketan): Reuse the upstream ones once Functionalization is done.
  torch::lazy::BackendDataPtr GetXlaData();
  void SetXlaData(torch::lazy::BackendDataPtr handle);

  // Retrieves the current IR XlaNode, or nullptr in case no active IR XlaNode
  // is available.
  // TODO(alanwaketan): Reuse the upstream ones once Functionalization is done.
  torch::lazy::Value CurrentIrValue() const;

  // Retrieves the IR Node representing this XLATensor. One will be created if
  // missing. Note that although this is a const API, it actually changes the
  // internal state of the object.
  // TODO(alanwaketan): Reuse the upstream ones once Functionalization is done.
  torch::lazy::Value GetIrValue() const;
  void SetIrValue(torch::lazy::Value ir_value, bool inplace = true);
  void SetInPlaceIrValue(torch::lazy::Value ir_value);

  // TODO(alanwaketan): Reuse the upstream one once Functionalization is done.
  c10::optional<at::Tensor> CurrentTensorData() const;

  // We don't use the upstream MakeOutputTensors to return XLATensorPtr instead.
  std::vector<XLATensorPtr> MakeOutputTensors(
      torch::lazy::NodePtr node, bool inherit_logical_type = true) const;

  void SetSubView(ViewInfo view_info) const;
  void ModifyCurrentView(ViewInfo view_info) const;
  XLATensorPtr CreateViewTensor(ViewInfo view_info) const;

  // We don't use the upstream CopyTensorToDevice in order to return
  // XLATensorPtr.
  XLATensorPtr CopyTensorToDevice(const torch::lazy::BackendDevice& device);

  // Applies the queue of operations in preparation for using the data.
  // Override to use XLAGraphExecutor.
  void ApplyPendingGraph() final;

  // To be noted, this returns XLATensor::Data instead of
  // torch::lazy::LazyTensor::Data.
  const std::shared_ptr<Data>& data() const;

  // XLA SPMD sharding spec annoation. The XLA tensor uses this to create
  // HloSharding for replication, manual and tile shardings.
  struct ShardingSpec {
    ShardingSpec(const xla::OpSharding& sharding) : sharding(sharding) {}
    ShardingSpec(const xla::OpSharding& sharding, const xla::Shape& shape)
        : sharding(sharding), shape(shape) {}

    xla::OpSharding sharding;
    // Optional source tensor shape unpartitioned.
    std::optional<xla::Shape> shape;
  };

  // Annotate the IR value with ShardingSpec.
  void SetShardingSpec(const ShardingSpec& sharding_spec);
  // Clear sharding annotation attached to the IR value and transfer sharded
  // data back to host.
  void ClearShardingSpec();
  ShardingSpecPtr sharding_spec() const;

  void SetStorage(const c10::Storage& storage) { storage_ = storage; }
  const c10::Storage& Storage() const { return storage_; }

  int64_t GetOpaqueHandle() const;

  // Override to enable SPMD.
  void AssignIrValue(torch::lazy::Value ir_value) const final;

 private:
  XLATensor(const at::Tensor& tensor, const torch::lazy::BackendDevice& device);
  XLATensor(torch::lazy::BackendDataPtr handle,
            c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  XLATensor(torch::lazy::Value ir_value,
            const torch::lazy::BackendDevice& device,
            c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  XLATensor(std::shared_ptr<View> view,
            const torch::lazy::BackendDevice& device,
            c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  XLATensor(std::shared_ptr<Data> data);

  static XLATensorPtr Create(
      std::shared_ptr<View> view, const torch::lazy::BackendDevice& device,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  // TODO(alanwaketan): Reuse the upstream one once Functionalization is done.
  void SetXlaData(torch::lazy::BackendDataPtr handle, bool sync);

  View::IrNode GetViewUpdate(const std::shared_ptr<View>& view) const;

  std::shared_ptr<View> UpdateView(std::shared_ptr<View> view,
                                   torch::lazy::Value ir_value) const;

  std::shared_ptr<View> CreateView(ViewInfo view_info) const;

  torch::lazy::Value MaybeCastIrValue(
      torch::lazy::Value ir_value, const torch::lazy::BackendDevice& device,
      c10::optional<at::ScalarType> logical_element_type) const;

  // Override to instantiate our own xla data.
  torch::lazy::Value GetIrValueForTensor(
      const at::Tensor& tensor,
      const torch::lazy::BackendDevice& device) const final;

  static bool UseEagerDebugMode();

  bool ShouldSyncIrNode();

  // We store two shared_ptr of Data in a XLATensor.
  // One in the LazyTensor class as the LazyTensor::Data type
  // for base class method to access. One here as the derived
  // XLATensor::Data type such that it's easier to access XLA
  // extra fields.
  std::shared_ptr<Data> data_;
  // Temporarily used to suport Tensor.is_alias_of().
  // This is a fake storage that doesn't store anything.
  // Instead it serves as a marker to mark LazyTensors that
  // points to the same storage, and thus alias of each other.
  // FIXME(alanwaketan): Remove this once we have functionalization (bdhirsh).
  c10::Storage storage_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_TENSOR_H_