#include "torch_xla/csrc/tensor.h"

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/ir_util.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/metrics.h>
#include <torch/csrc/lazy/core/tensor_util.h>
#include <torch/csrc/lazy/core/util.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <set>
#include <stdexcept>
#include <unordered_set>

#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/debug_util.h"
#include "torch_xla/csrc/dtype.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/layout_manager.h"
#include "torch_xla/csrc/ops/arithmetic_ir_ops.h"
#include "torch_xla/csrc/ops/cast.h"
#include "torch_xla/csrc/ops/custom_sharding.h"
#include "torch_xla/csrc/ops/device_data.h"
#include "torch_xla/csrc/ops/dynamic_ir.h"
#include "torch_xla/csrc/ops/expand.h"
#include "torch_xla/csrc/ops/ops.h"
#include "torch_xla/csrc/ops/view.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/runtime/cache.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/pjrt_computation_client.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/runtime/xla_util.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"
#include "torch_xla/csrc/xla_graph_executor.h"
#include "torch_xla/csrc/xla_sharding_util.h"
#include "tsl/platform/errors.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/shape_util.h"

namespace torch_xla {

namespace {
bool CanApplySharding(const XLATensor::ShardingSpecPtr sharding) {
  return !sharding ||
         sharding->sharding.type() == xla::OpSharding::REPLICATED ||
         sharding->sharding.type() == xla::OpSharding::UNKNOWN;
}
}  // namespace

XLATensor::Data::~Data() { XLAGraphExecutor::Get()->UnregisterTensor(this); }

XLATensorPtr XLATensor::Create(const at::Tensor& tensor,
                               const torch::lazy::BackendDevice& device) {
  XLA_CHECK_EQ(tensor.device().type(), at::kCPU);
  XLATensorPtr xtensor =
      c10::make_intrusive<XLATensor>(XLATensor(tensor, device));
  XLAGraphExecutor::Get()->RegisterTensor(xtensor->data());
  return xtensor;
}

XLATensorPtr XLATensor::Create(
    torch::lazy::BackendDataPtr handle,
    std::optional<at::ScalarType> logical_element_type) {
  XLATensorPtr xtensor = c10::make_intrusive<XLATensor>(
      XLATensor(std::move(handle), logical_element_type));
  XLAGraphExecutor::Get()->RegisterTensor(xtensor->data());
  return xtensor;
}

XLATensorPtr XLATensor::Create(
    torch::lazy::Value ir_value, const torch::lazy::BackendDevice& device,
    std::optional<at::ScalarType> logical_element_type,
    bool delay_eager_executation) {
  XLATensorPtr xtensor = c10::make_intrusive<XLATensor>(
      XLATensor(std::move(ir_value), device, logical_element_type));
  XLAGraphExecutor* graph_executor = XLAGraphExecutor::Get();
  graph_executor->RegisterTensor(xtensor->data());
  if (graph_executor->UseEagerMode() && !delay_eager_executation) {
    std::vector<XLATensorPtr> xtensors({xtensor});
    graph_executor->ApplyEagerSync(xtensors);
  }
  return xtensor;
}

XLATensorPtr XLATensor::Create(
    std::shared_ptr<View> view, const torch::lazy::BackendDevice& device,
    std::optional<at::ScalarType> logical_element_type) {
  XLATensorPtr xtensor = c10::make_intrusive<XLATensor>(
      XLATensor(std::move(view), device, logical_element_type));
  XLAGraphExecutor::Get()->RegisterTensor(xtensor->data());
  return xtensor;
}

XLATensorPtr XLATensor::Create(std::shared_ptr<Data> data) {
  return c10::make_intrusive<XLATensor>(XLATensor(std::move(data)));
}

XLATensor::XLATensor(const at::Tensor& tensor,
                     const torch::lazy::BackendDevice& device)
    : XLATensor(std::make_shared<Data>(tensor, device)) {}

XLATensor::XLATensor(torch::lazy::BackendDataPtr handle,
                     std::optional<at::ScalarType> logical_element_type)
    : XLATensor(std::make_shared<Data>(handle, handle->device(),
                                       logical_element_type)) {
  // if data is sharded we need to carry the sharding spec over.
  runtime::ComputationClient::DataPtr data =
      std::dynamic_pointer_cast<runtime::ComputationClient::Data>(handle);
  if (data->HasSharding()) {
    ShardingSpec sharding_spec(data->GetSharding(), data->shape());
    SetShardingSpec(sharding_spec);
  }
}

XLATensor::XLATensor(torch::lazy::Value ir_value,
                     const torch::lazy::BackendDevice& device,
                     std::optional<at::ScalarType> logical_element_type)
    : XLATensor(std::make_shared<Data>(std::move(ir_value), device,
                                       logical_element_type)) {
  // Preserve sharding if a new tensor is created from a sharded IR node.
  if (CurrentIrValue()) {
    auto* xla_node = dynamic_cast<XlaNode*>(CurrentIrValue().node.get());
    if (xla_node->GetSharding(CurrentIrValue().index)) {
      ShardingSpec sharding =
          ShardingSpec{*xla_node->GetSharding(CurrentIrValue().index),
                       xla_node->xla_shape()};
      SetShardingSpec(sharding);
    }
  }
}

XLATensor::XLATensor(std::shared_ptr<View> view,
                     const torch::lazy::BackendDevice& device,
                     std::optional<at::ScalarType> logical_element_type)
    : XLATensor(std::make_shared<Data>(std::move(view), device,
                                       logical_element_type)) {}

XLATensor::XLATensor(std::shared_ptr<Data> data)
    : torch::lazy::LazyTensor(data),
      data_(std::move(data)),
      storage_(c10::Storage(
          {}, 0,
          c10::DataPtr(nullptr, bridge::XlaDeviceToAtenDevice(data_->device)))),
      base_() {}

auto XLATensor::data() const -> const std::shared_ptr<Data>& {
  XLA_CHECK(data_ != nullptr) << "Trying to access a null cursor";
  return data_;
}

int64_t XLATensor::size(int64_t dim) const {
  auto xla_shape = shape();
  int rank = xla_shape.get().rank();
  int dim_index = torch::lazy::GetCanonicalDimensionIndex(dim, rank);
  return xla_shape.get().dimensions(dim_index);
}

at::ScalarType XLATensor::dtype() const {
  return data()->logical_element_type
             ? *data()->logical_element_type
             : MaybeUpcastToHostTorchType(shape().get().element_type());
}

std::optional<at::ScalarType> XLATensor::dtype_optional() const {
  return data()->logical_element_type;
}

runtime::util::MaybeRef<xla::Shape> XLATensor::shape() const {
  if (data()->view != nullptr) {
    return data()->view->shape();
  }
  if (data()->handle != nullptr) {
    return std::dynamic_pointer_cast<runtime::ComputationClient::Data>(
               data()->handle)
        ->shape();
  }
  if (data()->ir_value) {
    return GetXlaShape(data()->ir_value);
  }
  XLA_CHECK(data()->tensor_data);
  const torch::lazy::BackendDevice& device = GetDevice();
  return xla::ShapeUtil::MakeShape(
      MakeXlaPrimitiveType(data()->tensor_data->type().scalarType(), &device),
      XlaHelpers::I64List(data()->tensor_data->sizes()));
}

std::ptrdiff_t XLATensor::GetViewAliasId() const {
  return data()->view != nullptr
             ? reinterpret_cast<std::ptrdiff_t>(data()->view->alias().get())
             : 0;
}

torch::lazy::BackendDataPtr XLATensor::GetXlaData() {
  // XLA data can coexist with a view, but we need to check that the view did
  // not receive any updates before calling the current XLA valid.
  bool up_to_date = true;
  torch::lazy::Value ir_value;
  if (data()->view != nullptr) {
    View::IrNode ir_value_updated = GetViewUpdate(data()->view);
    up_to_date = !ir_value_updated.updated;
    ir_value = std::move(ir_value_updated.ir_value);
  }
  if (up_to_date) {
    torch::lazy::BackendDataPtr handle = CurrentDataHandle();
    if (handle != nullptr) {
      XLA_CHECK(handle->HasValue())
          << "Trying to access XLA data for tensor with ID " << GetUniqueId()
          << " while an async operation is in flight: " << handle->shape();
      return handle;
    }
  }
  if (ir_value) {
    // The view gave us an updated IR value. We usually do not have a valid IR
    // value field together with a view, but to allow code reuse in
    // ApplyPendingGraph() we temporarily set it here. The following call to
    // ApplyPendingGraph() will clear it.
    AssignIrValue(std::move(ir_value));
  }
  if (data()->ir_value) {
    torch::lazy::BackendDataPtr node_data =
        torch::lazy::getBackend()->GetComputationDataFromNode(
            data()->ir_value.node.get());
    // Current IR is a DeviceData Node, we can retrive the data handle directly
    // instead of triggering an additional execution.
    if (node_data) {
      data()->ir_value = torch::lazy::Value();
      data()->handle = node_data;
    } else {
      ApplyPendingGraph();
    }
  } else {
    XLA_CHECK(data()->tensor_data);
    data()->handle = TensorToXlaData(*data()->tensor_data, GetDevice());
  }
  return data()->handle;
}

void XLATensor::SetShardingSpec(const ShardingSpec& sharding, bool overwrite) {
  // Existing annotation must be cleared explicitly. We do not clear and
  // overwrite the existing sharding on the user's behalf. This is a no-op if
  // the same sharding already applied.
  ShardingSpecPtr current_sharding = sharding_spec();
  if (!current_sharding || overwrite ||
      current_sharding->sharding.type() == xla::OpSharding::REPLICATED ||
      current_sharding->sharding.type() == xla::OpSharding::UNKNOWN) {
    TORCH_LAZY_COUNTER("SetShardingSpec", 1);
    data()->sharding = std::make_shared<ShardingSpec>(sharding);
  } else {
    // Tensor is already sharding annotated, check if it is UNKNOWN or
    // the same sharding type.
    XLA_CHECK(ShardingUtil::EqualShardingSpecs(sharding, *sharding_spec()))
        << "Existing sharding annotation, "
        << current_sharding->sharding.DebugString()
        << ", must be cleared before applying a new one, "
        << sharding.sharding.DebugString();
  }
  // Sync to the node.
  dynamic_cast<XlaNode*>(GetIrValue().node.get())
      ->SetSharding(sharding_spec()->sharding, GetIrValue().index);
}
void XLATensor::ClearShardingSpec() {
  data()->sharding = nullptr;
  torch::lazy::Value ir_value = CurrentIrValue();
  if (ir_value) {
    // This should be a no-op if there is no sharding.
    dynamic_cast<XlaNode*>(ir_value.node.get())->ClearSharding();
  }
}

XLATensor::ShardingSpecPtr XLATensor::sharding_spec() const {
  ShardingSpecPtr sharding = data()->sharding;
  torch::lazy::Value ir_value = CurrentIrValue();
  if (ir_value) {
    auto* xla_node = dynamic_cast<XlaNode*>(ir_value.node.get());
    const auto* new_op_sharding = xla_node->GetSharding(ir_value.index).get();
    if (new_op_sharding &&
        (new_op_sharding->type() != xla::OpSharding::UNKNOWN)) {
      // Re-sync the sharding annotation from the node to the tensor if there is
      // one attached to the node. A new sharding annotation is attached
      // directly to the node, and gets synced to the tensor after this.
      // If sharding is attached via SetShardingSpec, then it flows from the
      // tensor to the node. If sharding is attached by the compiler pass, then
      // it first gets attached to the graph node, and then synced to the tensor
      // here.
      if (!sharding ||
          (sharding && !ShardingUtil::EqualOpShardings(*new_op_sharding,
                                                       sharding->sharding))) {
        TF_VLOG(5) << "Syncing node sharding (type=" << new_op_sharding->type()
                   << ") to tensor (shape=" << xla_node->xla_shape().ToString()
                   << ").";
        data()->sharding = std::make_shared<ShardingSpec>(
            *new_op_sharding, xla_node->xla_shape());
      }
    } else if (sharding) {
      // There is a case where the sharding spec on the tensor is not
      // propagated down to the node after a reset.
      xla_node->SetSharding(sharding->sharding, ir_value.index);
    }
  }
  return sharding;
}

void XLATensor::SetXlaData(torch::lazy::BackendDataPtr handle) {
  SetXlaData(std::move(handle), /*sync=*/true);
}

void XLATensor::SetXlaData(torch::lazy::BackendDataPtr handle, bool sync) {
  data()->handle = std::move(handle);
  // Assigning a device data should always clear the IR node, to allow graph
  // trimming. A view cannot be reset though, unless we are at a step-end
  // sync.
  AssignIrValue(torch::lazy::Value());
  if (sync) {
    data()->view = nullptr;
    data()->tensor_data = std::nullopt;
  }
  data()->is_cloned = false;
}

void XLATensor::SetIrValue(torch::lazy::Value ir_value, bool inplace,
                           bool delay_eager_executation) {
  data()->handle = nullptr;
  data()->tensor_data = std::nullopt;
  if (data()->view != nullptr && inplace) {
    // If we have an active view, SetIrValue() happens, and we are
    // within an in-place execution context, we need to update the view's
    // alias as well.
    data()->view = UpdateView(data()->view, std::move(ir_value));
    data()->generation += 1;
  } else {
    // Reset the view if we are not within an in-place execution context
    data()->view = nullptr;
    data()->generation = 1;
    AssignIrValue(std::move(ir_value));
  }
  data()->is_cloned = false;

  XLAGraphExecutor* graph_executor = XLAGraphExecutor::Get();
  // Update should also be triggered eagerly if configured
  if (graph_executor->UseEagerMode() && !delay_eager_executation &&
      ShouldSyncIrNode()) {
    std::vector<XLATensorPtr> xtensors({c10::make_intrusive<XLATensor>(*this)});
    graph_executor->ApplyEagerSync(xtensors);
  }
}

void XLATensor::SetInPlaceIrValue(torch::lazy::Value ir_value,
                                  bool delay_eager_executation) {
  auto xla_shape = shape();
  if (xla_shape.get().element_type() != GetXlaShape(ir_value).element_type()) {
    ir_value =
        torch_xla::MakeNode<Cast>(ir_value, xla_shape.get().element_type());
  }
  SetIrValue(std::move(ir_value), /*inplace=*/true, delay_eager_executation);
}

void XLATensor::AssignIrValue(torch::lazy::Value ir_value) const {
  TF_VLOG(6) << "Assign IR value: "
             << (ir_value ? ir_value->ToString() : "empty node");
  data()->ir_value = std::move(ir_value);
  data()->generation += 1;
  data()->is_cloned = false;
}

torch::lazy::Value XLATensor::GetIrValue() const {
  torch::lazy::Value ir_value = CurrentIrValue();
  if (ir_value) {
    return ir_value;
  }
  torch::lazy::BackendDataPtr handle = CurrentDataHandle();
  if (handle != nullptr) {
    // In case of tensor node, we do not clear the XLA data when we set the IR
    // node. This because we want further calls to GetIrValue() to fetch the
    // same IR node, and not create new ones (even though the lowering context
    // will still collapse them all into a single XLA parameter op). So call
    // which wants the XLA data will still find it, w/out having to fetch it
    // via a computation client from-server call.
    auto* data_info =
        static_cast<torch::lazy::LazyGraphExecutor::DeviceDataInfo*>(
            handle->info());
    bool read_only = data_info != nullptr && data_info->read_only;
    AssignIrValue(CreateTensorNode(handle, read_only));
    // CreateTensorNode will set the data info of the tensor to the current
    // unique_id. Here the alias id needs to be updated so that input output
    // alias can correctly work on the xla's custom inplace operation.
    data()->alias_id = GetUniqueId();
    return data()->ir_value;
  }
  std::optional<at::Tensor> tensor_data = CurrentTensorData();
  XLA_CHECK(tensor_data);
  AssignIrValue(GetIrValueForTensor(*tensor_data, GetDevice()));
  data()->tensor_data = std::nullopt;
  return data()->ir_value;
}

torch::lazy::Value XLATensor::CurrentIrValue() const {
  if (data()->view != nullptr) {
    return GetViewUpdate(data()->view).ir_value;
  }
  return data()->ir_value;
}

std::optional<at::Tensor> XLATensor::CurrentTensorData() const {
  if (data()->view != nullptr && !data()->view->IsUpToDate()) {
    return std::nullopt;
  }
  return data()->tensor_data;
}

torch::lazy::Value XLATensor::GetIrValueForTensor(
    const at::Tensor& tensor, const torch::lazy::BackendDevice& device) const {
  torch::lazy::BackendDataPtr data;
  bool read_only = false;
  if (tensor.dim() == 0 && tensor.numel() == 1) {
    at::Scalar value = tensor.item();
    if (torch::lazy::IsSpecialScalar(value)) {
      return ScalarOp(std::move(value),
                      MakeXlaPrimitiveType(tensor.scalar_type(), &device));
    }
    data = XLAGraphExecutor::Get()->GetDeviceData(tensor.cpu(), device);
    read_only = true;
  } else {
    TORCH_LAZY_TIMED("IrValueTensorToXlaData");
    data = TensorToXlaData(tensor, device);
  }
  return CreateTensorNode(std::move(data), read_only);
}

View::IrNode XLATensor::GetViewUpdate(const std::shared_ptr<View>& view) const {
  View::IrNode ir_value_updated = view->GetViewIrNode();
  if (ir_value_updated.updated) {
    data()->handle = nullptr;
    data()->tensor_data = std::nullopt;
  }
  return ir_value_updated;
}

std::shared_ptr<View> XLATensor::UpdateView(std::shared_ptr<View> view,
                                            torch::lazy::Value ir_value) const {
  if (GetXlaShape(ir_value).dimensions() != view->shape().dimensions()) {
    XLA_CHECK_EQ(
        runtime::util::Multiply<int64_t>(GetXlaShape(ir_value).dimensions()),
        runtime::util::Multiply<int64_t>(view->shape().dimensions()));

    ViewInfo view_info(ViewInfo::Type::kReshape, GetXlaShape(ir_value),
                       view->shape());
    view = view->CreateSubView(view_info.shape, view_info);
  }
  view->Update(std::move(ir_value));
  return view;
}

void XLATensor::SetSubView(ViewInfo view_info) const {
  data()->view = data()->view->CreateSubView(view_info.shape, view_info);
  data()->generation += 1;
}

void XLATensor::ModifyCurrentView(ViewInfo view_info) const {
  if (data()->view != nullptr) {
    SetSubView(view_info);
    return;
  }
  // This node is not a view. Since this function is meant to modify a view
  // in place, we need to turn this existing tensor into a view.
  torch::lazy::Value ir_value = GetIrValue();
  std::shared_ptr<Alias> alias = std::make_shared<Alias>(ir_value);
  data()->view =
      std::make_shared<View>(view_info.shape, alias, std::move(view_info));
  AssignIrValue(torch::lazy::Value());
}

std::shared_ptr<View> XLATensor::CreateView(ViewInfo view_info) const {
  if (data()->view != nullptr) {
    return data()->view->CreateSubView(view_info.shape, view_info);
  }
  // This node is not a view, and creating a view forks the current node into
  // becoming one itself. This means creating an alias with the current IR
  // XlaNode, and using the same alias for the created IR XlaNode.
  torch::lazy::Value ir_value = GetIrValue();
  std::shared_ptr<Alias> alias = std::make_shared<Alias>(ir_value);
  ViewInfo this_view_info(ViewInfo::Type::kNoOp, GetXlaShape(ir_value),
                          GetXlaShape(ir_value));
  data()->view = std::make_shared<View>(GetXlaShape(ir_value), alias,
                                        std::move(this_view_info));
  AssignIrValue(torch::lazy::Value());
  return std::make_shared<View>(view_info.shape, alias, view_info);
}

XLATensorPtr XLATensor::CreateViewTensor(ViewInfo view_info) const {
  auto new_tensor =
      Create(CreateView(std::move(view_info)), GetDevice(), dtype_optional());
  new_tensor->storage_ = Storage();
  return new_tensor;
}

at::Tensor XLATensor::ToTensor(bool detached) {
  at::Tensor tensor;
  std::optional<at::Tensor> tensor_data = CurrentTensorData();
  if (!tensor_data) {
    XLAGraphExecutor::Get()->DeviceBarrier(GetDevice());
    // The GetXlaData() call will trigger an ApplyPendingGraph() if an IR
    // XlaNode is available on the tensor.
    std::vector<at::Tensor> tensors =
        XlaDataToTensors({GetXlaData()}, {dtype()});
    tensor = std::move(tensors.front());
    if (!detached) {
      SetTensorData(tensor);
    }
  } else {
    tensor = *tensor_data;
    if (detached) {
      if (data()->ir_value || data()->handle != nullptr ||
          data()->view != nullptr) {
        // If we have other authoritive sources, just drop our reference and
        // transfer it to the caller.
        data()->tensor_data = std::nullopt;
      } else {
        // Otherwise we need to make a copy to prevent the caller changing our
        // version.
        tensor = torch::lazy::CopyTensor(tensor);
      }
    }
  }
  return tensor;
}

void XLATensor::ShallowCopyTo(XLATensorPtr dest) const {
  dest->SetScalarType(data()->logical_element_type);
  dest->SetIrValue(GetIrValue(), /*inplace=*/false);
  if (sharding_spec() != nullptr) {
    dest->SetShardingSpec(*sharding_spec());
  }
}

void XLATensor::SetScalarType(
    std::optional<at::ScalarType> logical_element_type) {
  data()->logical_element_type = logical_element_type;
}

void XLATensor::SetTensor(at::Tensor tensor) {
  SetTensorData(tensor);
  data()->view = nullptr;
  data()->handle = nullptr;
  AssignIrValue(torch::lazy::Value());
}

void XLATensor::UpdateFromTensor(at::Tensor tensor, bool sync) {
  torch::lazy::BackendDevice device = GetDevice();
  if (sync) {
    at::Tensor typed_tensor =
        torch::lazy::CopyTensor(tensor, dtype(), /*copy=*/false);
    SetIrValue(GetIrValueForTensor(typed_tensor, device),
               /*inplace=*/true);
  } else {
    at::Tensor coyped_tensor = torch::lazy::CopyTensor(tensor, dtype());
    SetTensorData(coyped_tensor);
    data()->handle = nullptr;
    // if shape is different,
    if (data()->sharding) {
      auto coyped_tensor_dims = XlaHelpers::I64List(coyped_tensor.sizes());
      auto sharding_dims = data()->sharding->shape.dimensions();
      if (coyped_tensor_dims != sharding_dims) {
        // sharding shape from origional tensor is different from the new cpu
        // tensor, we need to clear the sharding here.
        ClearShardingSpec();
      }
    }
    // ClearShardingSpec();
    AssignIrValue(torch::lazy::Value());
    if (data()->view != nullptr) {
      torch::lazy::Value ir_value = GetIrValueForTensor(coyped_tensor, device);
      data()->view = UpdateView(data()->view, std::move(ir_value));
    }
  }
}

void XLATensor::UpdateFromTensorOut(at::Tensor tensor) {
  if (data()->view != nullptr &&
      xla::ShapeUtil::ElementsIn(shape()) != tensor.numel()) {
    data()->view = nullptr;
  }
  UpdateFromTensor(std::move(tensor), /*sync=*/false);
}

void XLATensor::UpdateFromTensorOut(const XLATensorPtr& tensor) {
  if (data()->view != nullptr &&
      xla::ShapeUtil::ElementsIn(shape()) !=
          xla::ShapeUtil::ElementsIn(tensor->shape())) {
    data()->view = nullptr;
  }
  SetIrValue(tensor->GetIrValue(), /*inplace=*/true);
}

std::vector<XLATensorPtr> XLATensor::MakeOutputTensors(
    torch::lazy::NodePtr node, bool inherit_logical_type) const {
  std::vector<XLATensorPtr> tensors;
  tensors.reserve(node->num_outputs());
  for (size_t i = 0; i < node->num_outputs(); ++i) {
    if (inherit_logical_type) {
      tensors.push_back(CreateFrom(torch::lazy::Value(node, i)));
    } else {
      tensors.push_back(CreateFrom(torch::lazy::Value(node, i),
                                   /*logical_element_type=*/std::nullopt));
    }
  }
  return tensors;
}

XLATensorPtr XLATensor::CopyTensorToDevice(
    const torch::lazy::BackendDevice& device) {
  // TODO: This can be optimized via proper XRT/XLA computation.
  return Create(ToTensor(/*detached=*/true), device);
}

torch::lazy::Value XLATensor::MaybeCastIrValue(
    torch::lazy::Value ir_value, const torch::lazy::BackendDevice& device,
    std::optional<at::ScalarType> logical_element_type) const {
  if (!logical_element_type) {
    logical_element_type = dtype_optional();
  }
  if (logical_element_type &&
      RequiresRawTypeCasting(*logical_element_type, &device)) {
    ir_value = torch_xla::MakeNode<Cast>(ir_value, *logical_element_type);
  }
  return ir_value;
}

XLATensorPtr XLATensor::CreateFrom(torch::lazy::Value ir_value,
                                   bool delay_eager_executation) const {
  ir_value = MaybeCastIrValue(std::move(ir_value), GetDevice(),
                              /*logical_element_type=*/std::nullopt);
  return Create(std::move(ir_value), GetDevice(), dtype_optional(),
                delay_eager_executation);
}

XLATensorPtr XLATensor::CreateFrom(
    torch::lazy::Value ir_value,
    std::optional<at::ScalarType> logical_element_type_opt,
    bool delay_eager_executation) const {
  ir_value = MaybeCastIrValue(std::move(ir_value), GetDevice(),
                              logical_element_type_opt);
  return Create(std::move(ir_value), GetDevice(), logical_element_type_opt,
                delay_eager_executation);
}

XLATensorPtr XLATensor::CreateFrom(torch::lazy::Value ir_value,
                                   const torch::lazy::BackendDevice& device,
                                   at::ScalarType logical_element_type,
                                   bool delay_eager_executation) const {
  ir_value =
      MaybeCastIrValue(std::move(ir_value), device, logical_element_type);
  return Create(std::move(ir_value), device, logical_element_type,
                delay_eager_executation);
}

void XLATensor::ApplyPendingGraph() {
  XLAGraphExecutor::Get()->DeviceBarrier(GetDevice());
  // This method is called to ensure that the tensor data is available on
  // device, so that a call to CurrentDataHandle() returns a valid pointer.
  if (CurrentDataHandle() == nullptr) {
    std::vector<XLATensorPtr> tensors({c10::make_intrusive<XLATensor>(*this)});
    XLAGraphExecutor::Get()->SyncTensorsGraph(&tensors, {}, /*wait=*/true,
                                              /*sync_xla_data=*/false);
  }
}

bool XLATensor::ShouldSyncIrNode() {
  if (!this->data()->ir_value) {
    return false;
  }
  return this->data()->ir_value->op() != xla_device_data;
}

bool XLASymNodeImpl::is_bool() { return pytype_ == PyType::BOOL; }

bool XLASymNodeImpl::is_int() { return pytype_ == PyType::INT; }

bool XLASymNodeImpl::is_float() { return pytype_ == PyType::FLOAT; }

c10::SymNode XLASymNodeImpl::add(const c10::SymNode& other) {
  auto p_other = dynamic_cast<XLASymNodeImpl*>(other.get());
  XLA_CHECK(is_int()) << __FUNCTION__ << " with non-int NYI";
  XLA_CHECK(p_other->is_int()) << __FUNCTION__ << " with non-int NYI";
  auto n_add = torch_xla::MakeNode<SizeAdd>(node(), p_other->node());
  return c10::make_intrusive<XLASymNodeImpl>(n_add, PyType::INT);
}

c10::SymNode XLASymNodeImpl::sub(const c10::SymNode& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::size_");

  torch_xla::XLASymNodeImpl* p_other =
      dynamic_cast<XLASymNodeImpl*>(other.get());
  XLA_CHECK(is_int()) << __FUNCTION__ << " with non-int NYI";
  XLA_CHECK(p_other->is_int()) << __FUNCTION__ << " with non-int NYI";
  torch::lazy::NodePtr n_sub =
      torch_xla::MakeNode<SizeSub>(node(), p_other->node());
  return c10::make_intrusive<XLASymNodeImpl>(n_sub, PyType::INT);
}

c10::SymNode XLASymNodeImpl::mul(const c10::SymNode& other) {
  auto p_other = dynamic_cast<XLASymNodeImpl*>(other.get());
  XLA_CHECK(is_int()) << __FUNCTION__ << " with non-int NYI";
  XLA_CHECK(p_other->is_int()) << __FUNCTION__ << " with non-int NYI";
  auto n_mul = torch_xla::MakeNode<torch_xla::SizeMul>(node(), p_other->node());
  return c10::make_intrusive<XLASymNodeImpl>(n_mul, PyType::INT);
}

c10::SymNode XLASymNodeImpl::truediv(const c10::SymNode& other) {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
}

c10::SymNode XLASymNodeImpl::pow(const c10::SymNode& other) {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
}

c10::SymNode XLASymNodeImpl::floordiv(const c10::SymNode& other) {
  auto p_other = dynamic_cast<XLASymNodeImpl*>(other.get());
  XLA_CHECK(is_int()) << __FUNCTION__ << " with non-int NYI";
  XLA_CHECK(p_other->is_int()) << __FUNCTION__ << " with non-int NYI";
  auto n_div = torch_xla::MakeNode<SizeDiv>(node(), p_other->node());
  return c10::make_intrusive<XLASymNodeImpl>(n_div, PyType::INT);
}

c10::SymNode XLASymNodeImpl::mod(const c10::SymNode& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::size_");
  torch_xla::XLASymNodeImpl* p_other =
      dynamic_cast<XLASymNodeImpl*>(other.get());
  XLA_CHECK(is_int()) << __FUNCTION__ << " with non-int NYI";
  XLA_CHECK(p_other->is_int()) << __FUNCTION__ << " with non-int NYI";
  torch::lazy::NodePtr n_mod =
      torch_xla::MakeNode<SizeMod>(node(), p_other->node());
  return c10::make_intrusive<XLASymNodeImpl>(n_mod, PyType::INT);
}

c10::SymNode XLASymNodeImpl::eq(const c10::SymNode& other) {
  auto p_other = dynamic_cast<XLASymNodeImpl*>(other.get());
  XLA_CHECK(is_int()) << __FUNCTION__ << " with non-int NYI";
  XLA_CHECK(p_other->is_int()) << __FUNCTION__ << " with non-int NYI";
  auto n_eq = torch_xla::MakeNode<SizeEq>(node(), p_other->node());
  return c10::make_intrusive<XLASymNodeImpl>(n_eq, PyType::BOOL);
}

c10::SymNode XLASymNodeImpl::ne(const c10::SymNode& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::size_");
  auto p_other = dynamic_cast<XLASymNodeImpl*>(other.get());
  XLA_CHECK(is_int()) << __FUNCTION__ << " with non-int NYI";
  XLA_CHECK(p_other->is_int()) << __FUNCTION__ << " with non-int NYI";
  auto n_ne = torch_xla::MakeNode<SizeNe>(node(), p_other->node());
  return c10::make_intrusive<XLASymNodeImpl>(n_ne, PyType::BOOL);
}

c10::SymNode XLASymNodeImpl::gt(const c10::SymNode& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::size_");
  auto p_other = dynamic_cast<XLASymNodeImpl*>(other.get());
  XLA_CHECK(is_int()) << __FUNCTION__ << " with non-int NYI";
  XLA_CHECK(p_other->is_int()) << __FUNCTION__ << " with non-int NYI";
  auto n_gt = torch_xla::MakeNode<SizeGt>(node(), p_other->node());
  return c10::make_intrusive<XLASymNodeImpl>(n_gt, PyType::BOOL);
}

c10::SymNode XLASymNodeImpl::lt(const c10::SymNode& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::size_");
  auto p_other = dynamic_cast<XLASymNodeImpl*>(other.get());
  XLA_CHECK(is_int()) << __FUNCTION__ << " with non-int NYI";
  XLA_CHECK(p_other->is_int()) << __FUNCTION__ << " with non-int NYI";
  auto n_lt = torch_xla::MakeNode<SizeLt>(node(), p_other->node());
  return c10::make_intrusive<XLASymNodeImpl>(n_lt, PyType::BOOL);
}

c10::SymNode XLASymNodeImpl::le(const c10::SymNode& other) {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
}

c10::SymNode XLASymNodeImpl::ge(const c10::SymNode& other) {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::size_");
  auto p_other = dynamic_cast<XLASymNodeImpl*>(other.get());
  XLA_CHECK(is_int()) << __FUNCTION__ << " with non-int NYI";
  XLA_CHECK(p_other->is_int()) << __FUNCTION__ << " with non-int NYI";
  auto n_ge = torch_xla::MakeNode<SizeGe>(node(), p_other->node());
  return c10::make_intrusive<XLASymNodeImpl>(n_ge, PyType::BOOL);
}

c10::SymNode XLASymNodeImpl::ceil() {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
}

c10::SymNode XLASymNodeImpl::floor() {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
}

c10::SymNode XLASymNodeImpl::neg() {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
}

c10::SymNode XLASymNodeImpl::sym_min(const c10::SymNode& other) {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
}

c10::SymNode XLASymNodeImpl::sym_max(const c10::SymNode& other) {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
}

// Force guards when performing these logical operations

c10::SymNode XLASymNodeImpl::sym_or(const c10::SymNode& other) {
  auto a =
      guard_bool(__FILE__, __LINE__) || other->guard_bool(__FILE__, __LINE__);
  auto cnst = torch_xla::MakeNode<SizeConstant>(a);
  return c10::make_intrusive<XLASymNodeImpl>(cnst, PyType::BOOL);
}

c10::SymNode XLASymNodeImpl::sym_and(const c10::SymNode& other) {
  auto a =
      guard_bool(__FILE__, __LINE__) && other->guard_bool(__FILE__, __LINE__);
  auto cnst = torch_xla::MakeNode<SizeConstant>(a);
  return c10::make_intrusive<XLASymNodeImpl>(cnst, PyType::BOOL);
}

c10::SymNode XLASymNodeImpl::sym_not() {
  auto a = !guard_bool(__FILE__, __LINE__);
  auto cnst = torch_xla::MakeNode<SizeConstant>(a);
  return c10::make_intrusive<XLASymNodeImpl>(cnst, PyType::BOOL);
}

// NB: self is ignored here, only the arguments are used
c10::SymNode XLASymNodeImpl::is_contiguous(at::ArrayRef<c10::SymNode> sizes,
                                           at::ArrayRef<c10::SymNode> strides) {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
}
c10::SymNode XLASymNodeImpl::is_channels_last_contiguous_2d(
    at::ArrayRef<c10::SymNode> sizes, at::ArrayRef<c10::SymNode> strides) {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
}
c10::SymNode XLASymNodeImpl::is_channels_last_contiguous_3d(
    at::ArrayRef<c10::SymNode> sizes, at::ArrayRef<c10::SymNode> strides) {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
}
c10::SymNode XLASymNodeImpl::is_channels_last_strides_2d(
    at::ArrayRef<c10::SymNode> sizes, at::ArrayRef<c10::SymNode> strides) {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
}
c10::SymNode XLASymNodeImpl::is_channels_last_strides_3d(
    at::ArrayRef<c10::SymNode> sizes, at::ArrayRef<c10::SymNode> strides) {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
}

// It is used to compute contiguity fields on tensors like "is non overlapping
// and dense" and it's never fetched. If they are never fetched it is fine for
// them to error only if poked.
c10::SymNode XLASymNodeImpl::is_non_overlapping_and_dense(
    at::ArrayRef<c10::SymNode> sizes, at::ArrayRef<c10::SymNode> strides) {
  auto error_node = torch_xla::MakeNode<SizeError>();
  return c10::make_intrusive<XLASymNodeImpl>(error_node, PyType::BOOL);
}

c10::SymNode XLASymNodeImpl::clone() {
  TORCH_LAZY_FN_COUNTER_TIMED_TRACING("xla::size_");
  return c10::make_intrusive<XLASymNodeImpl>(node(), pytype_);
}

c10::SymNode XLASymNodeImpl::sym_float() {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
}

c10::SymNode XLASymNodeImpl::wrap_int(int64_t num) {
  auto cnst = torch_xla::MakeNode<SizeConstant>(num);
  return c10::make_intrusive<XLASymNodeImpl>(cnst, PyType::INT);
}

c10::SymNode XLASymNodeImpl::wrap_float(double num) {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
}

c10::SymNode XLASymNodeImpl::wrap_bool(bool num) {
  auto cnst = torch_xla::MakeNode<SizeConstant>(num);
  return c10::make_intrusive<XLASymNodeImpl>(cnst, PyType::BOOL);
}

int64_t XLASymNodeImpl::guard_int(const char* file, int64_t line) {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
}

double XLASymNodeImpl::guard_float(const char* file, int64_t line) {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
}

bool XLASymNodeImpl::guard_bool(const char* file, int64_t line) {
  // TODO: Take advantages of file and line.
  return bool_();
}

int64_t XLASymNodeImpl::int_() {
  std::shared_ptr<torch::lazy::DimensionNode> dn = torch_xla::DimCast(node());
  return dn->getDynamicValue();
}

bool XLASymNodeImpl::bool_() {
  auto dn = torch_xla::DimCast(node());
  return dn->getDynamicValue() != 0;
}

// "a SymInt has_hint" is equivalent to "a SymInt is backed". Unbacked SymInt is
// the result of a data dependent output like nonzero; we don't know what the
// value is because it's data dependent.
// Returning false here because PyTorch/XLA only creates a SymNodeImpl for
// nonzero output, such as in XLATensorImpl::SetupSymSizeProperties(). During
// propagation such as sz = t1.shape[0] + t2.shape[1] where former argument is
// an unbacked SymInt and latter is backed, sz remains to be an unbacked SymInt.
bool XLASymNodeImpl::has_hint() { return false; }

std::string XLASymNodeImpl::str() {
  return "<=" + std::to_string(DimCast(node().get())->getStaticValue());
}

int64_t XLATensor::GetHandle() const {
  torch::lazy::BackendDataPtr handle = CurrentDataHandle();
  if (handle != nullptr) {
    return std::dynamic_pointer_cast<runtime::ComputationClient::Data>(handle)
        ->GetHandle();
  }
  const auto backend_data =
      torch::lazy::getBackend()->GetComputationDataFromNode(
          GetIrValue().node.get());
  if (backend_data) {
    return backend_data->GetHandle();
  } else {
    XLA_CHECK(false) << "XlaTensor does not have data handle";
  }
}

void XLATensor::MarkDynamicDimension(uint32_t dim) {
  auto* xla_node = dynamic_cast<XlaNode*>(GetIrValue().node.get());
  xla_node->MarkDynamicDimension(dim);
}

bool XLATensor::SetNodeUserMetadata(
    std::shared_ptr<torch::lazy::UserMetaData> metadata) {
  auto* node = dynamic_cast<XlaNode*>(CurrentIrValue().node.get());
  // auto* node = dynamic_cast<torch::lazy::Node*>(GetIrValue().node.get());
  if (node != nullptr) {
    node->SetUserMetadataForSubGraph(metadata);
    return true;
  }
  return false;
}

}  // namespace torch_xla
