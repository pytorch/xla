#include "torch_xla/csrc/tensor.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <set>
#include <stdexcept>
#include <unordered_set>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/profiler/lib/traceme.h"
#include "third_party/xla_client/cache.h"
#include "third_party/xla_client/debug_macros.h"
#include "third_party/xla_client/env_vars.h"
#include "third_party/xla_client/sys_util.h"
#include "third_party/xla_client/thread_pool.h"
#include "third_party/xla_client/unique.h"
#include "third_party/xla_client/xla_util.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/lazy/core/hash.h"
#include "torch/csrc/lazy/core/helpers.h"
#include "torch/csrc/lazy/core/ir_util.h"
#include "torch/csrc/lazy/core/lazy_graph_executor.h"
#include "torch/csrc/lazy/core/metrics.h"
#include "torch/csrc/lazy/core/tensor_util.h"
#include "torch/csrc/lazy/core/util.h"
#include "torch_xla/csrc/computation.h"
#include "torch_xla/csrc/debug_util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ir_dump_util.h"
#include "torch_xla/csrc/layout_manager.h"
#include "torch_xla/csrc/op_by_op_executor.h"
#include "torch_xla/csrc/ops/arithmetic_ir_ops.h"
#include "torch_xla/csrc/ops/cast.h"
#include "torch_xla/csrc/ops/device_data.h"
#include "torch_xla/csrc/ops/dynamic_ir.h"
#include "torch_xla/csrc/ops/expand.h"
#include "torch_xla/csrc/ops/ops.h"
#include "torch_xla/csrc/ops/view.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"
#include "torch_xla/csrc/xla_graph_executor.h"
#include "torch_xla/csrc/xla_sharding_util.h"

namespace torch_xla {

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
    c10::optional<at::ScalarType> logical_element_type) {
  XLATensorPtr xtensor = c10::make_intrusive<XLATensor>(
      XLATensor(std::move(handle), logical_element_type));
  XLAGraphExecutor::Get()->RegisterTensor(xtensor->data());
  return xtensor;
}

XLATensorPtr XLATensor::Create(
    torch::lazy::Value ir_value, const torch::lazy::BackendDevice& device,
    c10::optional<at::ScalarType> logical_element_type) {
  XLATensorPtr xtensor = c10::make_intrusive<XLATensor>(
      XLATensor(std::move(ir_value), device, logical_element_type));
  XLAGraphExecutor::Get()->RegisterTensor(xtensor->data());
  if (UseEagerDebugMode()) {
    std::vector<XLATensorPtr> xtensors({xtensor});
    XLAGraphExecutor::Get()->ApplyEagerSync(xtensors);
  }
  return xtensor;
}

XLATensorPtr XLATensor::Create(
    std::shared_ptr<View> view, const torch::lazy::BackendDevice& device,
    c10::optional<at::ScalarType> logical_element_type) {
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
                     c10::optional<at::ScalarType> logical_element_type)
    : XLATensor(std::make_shared<Data>(handle, handle->device(),
                                       logical_element_type)) {}

XLATensor::XLATensor(torch::lazy::Value ir_value,
                     const torch::lazy::BackendDevice& device,
                     c10::optional<at::ScalarType> logical_element_type)
    : XLATensor(std::make_shared<Data>(std::move(ir_value), device,
                                       logical_element_type)) {
  // Preserve sharding if a new tensor is created from a sharded IR node.
  if (CurrentIrValue()) {
    auto* xla_node = dynamic_cast<XlaNode*>(CurrentIrValue().node.get());
    if (xla_node->GetSharding()) {
      ShardingSpec sharding =
          ShardingSpec{*xla_node->GetSharding(), xla_node->xla_shape()};
      SetShardingSpec(sharding);
    }
  }
  TryLimitGraphSize();
}

XLATensor::XLATensor(std::shared_ptr<View> view,
                     const torch::lazy::BackendDevice& device,
                     c10::optional<at::ScalarType> logical_element_type)
    : XLATensor(std::make_shared<Data>(std::move(view), device,
                                       logical_element_type)) {}

XLATensor::XLATensor(std::shared_ptr<Data> data)
    : torch::lazy::LazyTensor(data),
      data_(std::move(data)),
      storage_(c10::Storage(
          {}, 0,
          c10::DataPtr(nullptr, backendDeviceToAtenDevice(data_->device)))) {}

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
             : TensorTypeFromXlaType(shape().get().element_type());
}

c10::optional<at::ScalarType> XLATensor::dtype_optional() const {
  return data()->logical_element_type;
}

xla::util::MaybeRef<xla::Shape> XLATensor::shape() const {
  if (data()->view != nullptr) {
    return data()->view->shape();
  }
  if (data()->handle != nullptr) {
    return UnwrapXlaData(data()->handle)->shape();
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
          << "Trying to access XLA data while an async operation is in flight: "
          << handle->shape();
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

void XLATensor::SetShardingSpec(const ShardingSpec& sharding) {
  // Existing annotation must be cleared explicitly. We do not clear and
  // overwrite the existing sharding on the user's behalf. This is a no-op if
  // the same sharding already applied.
  if (!sharding_spec()) {
    TORCH_LAZY_COUNTER("SetShardingSpec", 1);
    data()->sharding = std::make_shared<ShardingSpec>(sharding);
  } else {
    XLA_CHECK(ShardingUtil::EqualShardingSpecs(sharding, *sharding_spec()))
        << "Existing sharding annotation, "
        << sharding_spec()->sharding.DebugString()
        << ", must be cleared before applying a new one, "
        << sharding.sharding.DebugString();
  }
  dynamic_cast<XlaNode*>(GetIrValue().node.get())
      ->SetSharding(sharding_spec()->sharding);
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
  if (sharding && ir_value) {
    // The copy of sharding annotation on the IR node should be the same.
    auto* xla_node = dynamic_cast<XlaNode*>(ir_value.node.get());
    if (xla_node->GetSharding()) {
      XLA_CHECK(ShardingUtil::EqualShardingSpecs(
          *sharding,
          ShardingSpec{*xla_node->GetSharding(), xla_node->xla_shape()}));
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
    data()->tensor_data = c10::nullopt;
  }
}

void XLATensor::SetIrValue(torch::lazy::Value ir_value, bool inplace) {
  data()->handle = nullptr;
  data()->tensor_data = c10::nullopt;
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
    TryLimitGraphSize();
  }
  if (UseEagerDebugMode() && ShouldSyncIrNode()) {
    std::vector<XLATensorPtr> xtensors({c10::make_intrusive<XLATensor>(*this)});
    XLAGraphExecutor::Get()->ApplyEagerSync(xtensors);
  }
}

void XLATensor::SetInPlaceIrValue(torch::lazy::Value ir_value) {
  auto xla_shape = shape();
  if (xla_shape.get().element_type() != GetXlaShape(ir_value).element_type()) {
    ir_value =
        torch::lazy::MakeNode<Cast>(ir_value, xla_shape.get().element_type());
  }
  SetIrValue(std::move(ir_value), /*inplace=*/true);
}

void XLATensor::AssignIrValue(torch::lazy::Value ir_value) const {
  TF_VLOG(5) << "Assign IR value: "
             << (ir_value ? ir_value->ToString() : "empty node");
  data()->ir_value = std::move(ir_value);
  data()->generation += 1;
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
    AssignIrValue(CreateTensorNode(handle, /*read_only=*/false));
    return data()->ir_value;
  }
  c10::optional<at::Tensor> tensor_data = CurrentTensorData();
  XLA_CHECK(tensor_data);
  AssignIrValue(GetIrValueForTensor(*tensor_data, GetDevice()));
  return data()->ir_value;
}

torch::lazy::Value XLATensor::CurrentIrValue() const {
  if (data()->view != nullptr) {
    return GetViewUpdate(data()->view).ir_value;
  }
  return data()->ir_value;
}

c10::optional<at::Tensor> XLATensor::CurrentTensorData() const {
  if (data()->view != nullptr && !data()->view->IsUpToDate()) {
    return c10::nullopt;
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
    data = XLAGraphExecutor::Get()->GetDeviceData(tensor, device);
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
    data()->tensor_data = c10::nullopt;
  }
  return ir_value_updated;
}

std::shared_ptr<View> XLATensor::UpdateView(std::shared_ptr<View> view,
                                            torch::lazy::Value ir_value) const {
  if (GetXlaShape(ir_value).dimensions() != view->shape().dimensions()) {
    XLA_CHECK_EQ(
        xla::util::Multiply<int64_t>(GetXlaShape(ir_value).dimensions()),
        xla::util::Multiply<int64_t>(view->shape().dimensions()));

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
  c10::optional<at::Tensor> tensor_data = CurrentTensorData();
  if (!tensor_data) {
    XLAGraphExecutor::Get()->DeviceBarrier(GetDevice());
    // The GetXlaData() call will trigger an ApplyPendingGraph() if an IR
    // XlaNode is available on the tensor.
    std::vector<at::Tensor> tensors = XlaDataToTensors({GetXlaData()}, dtype());
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
        data()->tensor_data = c10::nullopt;
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
    c10::optional<at::ScalarType> logical_element_type) {
  data()->logical_element_type = logical_element_type;
}

void XLATensor::SetTensor(at::Tensor tensor) {
  SetTensorData(tensor);
  data()->view = nullptr;
  data()->handle = nullptr;
  AssignIrValue(torch::lazy::Value());
}

void XLATensor::UpdateFromTensor(at::Tensor tensor, bool sync) {
  torch::lazy::BackendDevice device =
      xla::sys_util::GetEnvBool("XLA_USE_SPMD", false)
          ? ParseDeviceString("SPMD:0")
          : GetDevice();
  if (sync) {
    at::Tensor typed_tensor =
        torch::lazy::CopyTensor(tensor, dtype(), /*copy=*/false);
    SetIrValue(GetIrValueForTensor(typed_tensor, device),
               /*inplace=*/true);
  } else {
    at::Tensor coyped_tensor = torch::lazy::CopyTensor(tensor, dtype());
    SetTensorData(coyped_tensor);
    data()->handle = nullptr;
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
                                   /*logical_element_type=*/c10::nullopt));
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
    c10::optional<at::ScalarType> logical_element_type) const {
  if (!logical_element_type) {
    logical_element_type = dtype_optional();
  }
  if (logical_element_type &&
      RequiresRawTypeCasting(*logical_element_type, &device)) {
    ir_value = torch::lazy::MakeNode<Cast>(ir_value, *logical_element_type);
  }
  return ir_value;
}

XLATensorPtr XLATensor::CreateFrom(torch::lazy::Value ir_value) const {
  ir_value = MaybeCastIrValue(std::move(ir_value), GetDevice(),
                              /*logical_element_type=*/c10::nullopt);
  return Create(std::move(ir_value), GetDevice(), dtype_optional());
}

XLATensorPtr XLATensor::CreateFrom(
    torch::lazy::Value ir_value,
    c10::optional<at::ScalarType> logical_element_type_opt) const {
  ir_value = MaybeCastIrValue(std::move(ir_value), GetDevice(),
                              logical_element_type_opt);
  return Create(std::move(ir_value), GetDevice(), logical_element_type_opt);
}

XLATensorPtr XLATensor::CreateFrom(torch::lazy::Value ir_value,
                                   const torch::lazy::BackendDevice& device,
                                   at::ScalarType logical_element_type) const {
  ir_value =
      MaybeCastIrValue(std::move(ir_value), device, logical_element_type);
  return Create(std::move(ir_value), device, logical_element_type);
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

bool XLATensor::UseEagerDebugMode() {
  static const bool use_eager_debug_mode =
      xla::sys_util::GetEnvBool("XLA_USE_EAGER_DEBUG_MODE", false);
  return use_eager_debug_mode;
}

bool XLATensor::ShouldSyncIrNode() {
  if (!this->data()->ir_value) {
    return false;
  }
  return this->data()->ir_value->op() != xla_device_data;
}

bool XLASymNodeImpl::is_bool() { 
  //std::cout << "xw32, file=" << __FILE__ << ", line=" << __LINE__ << "function=" << __FUNCTION__ << ": " << std::endl;
  return pytype_ == PyType::BOOL; }

bool XLASymNodeImpl::is_int() { 
  //std::cout << "xw32, file=" << __FILE__ << ", line=" << __LINE__ << "function=" << __FUNCTION__ << ": " << std::endl;
  return pytype_ == PyType::INT; }

bool XLASymNodeImpl::is_float() { 
  //std::cout << "xw32, file=" << __FILE__ << ", line=" << __LINE__ << "function=" << __FUNCTION__ << ": " << std::endl;
  return pytype_ == PyType::FLOAT; }

c10::SymNode XLASymNodeImpl::add(const c10::SymNode& other) {
  //std::cout << "xw32, file=" << __FILE__ << ", line=" << __LINE__ << "function=" << __FUNCTION__ << ": " << std::endl;
  auto p_other = dynamic_cast<XLASymNodeImpl*>(other.get());
  XLA_CHECK(is_int()) << __FUNCTION__ << " with non-int NYI";
  XLA_CHECK(p_other->is_int()) << __FUNCTION__ << " with non-int NYI";
  auto n_add = torch::lazy::MakeNode<SizeAdd>(node(), p_other->node());
  return c10::make_intrusive<XLASymNodeImpl>(n_add, PyType::INT);
}

c10::SymNode XLASymNodeImpl::sub(const c10::SymNode& other) {
  TORCH_LAZY_FN_COUNTER("xla::size_");

  torch_xla::XLASymNodeImpl* p_other =
      dynamic_cast<XLASymNodeImpl*>(other.get());
  XLA_CHECK(is_int()) << __FUNCTION__ << " with non-int NYI";
  XLA_CHECK(p_other->is_int()) << __FUNCTION__ << " with non-int NYI";
  torch::lazy::NodePtr n_sub =
      torch::lazy::MakeNode<SizeSub>(node(), p_other->node());
  return c10::make_intrusive<XLASymNodeImpl>(n_sub, PyType::INT);
}

c10::SymNode XLASymNodeImpl::mul(const c10::SymNode& other) {
  auto p_other = dynamic_cast<XLASymNodeImpl*>(other.get());
  XLA_CHECK(is_int()) << __FUNCTION__ << " with non-int NYI";
  XLA_CHECK(p_other->is_int()) << __FUNCTION__ << " with non-int NYI";
  auto n_mul =
      torch::lazy::MakeNode<torch_xla::SizeMul>(node(), p_other->node());
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
  auto n_div = torch::lazy::MakeNode<SizeDiv>(node(), p_other->node());
  return c10::make_intrusive<XLASymNodeImpl>(n_div, PyType::INT);
}

c10::SymNode XLASymNodeImpl::mod(const c10::SymNode& other) {
  TORCH_LAZY_FN_COUNTER("xla::size_");
  torch_xla::XLASymNodeImpl* p_other =
      dynamic_cast<XLASymNodeImpl*>(other.get());
  XLA_CHECK(is_int()) << __FUNCTION__ << " with non-int NYI";
  XLA_CHECK(p_other->is_int()) << __FUNCTION__ << " with non-int NYI";
  torch::lazy::NodePtr n_mod =
      torch::lazy::MakeNode<SizeMod>(node(), p_other->node());
  return c10::make_intrusive<XLASymNodeImpl>(n_mod, PyType::INT);
}

c10::SymNode XLASymNodeImpl::eq(const c10::SymNode& other) {
  //std::cout << "xw32, file=" << __FILE__ << ", line=" << __LINE__ << "function=" << __FUNCTION__ << ": " << std::endl;
  auto p_other = dynamic_cast<XLASymNodeImpl*>(other.get());
  XLA_CHECK(is_int()) << __FUNCTION__ << " with non-int NYI";
  XLA_CHECK(p_other->is_int()) << __FUNCTION__ << " with non-int NYI";
  auto n_eq = torch::lazy::MakeNode<SizeEq>(node(), p_other->node());
  // std::cout << "xw32, file=" << __FILE__ << ", line=" << __LINE__ << "function=" << __FUNCTION__ << ": this.get_backtrace_when_created()=" << get_backtrace_when_created() << std::endl;
  // std::cout << "xw32, file=" << __FILE__ << ", line=" << __LINE__ << "function=" << __FUNCTION__ << ": p_other->get_backtrace_when_created()=" << p_other->get_backtrace_when_created() << std::endl;
  std::cout << "xw32, file=" << __FILE__ << ", line=" << __LINE__ << "function=" << __FUNCTION__ << ": this.get_xlaTensorImplAddr()=" << get_xlaTensorImplAddr() << std::endl;
  std::cout << "xw32, file=" << __FILE__ << ", line=" << __LINE__ << "function=" << __FUNCTION__ << ": p_other->get_xlaTensorImplAddr()=" << p_other->get_xlaTensorImplAddr() << std::endl;
  return c10::make_intrusive<XLASymNodeImpl>(n_eq, PyType::BOOL);
}

c10::SymNode XLASymNodeImpl::ne(const c10::SymNode& other) {
  TORCH_LAZY_FN_COUNTER("xla::size_");
  auto p_other = dynamic_cast<XLASymNodeImpl*>(other.get());
  XLA_CHECK(is_int()) << __FUNCTION__ << " with non-int NYI";
  XLA_CHECK(p_other->is_int()) << __FUNCTION__ << " with non-int NYI";
  auto n_ne = torch::lazy::MakeNode<SizeNe>(node(), p_other->node());
  return c10::make_intrusive<XLASymNodeImpl>(n_ne, PyType::BOOL);
}

c10::SymNode XLASymNodeImpl::gt(const c10::SymNode& other) {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
}

c10::SymNode XLASymNodeImpl::lt(const c10::SymNode& other) {
  TORCH_LAZY_FN_COUNTER("xla::size_");
  auto p_other = dynamic_cast<XLASymNodeImpl*>(other.get());
  XLA_CHECK(is_int()) << __FUNCTION__ << " with non-int NYI";
  XLA_CHECK(p_other->is_int()) << __FUNCTION__ << " with non-int NYI";
  auto n_lt = torch::lazy::MakeNode<SizeLt>(node(), p_other->node());
  return c10::make_intrusive<XLASymNodeImpl>(n_lt, PyType::BOOL);
}

c10::SymNode XLASymNodeImpl::le(const c10::SymNode& other) {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
}

c10::SymNode XLASymNodeImpl::ge(const c10::SymNode& other) {
  TORCH_LAZY_FN_COUNTER("xla::size_");
  auto p_other = dynamic_cast<XLASymNodeImpl*>(other.get());
  XLA_CHECK(is_int()) << __FUNCTION__ << " with non-int NYI";
  XLA_CHECK(p_other->is_int()) << __FUNCTION__ << " with non-int NYI";
  auto n_ge = torch::lazy::MakeNode<SizeGe>(node(), p_other->node());
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

c10::SymNode XLASymNodeImpl::sym_or(const c10::SymNode& other) {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
}

c10::SymNode XLASymNodeImpl::sym_and(const c10::SymNode& other) {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
}

c10::SymNode XLASymNodeImpl::sym_not() {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
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
c10::SymNode XLASymNodeImpl::is_non_overlapping_and_dense(
    at::ArrayRef<c10::SymNode> sizes, at::ArrayRef<c10::SymNode> strides) {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
}

c10::SymNode XLASymNodeImpl::clone() {
  //std::cout << "xw32, file=" << __FILE__ << ", line=" << __LINE__ << "function=" << __FUNCTION__ << ": " << std::endl;
  TORCH_LAZY_FN_COUNTER("xla::size_");
  return c10::make_intrusive<XLASymNodeImpl>(node(), pytype_);
}

c10::SymNode XLASymNodeImpl::sym_float() {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
}

c10::SymNode XLASymNodeImpl::wrap_int(int64_t num) {
  //std::cout << "xw32, file=" << __FILE__ << ", line=" << __LINE__ << "function=" << __FUNCTION__ << ": " << std::endl;
  auto cnst = torch::lazy::MakeNode<SizeConstant>(num);
  return c10::make_intrusive<XLASymNodeImpl>(cnst, PyType::INT);
}

c10::SymNode XLASymNodeImpl::wrap_float(double num) {
  XLA_CHECK(false) << "XLASymNodeImpl::" << __FUNCTION__
                   << " has not been implemented.";
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
  //std::cout << "xw32, file=" << __FILE__ << ", line=" << __LINE__ << "function=" << __FUNCTION__ << ": " << std::endl;
  // TODO: Take advantages of file and line.
  return bool_();
}

int64_t XLASymNodeImpl::int_() {
  //std::cout << "xw32, file=" << __FILE__ << ", line=" << __LINE__ << "function=" << __FUNCTION__ << ": " << std::endl;
  std::shared_ptr<torch::lazy::DimensionNode> dn = torch_xla::DimCast(node());
  return dn->getDynamicValue();
}

bool XLASymNodeImpl::bool_() {
  //std::cout << "xw32, file=" << __FILE__ << ", line=" << __LINE__ << "function=" << __FUNCTION__ << ": " << std::endl;
  auto dn = torch_xla::DimCast(node());
  return dn->getDynamicValue() != 0;
}

bool XLASymNodeImpl::has_hint() { 
  std::cout << "xw32, file=" << __FILE__ << ", line=" << __LINE__ << "function=" << __FUNCTION__ << ": " << std::endl;
  return false; }

std::string XLASymNodeImpl::str() {
  //std::cout << "xw32, file=" << __FILE__ << ", line=" << __LINE__ << "function=" << __FUNCTION__ << ": " << std::endl;
  return "<=" + std::to_string(DimCast(node().get())->getStaticValue());
}

int64_t XLATensor::GetOpaqueHandle() const {
  torch::lazy::BackendDataPtr handle = CurrentDataHandle();
  if (handle != nullptr) {
    return UnwrapXlaData(handle)->GetOpaqueHandle();
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

}  // namespace torch_xla
