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
#include "tensorflow/compiler/xla/xla_client/cache.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/env_vars.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/thread_pool.h"
#include "tensorflow/compiler/xla/xla_client/unique.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/profiler/lib/traceme.h"
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
  TORCH_CHECK(data_ != nullptr, "Trying to access a null cursor");
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
    ApplyPendingGraph();
  } else {
    XLA_CHECK(data()->tensor_data);
    data()->handle = TensorToXlaData(*data()->tensor_data, GetDevice());
  }
  return data()->handle;
}

XLATensor::ShardingSpecPtr XLATensor::sharding_spec() const {
  torch::lazy::Value ir_value = CurrentIrValue();
  if (ir_value) {
    XLA_CHECK(ir_value.node != nullptr) << "Tyring to access a null cursor";
    auto sharding = dynamic_cast<XlaNode*>(ir_value.node.get())->GetSharding();
    if (sharding == nullptr) {
      return nullptr;
    }
    return std::make_shared<ShardingSpec>(*sharding);
  }
  return nullptr;
}

void XLATensor::SetShardingSpec(const ShardingSpec& sharding) {
  // Existing annotation must be cleared explicitly. We do not clear and
  // overwrite the existing sharding on the user's behalf. This is a no-op if
  // the same sharding already applied.
  if (sharding_spec() == nullptr ||
      !ShardingUtil::EqualShardingSpecs(sharding, *sharding_spec())) {
    TORCH_LAZY_COUNTER("SetShardingSpec", 1);
    XLA_CHECK(GetIrValue().node != nullptr) << "Tyring to access a null cursor";
    dynamic_cast<XlaNode*>(GetIrValue().node.get())
        ->SetSharding(sharding.sharding);
  }
}
void XLATensor::ClearShardingSpec() {
  torch::lazy::Value ir_value = CurrentIrValue();
  if (ir_value) {
    // This should be a no-op if there is no sharding.
    dynamic_cast<XlaNode*>(ir_value.node.get())->ClearSharding();
  }
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
  ShardingSpecPtr sharding = sharding_spec();
  if (sharding != nullptr) {
    if (!ir_value) {
      // Create a tensor node if applicable, re-use the current IR otherwise.
      // TODO(yeounoh) this has some performance implications for convolution.
      ir_value = GetIrValue();
    }
    dynamic_cast<XlaNode*>(ir_value.node.get())
        ->SetSharding(sharding->sharding);
  }
  data()->ir_value = std::move(ir_value);
  data()->generation += 1;
}

void XLATensor::TryLimitGraphSize() {
  static const size_t kCheckFrequency =
      xla::sys_util::GetEnvInt("XLA_TRIM_GRAPH_CHECK_FREQUENCY", 5000);
  static const size_t kMaxPendingGraphSize =
      xla::sys_util::GetEnvInt("XLA_TRIM_GRAPH_SIZE", 100000);
  if (data()->ir_value &&
      XLAGraphExecutor::Get()->IncTrimCounter() % kCheckFrequency == 0) {
    size_t graph_size =
        torch::lazy::Util::GetGraphSize({data()->ir_value.node.get()});
    if (graph_size > kMaxPendingGraphSize) {
      TORCH_LAZY_COUNTER("TrimIrGraph", 1);
      ApplyPendingGraph();
    }
  }
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

torch::lazy::Value XLATensor::CreateTensorNode(torch::lazy::BackendDataPtr data,
                                               bool read_only) const {
  data->SetInfo(
      std::make_shared<torch::lazy::LazyGraphExecutor::DeviceDataInfo>(
          GetUniqueId(), read_only));
  return torch::lazy::MakeNode<DeviceData>(std::move(data));
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

int64_t XLATensor::GetNextTensorId() {
  static std::atomic<int64_t>* id_generator = new std::atomic<int64_t>(1);
  return id_generator->fetch_add(1);
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

bool XLASymNodeImpl::is_int() {
  // TODO: handle not is int
  return true;
}

bool XLASymNodeImpl::is_float() {
  // TODO: handle not is int
  return false;
}

bool XLASymNodeImpl::bool_() {
  auto dn = torch_xla::DimCast(node());
  return dn->getDynamicValue() != 0;
}

int64_t XLASymNodeImpl::int_() {
  std::shared_ptr<torch::lazy::DimensionNode> dn = torch_xla::DimCast(node());
  return dn->getDynamicValue();
}

c10::SymNode XLASymNodeImpl::eq(const c10::SymNode& other) {
  auto pother = dynamic_cast<XLASymNodeImpl*>(other.get());
  auto neq = torch::lazy::MakeNode<SizeEq>(node(), pother->node());
  return c10::make_intrusive<XLASymNodeImpl>(neq);
}

c10::SymNode XLASymNodeImpl::add(const c10::SymNode& other) {
  auto pother = dynamic_cast<XLASymNodeImpl*>(other.get());
  auto nadd = torch::lazy::MakeNode<SizeAdd>(node(), pother->node());
  return c10::make_intrusive<XLASymNodeImpl>(nadd);
}

c10::SymNode XLASymNodeImpl::mul(const c10::SymNode& other) {
  auto pother = dynamic_cast<XLASymNodeImpl*>(other.get());
  auto nmul = torch::lazy::MakeNode<torch_xla::SizeMul>(node(), pother->node());
  return c10::make_intrusive<XLASymNodeImpl>(nmul);
}

c10::SymNode XLASymNodeImpl::wrap_int(int64_t num) {
  auto cnst = torch::lazy::MakeNode<SizeConstant>(num);
  return c10::make_intrusive<XLASymNodeImpl>(cnst);
}

c10::SymNode XLASymNodeImpl::floordiv(const c10::SymNode& other) {
  auto pother = dynamic_cast<XLASymNodeImpl*>(other.get());
  auto ndiv = torch::lazy::MakeNode<SizeDiv>(node(), pother->node());
  return c10::make_intrusive<XLASymNodeImpl>(ndiv);
}

std::string XLASymNodeImpl::str() {
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
