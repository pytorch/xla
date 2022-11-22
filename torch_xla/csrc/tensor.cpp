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
  XLAGraphExecutor::Get()->RegisterTensor(xtensor->data_ptr());
  return xtensor;
}

XLATensorPtr XLATensor::Create(
    torch::lazy::BackendDataPtr xla_data,
    c10::optional<at::ScalarType> logical_element_type) {
  XLATensorPtr xtensor = c10::make_intrusive<XLATensor>(
      XLATensor(std::move(xla_data), logical_element_type));
  XLAGraphExecutor::Get()->RegisterTensor(xtensor->data_ptr());
  return xtensor;
}

XLATensorPtr XLATensor::Create(
    torch::lazy::Value ir_value, const torch::lazy::BackendDevice& device,
    c10::optional<at::ScalarType> logical_element_type) {
  XLATensorPtr xtensor = c10::make_intrusive<XLATensor>(
      XLATensor(std::move(ir_value), device, logical_element_type));
  XLAGraphExecutor::Get()->RegisterTensor(xtensor->data_ptr());
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
  XLAGraphExecutor::Get()->RegisterTensor(xtensor->data_ptr());
  return xtensor;
}

XLATensor::XLATensor(const at::Tensor& tensor,
                     const torch::lazy::BackendDevice& device)
    : XLATensor(std::make_shared<Data>(tensor, device)) {}

XLATensor::XLATensor(torch::lazy::BackendDataPtr xla_data,
                     c10::optional<at::ScalarType> logical_element_type)
    : XLATensor(std::make_shared<Data>(xla_data, xla_data->device(),
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
    : data_(std::move(data)),
      storage_(c10::Storage(
          {}, 0,
          c10::DataPtr(nullptr, backendDeviceToAtenDevice(data_->device)))) {}

XLATensor::Data* XLATensor::data() const {
  XLA_CHECK(data_ != nullptr) << "Trying to access a null cursor";
  return data_.get();
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
  if (data()->xla_data != nullptr) {
    return UnwrapXlaData(data()->xla_data)->shape();
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

const torch::lazy::BackendDevice& XLATensor::GetDevice() const {
  return data()->device;
}

int64_t XLATensor::GetUniqueId() const { return data()->unique_id; }

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
    torch::lazy::BackendDataPtr xla_data = CurrentXlaData();
    if (xla_data != nullptr) {
      XLA_CHECK(xla_data->HasValue())
          << "Trying to access XLA data while an async operation is in flight: "
          << xla_data->shape();
      return xla_data;
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
    data()->xla_data = TensorToXlaData(*data()->tensor_data, GetDevice());
  }
  return data()->xla_data;
}

torch::lazy::BackendDataPtr XLATensor::CurrentXlaData() const {
  return data()->xla_data;
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

void XLATensor::SetShardingSpec(const ShardingSpec& sharding_spec) {
  XLA_CHECK(GetIrValue().node != nullptr) << "Tyring to access a null cursor";
  dynamic_cast<XlaNode*>(GetIrValue().node.get())
      ->SetSharding(sharding_spec.sharding);
}
void XLATensor::ClearShardingSpec() {
  torch::lazy::Value ir_value = CurrentIrValue();
  if (ir_value) {
    dynamic_cast<XlaNode*>(GetIrValue().node.get())->ClearSharding();
  }
}

void XLATensor::SetXlaData(torch::lazy::BackendDataPtr xla_data) {
  SetXlaData(std::move(xla_data), /*sync=*/true);
}

void XLATensor::SetXlaData(torch::lazy::BackendDataPtr xla_data, bool sync) {
  data()->xla_data = std::move(xla_data);
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
  data()->xla_data = nullptr;
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
    // Sharded xla_data is accompanied by sharding annotation.
    // Use unsynced ir_value or xla_data to hold the annotation.
    // TODO(yeounoh): This does not propagate sharding to views.
    if (!ir_value) {
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
  torch::lazy::BackendDataPtr xla_data = CurrentXlaData();
  if (xla_data != nullptr) {
    // In case of tensor node, we do not clear the XLA data when we set the IR
    // node. This because we want further calls to GetIrValue() to fetch the
    // same IR node, and not create new ones (even though the lowering context
    // will still collapse them all into a single XLA parameter op). So call
    // which wants the XLA data will still find it, w/out having to fetch it
    // via a computation client from-server call.
    AssignIrValue(CreateTensorNode(xla_data, /*read_only=*/false));
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

void XLATensor::SetTensorData(at::Tensor tensor_data) {
  data()->tensor_data = std::move(tensor_data);
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
    data()->xla_data = nullptr;
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
      if (data()->ir_value || data()->xla_data != nullptr ||
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
  data()->xla_data = nullptr;
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
    data()->xla_data = nullptr;
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
  // device, so that a call to CurrentXlaData() returns a valid pointer.
  if (CurrentXlaData() == nullptr) {
    std::vector<XLATensorPtr> tensors({c10::make_intrusive<XLATensor>(*this)});
    XLAGraphExecutor::Get()->SyncTensorsGraph(&tensors, {}, /*wait=*/true,
                                              /*sync_xla_data=*/false);
  }
}

void XLATensor::ApplyEagerSync(std::vector<XLATensorPtr>& tensors) {
  SyncTensorsGraph(&tensors, {}, /*wait=*/false, /*sync_xla_data=*/false);
}

XLATensor::SyncTensorCollection XLATensor::CollectSyncTensors(
    const std::vector<XLATensorPtr>& tensors, const SyncTensorsConfig& config) {
  tensorflow::profiler::TraceMe activity(
      "CollectSyncTensors", tensorflow::profiler::TraceMeLevel::kInfo);
  xla::util::Unique<torch::lazy::BackendDevice> unique_device;
  for (size_t i = 0; i < tensors.size(); ++i) {
    unique_device.set(tensors[i]->GetDevice());
  }
  SyncTensorCollection coll;
  if (!unique_device) {
    return coll;
  }

  std::vector<at::Tensor> at_tensors;
  std::vector<std::string> devices;
  std::vector<ShardingSpecPtr> shardings;
  std::vector<size_t> at_tensor_index;
  std::unordered_set<int64_t> tensor_ids;
  // The force_xla_data controls aliasing compilation, so effectively the same
  // graph with on/off force_xla_data should not match, hash wise.
  coll.hash = torch::lazy::MHash(config.force_xla_data);
  coll.config = config;
  coll.device = *unique_device;
  coll.indices.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensor_ids.insert(tensors[i]->GetUniqueId()).second &&
        tensors[i]->CurrentXlaData() == nullptr) {
      torch::lazy::Value ir_value = tensors[i]->CurrentIrValue();
      if (ir_value) {
        if (ShouldSyncIrValue(ir_value)) {
          // Add only tensors which need to be synced.
          coll.hash = torch::lazy::HashCombine(coll.hash, ir_value.hash());
          coll.indices.push_back(i);
        }
      } else if (config.force_xla_data) {
        // The tensor only has at::Tensor data. We need to queue it for a
        // device upload.
        c10::optional<at::Tensor> tensor_data = tensors[i]->CurrentTensorData();
        XLA_CHECK(tensor_data);
        at_tensors.push_back(*tensor_data);
        shardings.push_back(tensors[i]->sharding_spec());
        devices.push_back(tensors[i]->GetDevice().toString());
        at_tensor_index.push_back(i);
      }
    }
  }
  // Mix the hash with the resource domain hashes as compile handles are only
  // valid within a domain (usually a single host).
  coll.hash = torch::lazy::MHash(
      coll.hash,
      xla::ComputationClient::Get()->GetResourceDomain(coll.device.toString()));
  if (!at_tensors.empty()) {
    TORCH_LAZY_COUNTER("SyncTensorsToData", at_tensors.size());
    // Create data handles with shardings. If a tensor has a
    // sharding annotation, then a BackendDataPtr with PjRtShardedData is
    // returned; if there is no sharding annotation, then a BackendDataPtr with
    // PjRtData is returned.
    std::vector<torch::lazy::BackendDataPtr> handles =
        CreateTensorsData(at_tensors, shardings, devices);
    for (size_t i = 0; i < handles.size(); ++i) {
      // If we are here, it means that the IR torch::lazy::Value for the
      // tensor is not present. Also, we uploaded the at::Tensor data to the
      // device, but such data is still valid so we leave it live on the XLA
      // tensor (so that a following ToTensor() does not need to fetch it from
      // device).
      tensors[at_tensor_index[i]]->data()->xla_data = std::move(handles[i]);
    }
  }
  TF_VLOG(4) << "Tensors graph hash " << torch::lazy::HashToString(coll.hash)
             << " on device " << coll.device;
  return coll;
}

XLATensor::ComputationCache::TypePtr XLATensor::LookupCachedCompile(
    const std::vector<XLATensorPtr>& tensors, const torch::lazy::hash_t& hash) {
  ComputationCache::TypePtr cached_computation =
      GetComputationCache()->Get(hash);
  if (cached_computation == nullptr) {
    TORCH_LAZY_COUNTER("UncachedCompile", 1);
    return nullptr;
  }
  TF_VLOG(5) << "Graph hash " << torch::lazy::HashToString(hash)
             << " is computation hash "
             << torch::lazy::HashToString(torch::lazy::Hash(
                    cached_computation->computation->computation()
                        .proto()
                        .SerializeAsString()));
  TORCH_LAZY_COUNTER("CachedCompile", 1);
  return cached_computation;
}

std::shared_ptr<XLATensor::Async> XLATensor::TryRunCachedSync(
    std::vector<XLATensorPtr>* tensors, SyncTensorCollection* coll,
    PostOrderData* po_data,
    const std::vector<torch::lazy::BackendDataPtr>& tensor_data_vec) {
  ComputationCache::TypePtr cached_computation =
      LookupCachedCompile(*tensors, coll->hash);
  if (cached_computation == nullptr) {
    return nullptr;
  }
  XLA_VALUE_METRIC("TensorsGraphSize", po_data->post_order.size());
  TF_VLOG(5) << "TensorsGraphSize=" << po_data->post_order.size();

  return ScheduleSyncTensorsGraph(
      tensors, coll, std::move(po_data->parameters_data),
      coll->device.toString(), std::move(cached_computation), tensor_data_vec);
}

XLATensor::ComputationCache* XLATensor::GetComputationCache() {
  static const size_t kMaxCacheSize =
      xla::sys_util::GetEnvInt("XLA_COMPILATION_CACHE_SIZE", 1024);
  static ComputationCache* cache = new ComputationCache(kMaxCacheSize);
  return cache;
}

XLATensor::PostOrderData XLATensor::RunPostOrder(
    const std::vector<torch::lazy::Value>& ir_values,
    SyncTensorCollection* coll) {
  tensorflow::profiler::TraceMe activity(
      "RunPostOrder", tensorflow::profiler::TraceMeLevel::kInfo);
  std::vector<const torch::lazy::Node*> roots;
  roots.reserve(ir_values.size());
  for (auto ir_value : ir_values) {
    roots.push_back(ir_value.node.get());
  }
  PostOrderData po_data;
  po_data.post_order =
      torch::lazy::Util::ComputePostOrder(roots, &po_data.emission_map);
  std::unordered_map<xla::ComputationClient::Data::OpaqueHandle, size_t>
      data_handles;

  for (auto node : po_data.post_order) {
    const auto backend_data =
        torch::lazy::getBackend()->GetComputationDataFromNode(node);
    if (backend_data != nullptr) {
      /* Acceptable race condition: HasValue may return false. This is OK
       * since the conditional barrier is a performance optimization. */
      if (!backend_data->HasValue()) {
        TensorCollectionBarrier(coll);
      }
      xla::ComputationClient::Data::OpaqueHandle handle =
          backend_data->GetHandle();
      auto it = data_handles.find(handle);
      if (it != data_handles.end()) {
        po_data.parameter_sequence.push_back(it->second);
      } else {
        po_data.parameter_sequence.push_back(po_data.parameters_data.size());
        data_handles[handle] = po_data.parameters_data.size();
        po_data.parameters_data.push_back(backend_data);
      }
    }
  }
  return po_data;
}

std::vector<torch::lazy::Value> XLATensor::CollectRoots(
    const std::vector<XLATensorPtr>& tensors,
    absl::Span<const size_t> indices) {
  std::vector<torch::lazy::Value> roots;
  roots.reserve(indices.size());
  for (auto index : indices) {
    roots.push_back(tensors.at(index)->CurrentIrValue());
  }
  return roots;
}

void XLATensor::ExtractIRAndPrepareXlaData_(
    std::vector<XLATensorPtr>* tensors, const SyncTensorsConfig& config,
    const absl::Span<const size_t> indices,
    std::vector<torch::lazy::Value>& ir_values,
    std::vector<torch::lazy::BackendDataPtr>& tensor_data_vec) {
  tensorflow::profiler::TraceMe activity(
      "ExtractIRAndPrepareXlaData_", tensorflow::profiler::TraceMeLevel::kInfo);
  ir_values.reserve(indices.size());
  tensor_data_vec.reserve(indices.size());
  for (auto index : indices) {
    XLATensorPtr& tensor = (*tensors)[index];
    torch::lazy::Value ir_value = tensor->CurrentIrValue();
    ir_values.push_back(ir_value);
    const torch::lazy::BackendDevice& tensor_device = tensor->GetDevice();
    xla::Shape shape = MakeShapeWithDeviceLayout(
        tensor->shape(), static_cast<XlaDeviceType>(tensor_device.type()));
    torch::lazy::BackendDataPtr xla_data =
        WrapXlaData(xla::ComputationClient::Get()->CreateDataPlaceholder(
            tensor_device.toString(), std::move(shape)));
    tensor_data_vec.push_back(xla_data);
    if (tensor->CurrentXlaData() == nullptr && config.force_xla_data) {
      tensor->AssignIrValue(torch::lazy::Value());
    }
  }
}

std::vector<torch::lazy::BackendDataPtr> XLATensor::SetTensorData(
    std::vector<XLATensorPtr>* tensors, const SyncTensorsConfig& config,
    absl::Span<const size_t> indices,
    const std::vector<torch::lazy::BackendDataPtr>& tensor_data_vec) {
  tensorflow::profiler::TraceMe activity(
      "SetTensorData", tensorflow::profiler::TraceMeLevel::kInfo);
  std::vector<torch::lazy::BackendDataPtr> tensors_data;
  tensors_data.reserve(indices.size());
  for (int i = 0; i < indices.size(); i++) {
    auto index = indices[i];
    XLATensorPtr& tensor = (*tensors)[index];
    // If the config.force_xla_data flag is true, the purpose of this tensor
    // sync operation is to truncate the IR graph and materialize device data
    // in place of IR graph, on selected tensors. But since operation will
    // complete asynchronously, if a tensor does not already have device data,
    // we need to install a placeholder. Since at this point we hold a lock on
    // the device where the tensors reside (locks held within the coll
    // structure, and moved into the async variable), any other operation
    // trying to access the tensor's device data will have to wait until the
    // asynchronous operation completes.
    torch::lazy::BackendDataPtr xla_data = tensor->CurrentXlaData();
    if (xla_data == nullptr && config.force_xla_data) {
      xla_data = tensor_data_vec[i];
      // Note: We are not using SetXlaData method here since that method
      // resets the ir_value. We have already done the resetting as part
      // of ExtractIRAndPrepareXlaData_ to overlap with previous execution.
      tensor->data()->xla_data = xla_data;
      tensor->data()->view = nullptr;
      tensor->data()->tensor_data = c10::nullopt;
    }
    tensors_data.emplace_back(std::move(xla_data));
  }
  return tensors_data;
}

std::shared_ptr<XLATensor::Async> XLATensor::ScheduleSyncTensorsGraph(
    SyncTensorCollection* coll,
    std::vector<torch::lazy::BackendDataPtr> parameters_data,
    std::vector<torch::lazy::BackendDataPtr> tensors_data,
    ComputationCache::TypePtr cached_computation) {
  tensorflow::profiler::TraceMe activity(
      "ScheduleSyncTensorsGraph", tensorflow::profiler::TraceMeLevel::kInfo);
  TensorCollectionBarrier(coll);
  std::shared_ptr<Async> async = std::make_shared<Async>(
      coll, std::move(parameters_data), std::move(tensors_data),
      std::move(cached_computation));

  auto syncfn = [async, hash = coll->hash]() {
    try {
      std::vector<torch::lazy::BackendDataPtr> results;
      // Execute replicated if the compiled computation is partitioned.
      if (async->cached_computation->is_sharded) {
        std::vector<std::string> devices =
            xla::ComputationClient::Get()->GetAllDevices();
        std::vector<std::vector<xla::ComputationClient::DataPtr>>
            device_arguments = torch_xla::ShardingUtil::InputHandler(
                UnwrapXlaData(async->parameters_data), devices);
        xla::ComputationClient::ExecuteReplicatedOptions execute_options;

        TF_VLOG(3) << "Executing IR graph hash "
                   << torch::lazy::HashToString(hash) << " on all devices.";
        // TODO(jwtan): Remove the WrapXlaData when inherits LazyGraphExecutor.
        results = WrapXlaData(xla::ComputationClient::Get()->ExecuteReplicated(
            *async->cached_computation->computation->client_computation(),
            device_arguments, devices,
            execute_options)[0]);  // TODO(yeounoh) assumes replicated outputs
        TF_VLOG(3) << "Executing IR graph hash "
                   << torch::lazy::HashToString(hash)
                   << " on all devices, done!";
      } else {
        TF_VLOG(3) << "Executing IR graph hash "
                   << torch::lazy::HashToString(hash) << " on device "
                   << async->device << " ...";
        results = torch::lazy::getBackend()->ExecuteComputation(
            async->cached_computation->computation, async->parameters_data,
            ParseDeviceString(async->device));
        TF_VLOG(3) << "Executing IR graph hash "
                   << torch::lazy::HashToString(hash) << " on device "
                   << async->device << " done!";
      }
      for (size_t i = 0; i < results.size(); ++i) {
        if (async->tensors_data[i] != nullptr) {
          async->tensors_data[i]->Assign(*results[i]);
        } else {
          async->tensors_data[i] = std::move(results[i]);
        }
      }
    } catch (...) {
      // There are two paths of discovery of an exception happening on an
      // asynchronous task. One happens if the creator of the asynchronous task
      // explicitly waits for completion, in which case the exception will be
      // thrown from the Wait() API. Re-throwing the exception below makes sure
      // this will be captured by the completer function created below, and
      // surfaced by the Wait() API. But we also need to surface the exception
      // even in case the caller does not wait, and that is accomplished by
      // setting the unlockers status. In that case the exception will be
      // surfaced when the user tries to acquire the device locks the next time.
      std::exception_ptr exptr = std::current_exception();
      for (auto& unlocker : async->unlocker) {
        unlocker.SetStatus(exptr);
      }
      throw;
    }
  };

  xla::env::ScheduleIoClosure(async->mwait.Completer(std::move(syncfn)));
  return async;
}

std::shared_ptr<XLATensor::Async> XLATensor::ScheduleSyncTensorsGraph(
    std::vector<XLATensorPtr>* tensors, SyncTensorCollection* coll,
    std::vector<torch::lazy::BackendDataPtr> parameters_data,
    std::string device, ComputationCache::TypePtr cached_computation,
    const std::vector<torch::lazy::BackendDataPtr>& tensor_data_vec) {
  auto tensors_data =
      SetTensorData(tensors, coll->config, coll->indices, tensor_data_vec);
  return ScheduleSyncTensorsGraph(coll, std::move(parameters_data),
                                  std::move(tensors_data),
                                  std::move(cached_computation));
}

void XLATensor::SyncTensorsGraph(std::vector<XLATensorPtr>* tensors,
                                 absl::Span<const std::string> devices,
                                 bool wait, bool sync_xla_data) {
  TF_VLOG(4) << "Trying to sync the value of " << tensors->size()
             << " tensor(s)";
  tensorflow::profiler::TraceMe activity(
      "SyncTensorsGraph", tensorflow::profiler::TraceMeLevel::kInfo);
  static const bool op_by_op =
      xla::sys_util::GetEnvBool("XLA_SYNC_TENSORS_OPBYOP", false);
  SyncTensorsConfig config;
  config.sync_xla_data = sync_xla_data;
  if (op_by_op) {
    OpByOpAsync async = SyncTensorsGraphOpByOp(tensors, devices, config);
    if (wait) {
      async.Wait();
    }
  } else {
    auto async = SyncTensorsGraphInternal(tensors, devices, config);
    if (wait && async != nullptr) {
      async->mwait.Wait();
    }
  }
}

void XLATensor::SyncLiveTensorsGraph(const torch::lazy::BackendDevice* device,
                                     absl::Span<const std::string> devices,
                                     bool wait) {
  tensorflow::profiler::TraceMe activity(
      "SyncLiveTensorsGraph", tensorflow::profiler::TraceMeLevel::kInfo);
  auto tensors = GetLiveTensors(device);
  TF_VLOG(4) << tensors.size() << " live tensors: devices=("
             << absl::StrJoin(devices, ",") << ")";
  SyncTensorsGraph(&tensors, devices, wait, /*sync_xla_data=*/true);
}

void XLATensor::MarkStep(const torch::lazy::BackendDevice& device) {
  // TODO(jwtan): Replace this with TORCH_LAZY_COUNTER. We need MarkStep to
  // remain as XLA_COUNTER to support xla::metrics::CreatePerformanceReport().
  XLA_COUNTER("MarkStep", 1);
  DeviceContextArena::Get()->MarkStep(device);
  torch::lazy::ScopePusher::ResetScopes();
  g_tls_data.Reset();
}

void XLATensor::WaitDeviceOps(absl::Span<const std::string> devices) {
  std::set<torch::lazy::BackendDevice> wait_devices;
  if (!devices.empty()) {
    for (auto& device_str : devices) {
      wait_devices.insert(ParseDeviceString(device_str));
    }
  } else {
    for (auto& device_str : xla::ComputationClient::Get()->GetLocalDevices()) {
      wait_devices.insert(ParseDeviceString(device_str));
    }
  }
  // The LockDevices() API returns a vector of xla::util::ExceptionCleanup
  // object, which is going to be freed immediately, turning this operation
  // into a lock barrier.
  LockDevices(wait_devices);
}

XLATensor::OpByOpAsync XLATensor::SyncTensorsGraphOpByOp(
    std::vector<XLATensorPtr>* tensors, absl::Span<const std::string> devices,
    const SyncTensorsConfig& config) {
  struct Async {
    explicit Async(SyncTensorCollection coll,
                   std::vector<torch::lazy::BackendDataPtr> tensors_data,
                   std::vector<torch::lazy::Value> roots,
                   absl::Span<const std::string> devices)
        : coll(std::move(coll)),
          tensors_data(std::move(tensors_data)),
          roots(std::move(roots)),
          devices(devices.begin(), devices.end()) {}

    SyncTensorCollection coll;
    std::vector<torch::lazy::BackendDataPtr> tensors_data;
    std::vector<torch::lazy::Value> roots;
    std::vector<std::string> devices;
  };

  SyncTensorCollection coll = CollectSyncTensors(*tensors, config);
  DebugUtil::SaveTensorsGraphInfo("SyncTensorsGraphOpByOp", *tensors,
                                  &coll.indices);

  std::vector<torch::lazy::Value> roots = CollectRoots(*tensors, coll.indices);
  std::vector<torch::lazy::Value> ir_values;
  std::vector<torch::lazy::BackendDataPtr> tensor_data_vec;
  ExtractIRAndPrepareXlaData_(tensors, coll.config, coll.indices, ir_values,
                              tensor_data_vec);
  auto tensors_data =
      SetTensorData(tensors, coll.config, coll.indices, tensor_data_vec);
  TensorCollectionBarrier(&coll);
  auto async = std::make_shared<Async>(std::move(coll), std::move(tensors_data),
                                       std::move(roots), devices);
  auto syncfn = [async]() -> int {
    try {
      TF_VLOG(3) << "Executing (OpByOp) IR graph hash "
                 << torch::lazy::HashToString(async->coll.hash) << " on device "
                 << async->coll.device << " ...";
      std::vector<torch::lazy::BackendDataPtr> results =
          OpByOpExecutor::Get()->Execute(
              async->roots, async->coll.device.toString(), async->devices);
      TF_VLOG(3) << "Executing (OpByOp) IR graph hash "
                 << torch::lazy::HashToString(async->coll.hash) << " on device "
                 << async->coll.device << " done!";

      for (size_t i = 0; i < results.size(); ++i) {
        if (async->tensors_data[i] != nullptr) {
          async->tensors_data[i]->Assign(*results[i]);
        }
      }
    } catch (...) {
      std::exception_ptr exptr = std::current_exception();
      for (auto& unlocker : async->coll.unlocker) {
        unlocker.SetStatus(exptr);
      }
      throw;
    }
    return 0;
  };
  OpByOpAsync async_op(std::move(syncfn));
  return async_op.Schedule();
}

std::vector<std::pair<int64_t, int64_t>> XLATensor::BuildInputOutputAliases(
    const std::vector<XLATensorPtr>& tensors, absl::Span<const size_t> indices,
    LoweringContext* lowering_ctx) {
  std::unordered_map<int64_t, size_t> output_tensor_id_map;
  std::vector<std::pair<int64_t, int64_t>> input_output_alias_pair;
  for (size_t i = 0; i < indices.size(); ++i) {
    size_t tensor_index = indices[i];
    int64_t tensor_id = tensors[tensor_index]->GetUniqueId();
    output_tensor_id_map[tensor_id] = i;
  }
  // TODO we need xla_shape here.
  const auto& parameters_data = lowering_ctx->GetParametersData();
  std::vector<ssize_t> alias_map(indices.size(), -1);
  for (size_t i = 0; i < parameters_data.size(); ++i) {
    auto* data_info =
        static_cast<torch::lazy::LazyGraphExecutor::DeviceDataInfo*>(
            parameters_data[i]->info());
    if (data_info != nullptr && !data_info->read_only) {
      auto it = output_tensor_id_map.find(data_info->tensor_id);
      if (it != output_tensor_id_map.end()) {
        size_t output_index = it->second;
        xla::XlaOp root = lowering_ctx->GetResult(output_index);
        const xla::Shape& root_shape = XlaHelpers::ShapeOfXlaOp(root);
        auto parameter_data_shape = UnwrapXlaData(parameters_data[i])->shape();
        if (parameter_data_shape == root_shape && alias_map[output_index] < 0) {
          // parameter is not a tuple so param_index will always be {}
          lowering_ctx->builder()->SetUpAlias(
              {/*output_index=*/static_cast<int64_t>(output_index)},
              /*param_number=*/i, /*param_index=*/{});
          alias_map[output_index] = i;
          input_output_alias_pair.push_back(std::make_pair(i, output_index));

          TF_VLOG(6) << "Aliased paramter " << i << " with output "
                     << output_index << ": " << parameter_data_shape;
        }
      }
    }
  }
  XLA_VALUE_METRIC("InputOutputAliasCount", alias_map.size());
  return input_output_alias_pair;
}

XLATensor::CompilationResult XLATensor::Compile(
    const std::vector<XLATensorPtr>& tensors,
    absl::Span<const std::string> devices, const SyncTensorCollection& coll,
    PostOrderData* po_data, const std::vector<torch::lazy::Value>& ir_values) {
  tensorflow::profiler::TraceMe activity(
      [&] {
        return tensorflow::profiler::TraceMeEncode(
            "XLATensor::Compile",
            {{"graph_hash", torch::lazy::HashToString(coll.hash)}});
      },
      tensorflow::profiler::TraceMeLevel::kInfo);
  static const bool enable_aliasing =
      xla::sys_util::GetEnvBool("XLA_ENABLE_PARAM_ALIASING", true);
  static const size_t parameter_wrapping_threadshold =
      xla::sys_util::GetEnvInt("XLA_PARAMETER_WRAPPING_THREADSHOLD", 3200);
  static const bool using_pjrt =
      xla::sys_util::GetEnvString("PJRT_DEVICE", "").size() > 0;
  LoweringContext lowering_ctx("SyncTensorsGraph", coll.device,
                               po_data->post_order,
                               std::move(po_data->emission_map));
  for (auto ir_value : ir_values) {
    xla::XlaOp root = lowering_ctx.GetOutputOp(
        torch::lazy::Output(ir_value.node.get(), ir_value.index));
    lowering_ctx.AddResult(root);
  }
  // Annotate HLO sharding selectively in the compuation.
  bool is_sharded = ShardingUtil::SetHloSharding(&lowering_ctx);

  std::vector<std::pair<int64_t, int64_t>> input_output_alias_pair;
  // TODO(yeounoh) aliasing is disabled for partitioned computation,
  // since the current aliasing compares the unpartitioned input and output
  // shapes which can lead to an incorrect aliasing pairs if sharded.
  if (enable_aliasing && coll.config.sync_xla_data && !is_sharded) {
    // We can only alias at the step barrier, when force_xla_data is true.
    // Consider the case:
    //   1. Tensor A(DEVICE_DATA)
    //   2. Tensor B = A + 0.9
    //   3. A += 0.4
    // If we activate aliasing for A's graph, and we do:
    //   print(A)
    //   print(A)
    // The first print will update DEVICE_DATA' with DEVICE_DATA+0.4, and the
    // second print will again update DEVICE_DATA" with DEVICE_DATA'+0.4, which
    // will lead to incorrect results.
    // We cannot normally turn A's state into DEVICE_DATA, as if any of the
    // sources is a view, this will not lead to correct results (as A's value
    // taken at different times need to reflect view source changes):
    //   1. Tensor A = some_graph_with_view_source(V)
    //   2. print(A)
    //   3. V += 1
    //   4. print(A)
    // The second print should reflect the new value due to V's changes.
    // Also in the first example, unless we are doing a step barrier and hence
    // include all live tensors, if the B value is not part of the graph, it
    // will later fetch the new value of A, which is incorrect.
    // But, when we issue a step barrier (force_xla_data == true) we have to
    // turn everything into DEVICE_DATA, so we can activate aliasing.
    input_output_alias_pair =
        BuildInputOutputAliases(tensors, coll.indices, &lowering_ctx);
  }

  xla::XlaComputation computation = ConsumeValue(lowering_ctx.BuildXla());
  xla::ProgramShape program_shape = ConsumeValue(computation.GetProgramShape());

  bool should_wrap_parameter =
      (program_shape.parameters_size() >= parameter_wrapping_threadshold) &&
      using_pjrt;
  if (should_wrap_parameter) {
    TF_VLOG(3) << "Wrapping graph with " << program_shape.parameters_size()
               << " parameters. Threadshold = "
               << parameter_wrapping_threadshold;
    computation = ConsumeValue(XlaHelpers::WrapXlaComputation(
        computation, program_shape.parameters(), input_output_alias_pair));
    program_shape = ConsumeValue(computation.GetProgramShape());
  }
  xla::Shape shape = MakeShapeWithDeviceLayout(
      program_shape.result(), static_cast<XlaDeviceType>(coll.device.type()));

  std::vector<xla::ComputationClient::CompileInstance> instances;
  instances.push_back({std::move(computation), coll.device.toString(),
                       xla::ComputationClient::Get()->GetCompilationDevices(
                           coll.device.toString(), devices),
                       &shape, should_wrap_parameter, is_sharded});

  TF_VLOG(3) << "Compiling IR graph hash "
             << torch::lazy::HashToString(coll.hash) << " on device "
             << coll.device << " ...";
  std::vector<std::shared_ptr<xla::ComputationClient::Computation>>
      computations =
          xla::ComputationClient::Get()->Compile(std::move(instances));
  TF_VLOG(3) << "Compiling IR graph hash "
             << torch::lazy::HashToString(coll.hash) << " on device "
             << coll.device << " done!";
  TF_VLOG(5)
      << "Graph hash " << torch::lazy::HashToString(coll.hash)
      << " is computation hash "
      << torch::lazy::HashToString(torch::lazy::Hash(
             computations.front()->computation().proto().SerializeAsString()));
  if (should_wrap_parameter) {
    XLA_CHECK_EQ(program_shape.parameters_size(), 1);
    XLA_CHECK_EQ(program_shape.parameters()[0].tuple_shapes_size(),
                 po_data->parameters_data.size());
  } else {
    XLA_CHECK_EQ(program_shape.parameters_size(),
                 po_data->parameters_data.size());
  }

  return {/*device=*/coll.device,
          /*emitted_nodes=*/lowering_ctx.GetEmittedNodeCount(),
          /*computation=*/
          std::make_shared<Computation>(std::move(computations.front())),
          /*parameters_data=*/std::move(po_data->parameters_data),
          /*is_sharded=*/is_sharded};
}

std::shared_ptr<XLATensor::Async> XLATensor::SyncTensorsGraphInternal(
    std::vector<XLATensorPtr>* tensors, absl::Span<const std::string> devices,
    const SyncTensorsConfig& config) {
  tensorflow::profiler::TraceMe activity(
      "SyncTensorsGraphInternal", tensorflow::profiler::TraceMeLevel::kInfo);
  SyncTensorCollection coll = CollectSyncTensors(*tensors, config);
  if (coll.indices.empty()) {
    /* Enure previous execution is complete before exiting this
     * function */
    TensorCollectionBarrier(&coll);
    return nullptr;
  }
  DebugUtil::SaveTensorsGraphInfo("ScheduleSyncTensorsGraph", *tensors,
                                  &coll.indices);
  std::vector<torch::lazy::Value> ir_values;
  std::vector<torch::lazy::BackendDataPtr> tensor_data_vec;
  ExtractIRAndPrepareXlaData_(tensors, coll.config, coll.indices, ir_values,
                              tensor_data_vec);
  PostOrderData po_data = RunPostOrder(ir_values, &coll);

  coll.hash = torch::lazy::HashCombine(
      coll.hash, torch::lazy::Hash(po_data.parameter_sequence));
  TF_VLOG(4) << "Parameter sequence graph hash "
             << torch::lazy::HashToString(coll.hash);
  std::shared_ptr<Async> async =
      TryRunCachedSync(tensors, &coll, &po_data, tensor_data_vec);
  if (async != nullptr) {
    return async;
  }

  CompilationResult compile_result =
      Compile(*tensors, devices, coll, &po_data, ir_values);

  XLA_VALUE_METRIC("TensorsGraphSize", compile_result.emitted_nodes);
  TF_VLOG(5) << "TensorsGraphSize=" << compile_result.emitted_nodes;

  auto cached_computation = std::make_shared<CachedComputation>(
      std::move(compile_result.computation), compile_result.is_sharded);
  GetComputationCache()->Add(coll.hash, cached_computation);

  return ScheduleSyncTensorsGraph(
      tensors, &coll, std::move(compile_result.parameters_data),
      compile_result.device.toString(), std::move(cached_computation),
      tensor_data_vec);
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
  torch::lazy::BackendDataPtr xla_data = CurrentXlaData();
  if (xla_data != nullptr) {
    return UnwrapXlaData(xla_data)->GetOpaqueHandle();
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
