#include "tensor.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <mutex>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "data_ops.h"
#include "helpers.h"
#include "lowering_context.h"
#include "ops/arithmetic_ir_ops.h"
#include "ops/avg_pool2d.h"
#include "ops/avg_pool2d_backward.h"
#include "ops/conv2d.h"
#include "ops/conv2d_backward.h"
#include "ops/cross_replica_sum.h"
#include "ops/device_data.h"
#include "ops/generic.h"
#include "ops/infer_output_shape.h"
#include "ops/max_pool2d.h"
#include "ops/max_pool2d_backward.h"
#include "ops/max_pool2d_indices.h"
#include "ops/ops.h"
#include "ops/scalar.h"
#include "ops/softmax.h"
#include "ops/softmax_backward.h"
#include "ops/threshold.h"
#include "ops/threshold_backward.h"
#include "ops/view.h"
#include "tensor_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"
#include "tensorflow/compiler/xla/xla_client/thread_pool.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "torch/csrc/autograd/variable.h"
#include "translator.h"

namespace torch_xla {
namespace {

void SetMulti(std::vector<XLATensor>* dest_tuple,
              std::vector<std::shared_ptr<xla::ComputationClient::Data>>
                  new_dest_elements,
              const std::vector<size_t>& index_mapping) {
  XLA_CHECK_EQ(index_mapping.size(), new_dest_elements.size());
  // Replace the underlying data for the destination tensors with the data in
  // "new_dest_elements".
  for (size_t i = 0; i < new_dest_elements.size(); ++i) {
    size_t dest_tuple_index = index_mapping[i];
    // Prefer not to make SetXlaData() non-const.
    (*dest_tuple)[dest_tuple_index].SetXlaData(std::move(new_dest_elements[i]));
  }
}

}  // namespace

// The tensors arena tracks all the XLA tensors which are currently live. This
// is used to create XLA computation "barriers" in order to flush pending
// operations and ensure the same XLA computations are created during the
// training loops.
class XLATensor::TensorsArena {
 public:
  static TensorsArena* Get() {
    static TensorsArena* arena = new TensorsArena();
    return arena;
  }

  void RegisterTensor(std::shared_ptr<Data> data) {
    std::lock_guard<std::mutex> lock(lock_);
    tensors_data_.emplace(data.get(), data);
  }

  void UnregisterTensor(Data* data) {
    std::lock_guard<std::mutex> lock(lock_);
    tensors_data_.erase(data);
  }

  std::vector<XLATensor> GetTensors() {
    std::vector<std::shared_ptr<Data>> data_pointers;
    {
      std::lock_guard<std::mutex> lock(lock_);
      for (auto& ptr_wptr : tensors_data_) {
        std::shared_ptr<Data> data = ptr_wptr.second.lock();
        if (data != nullptr) {
          data_pointers.push_back(std::move(data));
        }
      }
    }
    std::vector<XLATensor> tensors;
    for (auto& data : data_pointers) {
      tensors.push_back(XLATensor(std::move(data)));
    }
    return tensors;
  }

 private:
  std::mutex lock_;
  std::map<Data*, std::weak_ptr<Data>> tensors_data_;
};

XLATensor::Data::~Data() { TensorsArena::Get()->UnregisterTensor(this); }

XLATensor XLATensor::Create(const at::Tensor& tensor, const Device& device,
                            bool requires_grad) {
  XLATensor xtensor(tensor, device, requires_grad);
  TensorsArena::Get()->RegisterTensor(xtensor.data_ptr());
  return xtensor;
}

XLATensor XLATensor::Create(
    std::shared_ptr<xla::ComputationClient::Data> xla_data,
    bool requires_grad) {
  XLATensor xtensor(std::move(xla_data), requires_grad);
  TensorsArena::Get()->RegisterTensor(xtensor.data_ptr());
  return xtensor;
}

XLATensor XLATensor::Create(ir::Value ir_node, const Device& device) {
  XLATensor xtensor(std::move(ir_node), device);
  TensorsArena::Get()->RegisterTensor(xtensor.data_ptr());
  return xtensor;
}

XLATensor XLATensor::Create(std::shared_ptr<View> view, const Device& device) {
  XLATensor xtensor(std::move(view), device);
  TensorsArena::Get()->RegisterTensor(xtensor.data_ptr());
  return xtensor;
}

XLATensor::XLATensor(const at::Tensor& tensor, const Device& device,
                     bool requires_grad)
    : data_(std::make_shared<Data>(tensor, device)) {
  data()->requires_grad = requires_grad;
}

XLATensor::XLATensor(std::shared_ptr<xla::ComputationClient::Data> xla_data,
                     bool requires_grad)
    : data_(std::make_shared<Data>(xla_data, Device(xla_data->device()))) {
  data()->requires_grad = requires_grad;
}

XLATensor::XLATensor(ir::Value ir_node, const Device& device)
    : data_(std::make_shared<Data>(std::move(ir_node), device)) {
  TryLimitGraphSize();
}

XLATensor::XLATensor(std::shared_ptr<View> view, const Device& device)
    : data_(std::make_shared<Data>(std::move(view), device)) {}

XLATensor::XLATensor(std::shared_ptr<Data> data) : data_(std::move(data)) {}

XLATensor::Data* XLATensor::data() const {
  XLA_CHECK(data_ != nullptr) << "Trying to access a null cursor";
  return data_.get();
}

c10::optional<XLATensor> XLATensor::grad() const {
  if (data()->grad == nullptr) {
    return c10::nullopt;
  }
  return *data()->grad;
}

void XLATensor::SetGradient(const XLATensor& grad) {
  if (data()->grad == nullptr) {
    data()->grad = std::make_shared<XLATensor>(grad);
  } else {
    data()->grad->ReferenceDataFrom(grad);
  }
}

at::ScalarType XLATensor::dtype() const {
  return TensorTypeFromXlaType(shape().get().element_type());
}

xla::util::MaybeRef<xla::Shape> XLATensor::shape() const {
  if (data()->view != nullptr) {
    return data()->view->shape;
  }
  if (data()->xla_data != nullptr) {
    return data()->xla_data->shape();
  }
  const Device& device = GetDevice();
  if (data()->ir_node) {
    const xla::Shape& node_shape = data()->ir_node.shape();
    return MakeArrayShapeFromDimensions(
        node_shape.dimensions(), node_shape.element_type(), device.hw_type);
  }
  XLA_CHECK(data()->tensor_data);
  return MakeArrayShapeFromDimensions(
      data()->tensor_data->sizes(),
      XlaHelpers::MakeXlaPrimitiveType(data()->tensor_data->type().scalarType(),
                                       &device),
      device.hw_type);
}

const Device& XLATensor::GetDevice() const { return data()->device; }

xla::int64 XLATensor::GetUniqueId() const { return data()->unique_id; }

std::shared_ptr<xla::ComputationClient::Data> XLATensor::GetXlaData() {
  bool up_to_date = true;
  if (data()->view != nullptr) {
    ViewIrNode ir_node_updated = GetViewIrNode(data()->view.get());
    if (ir_node_updated.updated || !data()->ir_node) {
      up_to_date = false;
      data()->ir_node = ir_node_updated.ir_node;
    }
  }
  if (up_to_date) {
    std::shared_ptr<xla::ComputationClient::Data> xla_data = CurrentXlaData();
    if (xla_data != nullptr) {
      return xla_data;
    }
  }
  if (data()->ir_node) {
    ApplyPendingGraph();
  } else {
    XLA_CHECK(data()->tensor_data);
    data()->xla_data = TensorToXlaData(*data()->tensor_data, GetDevice());
  }
  return data()->xla_data;
}

std::shared_ptr<xla::ComputationClient::Data> XLATensor::CurrentXlaData()
    const {
  if (data()->xla_data != nullptr) {
    // When we set a new Node for a tensor, we leave the XLA data pointer alive,
    // as it is needed in order for the cached tensor apply operation to work.
    // See comment in the SetIrNode() API.
    // In order to verify that that data is still valid as far as current tensor
    // data POV, we need to verify that the eventual IR Node is a DeviceData
    // node, and that its ComputationClient data pointer matches.
    ir::Value ir_node = CurrentIrNode();
    if (!ir_node) {
      // If there is no IR node, then the XLA data is valid.
      return data()->xla_data;
    }
    const ir::ops::DeviceData* device_data =
        dynamic_cast<const ir::ops::DeviceData*>(ir_node.node.get());
    if (device_data != nullptr &&
        device_data->data().get() == data()->xla_data.get()) {
      return data()->xla_data;
    }
  }
  return nullptr;
}

std::string XLATensor::DumpGraphNodeComputation() const {
  std::string hlo_text;
  ir::Value ir_node = CurrentIrNode();
  if (ir_node) {
    ir::LoweringContext lowering_ctx("DumpGraphNodeComputation");
    xla::XlaOp root = lowering_ctx.GetOutputOp(ir_node);
    auto computation = lowering_ctx.Build(root).ConsumeValueOrDie();
    hlo_text =
        xla::xrt_util::GetComputationHloText(computation).ConsumeValueOrDie();
  }
  return hlo_text;
}

void XLATensor::SetXlaData(
    std::shared_ptr<xla::ComputationClient::Data> xla_data) {
  XLA_CHECK(xla::ShapeUtil::Equal(shape(), xla_data->shape()))
      << shape() << " vs " << xla_data->shape() << "\n"
      << DumpGraphNodeComputation();
  data()->xla_data = std::move(xla_data);
  data()->ir_node = ir::Value();
  data()->tensor_data = c10::nullopt;
}

void XLATensor::SetIrNode(ir::Value ir_node) {
  if (data()->view != nullptr) {
    // If we have an active view, and a SetIrNode() happens, it means we are
    // within an in-place execution context, and we need to update the view's
    // alias as well.
    data()->view->alias->ir_node = ir_node;
  }
  // We do not want to nullify that XLA data pointer here, as otherwise the
  // tensor apply computation caching will not work correctly.
  // If A is a tensor, a typical optimizer step computation will do:
  //  A' = F(A)
  // The cached apply computation will want to find the previous XLA data for
  // A's unique ID (as that data will be input to F()), but if setting A's IR
  // node nullify that, it will not be found.
  // We do have logic in CurrentXlaData() to verify that the XLA data pointer is
  // actually valid, as far as tensor value goes.
  data()->ir_node = std::move(ir_node);
  data()->tensor_data = c10::nullopt;
  TryLimitGraphSize();
}

void XLATensor::TryLimitGraphSize() {
  // If we are accumulating too many nodes in the pending graph, render the XLA
  // by executing the pending graph.
  static const size_t kMaxPendingGraphSize = 1000;
  if (data()->ir_node && data()->ir_node->graph_size() > kMaxPendingGraphSize) {
    ApplyPendingGraph();
  }
}

ir::Value XLATensor::GetIrNode() const {
  ir::Value ir_node = CurrentIrNode();
  if (ir_node) {
    return ir_node;
  }
  std::shared_ptr<xla::ComputationClient::Data> xla_data = CurrentXlaData();
  if (xla_data != nullptr) {
    // In case of tensor node, we do not clear the XLA data when we set the IR
    // node. This because we want further calls to GetIrNode() to fetch the same
    // IR node, and not create new ones (even though the lowering context will
    // still collapse them all into a single XLA parameter op).
    // So call which wants the XLA data will still find it, w/out having to
    // fetch it via a computation client from-server call.
    data()->ir_node = CreateTensorNode(xla_data);
    return data()->ir_node;
  }
  const c10::optional<at::Tensor>& tensor_data = CurrentTensorData();
  XLA_CHECK(tensor_data);
  // Now we have a tensor data. Do we force the creation of device memory, or we
  // generate an IR Node Constant for it?
  // TODO: For now force device data, but considerations about tensor size could
  // drive different logic.
  data()->xla_data = TensorToXlaData(*tensor_data, GetDevice());
  data()->ir_node = CreateTensorNode(data()->xla_data);
  return data()->ir_node;
}

ir::Value XLATensor::CurrentIrNode() const {
  if (data()->view != nullptr) {
    return GetViewIrNode(data()->view.get()).ir_node;
  }
  return data()->ir_node;
}

void XLATensor::SetTensorData(at::Tensor tensor_data) {
  data()->tensor_data = std::move(tensor_data);
}

const c10::optional<at::Tensor>& XLATensor::CurrentTensorData() const {
  return data()->tensor_data;
}

void XLATensor::ReferenceDataFrom(const XLATensor& source) {
  XLA_CHECK_EQ(data()->device, source.data()->device);
  XLA_CHECK(xla::ShapeUtil::Equal(shape(), source.shape()))
      << shape() << " vs " << source.shape();

  data()->xla_data = source.data()->xla_data;
  data()->ir_node = source.data()->ir_node;
  data()->tensor_data = source.data()->tensor_data;
}

std::vector<int64_t> XLATensor::DimensionSizes() const {
  auto tensor_shape = shape();
  return xla::util::ToVector<int64_t>(tensor_shape.get().dimensions());
}

at::Tensor XLATensor::ToTensor() {
  c10::optional<at::Tensor> tensor_data = CurrentTensorData();
  if (!tensor_data) {
    // The GetXlaData() call will trigger an ApplyPendingGraph() if an IR Node
    // is available on the tensor.
    std::vector<xla::Literal> literals =
        xla::ComputationClient::Get()->TransferFromServer({GetXlaData()});
    tensor_data = MakeTensorFromXlaLiteral(literals.front());
    SetTensorData(*tensor_data);
  }
  return *tensor_data;
}

void XLATensor::DiscardXlaData() {
  XLA_CHECK(data()->tensor_data);
  data()->xla_data = nullptr;
  data()->ir_node = ir::Value();
  data()->view = nullptr;
}

at::Tensor XLATensor::ToMutableTensor() {
  at::Tensor tensor_data = ToTensor();
  // In case of the ATEN Tensor data being possibly dirty, we do clear both the
  // IR Node and the XLA data. This API will be called to feed the tensor data
  // to ATEN APIs, and when we get to that point, we already lost the full XLA
  // fusion deal (and hence we do not need to keep the XLA data around for
  // caching computations).
  DiscardXlaData();
  return tensor_data;
}

std::vector<XLATensor> XLATensor::GetLiveTensors() {
  return TensorsArena::Get()->GetTensors();
}

std::vector<at::Tensor> XLATensor::GetTensors(
    std::vector<XLATensor>* tensors, const std::vector<bool>* writeable) {
  // TODO(dlibenzi): We do apply/compute and then fetch. Changing the API to
  // support getting handles and data might save a few pennies here.
  ApplyPendingGraph(tensors, /*apply_context=*/nullptr);

  std::vector<std::shared_ptr<xla::ComputationClient::Data>> tensors_data;
  for (auto& tensor : *tensors) {
    if (!tensor.CurrentTensorData()) {
      tensors_data.push_back(tensor.GetXlaData());
    }
  }
  std::vector<xla::Literal> literals =
      xla::ComputationClient::Get()->TransferFromServer(tensors_data);
  std::vector<at::Tensor> results;
  size_t literals_index = 0;
  results.reserve(tensors->size());
  for (size_t i = 0; i < tensors->size(); ++i) {
    const c10::optional<at::Tensor>& tensor_data =
        (*tensors)[i].CurrentTensorData();
    if (tensor_data) {
      results.push_back(*tensor_data);
    } else {
      XLA_CHECK_LT(literals_index, literals.size());
      results.push_back(MakeTensorFromXlaLiteral(literals[literals_index]));
      ++literals_index;
    }
  }
  if (writeable != nullptr) {
    XLA_CHECK_EQ(tensors->size(), writeable->size());
    for (size_t i = 0; i < tensors->size(); ++i) {
      if ((*writeable)[i]) {
        (*tensors)[i].DiscardXlaData();
      }
    }
  }
  return results;
}

std::vector<XLATensor> XLATensor::CreateTensors(
    const std::vector<at::Tensor>& tensors,
    const std::vector<std::string>& devices) {
  std::vector<std::shared_ptr<xla::ComputationClient::Data>> handles =
      CreateTensorsData(tensors, devices);
  std::vector<XLATensor> xla_tensors;
  for (size_t i = 0; i < handles.size(); ++i) {
    xla_tensors.push_back(
        Create(std::move(handles[i]), tensors[i].requires_grad()));
  }
  return xla_tensors;
}

ir::Value XLATensor::CreateTensorNode(
    std::shared_ptr<xla::ComputationClient::Data> data) {
  return ir::ops::DeviceDataOp(std::move(data));
}

xla::int64 XLATensor::GetNextTensorId() {
  static std::atomic<xla::int64>* id_generator = new std::atomic<xla::int64>(1);
  return id_generator->fetch_add(1);
}

XLATensor::ViewIrNode XLATensor::GetViewIrNode(View* view) {
  if (view->ir_node &&
      view->ir_node->operand(0).node == view->alias->ir_node.node.get()) {
    // If the existing ir_node (which is a ir::ops::View) operand(0) still
    // matches the current aliased node, the current IR Node is still valid.
    return {view->ir_node, false};
  }
  view->ir_node = ir::MakeNode<ir::ops::View>(view->alias->ir_node,
                                              view->shape.dimensions());
  return {view->ir_node, true};
}

XLATensor XLATensor::add(const XLATensor& other,
                         const at::Scalar& alpha) const {
  ir::NodePtr constant = ir::ops::ScalarOp(alpha.toDouble(), other.shape());
  return Create(GetIrNode() + other.GetIrNode() * constant, data()->device);
}

void XLATensor::add_(const XLATensor& other, const at::Scalar& alpha) {
  ir::NodePtr constant = ir::ops::ScalarOp(alpha.toDouble(), other.shape());
  SetIrNode(GetIrNode() + other.GetIrNode() * constant);
}

XLATensor XLATensor::mul(const XLATensor& other) const {
  return Create(GetIrNode() * other.GetIrNode(), data()->device);
}

XLATensor XLATensor::mul(const at::Scalar& other) const {
  ir::NodePtr constant = ir::ops::ScalarOp(other.toDouble(), shape());
  return Create(GetIrNode() * constant, data()->device);
}

void XLATensor::mul_(const XLATensor& other) {
  SetIrNode(GetIrNode() * other.GetIrNode());
}

void XLATensor::mul_(const at::Scalar& other) {
  ir::NodePtr constant = ir::ops::ScalarOp(other.toDouble(), shape());
  SetIrNode(GetIrNode() * constant);
}

XLATensor XLATensor::div(const XLATensor& other) const {
  return Create(GetIrNode() / other.GetIrNode(), data()->device);
}

XLATensor XLATensor::div(const at::Scalar& other) const {
  ir::NodePtr constant = ir::ops::ScalarOp(other.toDouble(), shape());
  return Create(GetIrNode() / constant, data()->device);
}

void XLATensor::div_(const XLATensor& other) {
  SetIrNode(GetIrNode() / other.GetIrNode());
}

void XLATensor::div_(const at::Scalar& other) {
  ir::NodePtr constant = ir::ops::ScalarOp(other.toDouble(), shape());
  SetIrNode(GetIrNode() / constant);
}

void XLATensor::zero_() { SetIrNode(ir::ops::ScalarOp(0.0, shape())); }

void XLATensor::addcdiv_(const at::Scalar& value, const XLATensor& tensor1,
                         const XLATensor& tensor2) {
  ir::NodePtr constant =
      ir::ops::ScalarOp(value.toDouble(), tensor1.shape().get().element_type());
  ir::Value div = tensor1.GetIrNode() / tensor2.GetIrNode();
  SetIrNode(GetIrNode() + div * constant);
}

void XLATensor::addcmul_(const at::Scalar& value, const XLATensor& tensor1,
                         const XLATensor& tensor2) {
  ir::NodePtr constant =
      ir::ops::ScalarOp(value.toDouble(), tensor1.shape().get().element_type());
  ir::Value mul = tensor1.GetIrNode() * tensor2.GetIrNode();
  SetIrNode(GetIrNode() + mul * constant);
}

xla::int64 XLATensor::size(int dim) const {
  auto xla_shape = shape();
  int rank = xla_shape.get().dimensions_size();
  int min_shape_dim = -rank;
  int max_shape_dim = rank - 1;
  XLA_CHECK(min_shape_dim <= dim && dim <= max_shape_dim) << absl::StrCat(
      "Dimension out of range (expected to be in range of [", min_shape_dim,
      ", ", max_shape_dim, "], but got ", dim, ")");
  int dim_index = dim < 0 ? rank + dim : dim;
  XLA_CHECK_GE(dim_index, 0);
  XLA_CHECK_LT(dim_index, rank);
  return xla_shape.get().dimensions(dim_index);
}

XLATensor XLATensor::relu() const {
  return Create(ir::ops::ReluOp(GetIrNode()), GetDevice());
}

XLATensor XLATensor::threshold(float threshold, float value) const {
  return Create(ir::MakeNode<ir::ops::Threshold>(GetIrNode(), threshold, value),
                GetDevice());
}

XLATensor XLATensor::conv2d(
    const XLATensor& weight, const XLATensor& bias,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding,
    bool use_full_conv_precision) const {
  ir::NodePtr ir_node = ir::MakeNode<ir::ops::Conv2d>(
      GetIrNode(), weight.GetIrNode(), bias.GetIrNode(), stride, padding,
      use_full_conv_precision);
  return Create(ir_node, GetDevice());
}

XLATensor XLATensor::conv2d(
    const XLATensor& weight,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding,
    bool use_full_conv_precision) const {
  ir::NodePtr ir_node =
      ir::MakeNode<ir::ops::Conv2d>(GetIrNode(), weight.GetIrNode(), stride,
                                    padding, use_full_conv_precision);
  return Create(ir_node, GetDevice());
}

XLATensor XLATensor::addmm(const XLATensor& weight, const XLATensor& bias,
                           bool use_full_conv_precision) const {
  return Create(ir::ops::AddMatMulOp(GetIrNode(), weight.GetIrNode(),
                                     bias.GetIrNode(), use_full_conv_precision),
                GetDevice());
}

XLATensor XLATensor::max_pool2d(
    tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding) const {
  return Create(ir::MakeNode<ir::ops::MaxPool2d>(GetIrNode(), kernel_size,
                                                 stride, padding),
                GetDevice());
}

XLATensor XLATensor::max_pool2d_indices(
    tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding) const {
  return Create(ir::MakeNode<ir::ops::MaxPool2dIndices>(
                    GetIrNode(), kernel_size, stride, padding),
                GetDevice());
}

XLATensor XLATensor::avg_pool2d(
    tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding,
    bool count_include_pad) const {
  return Create(
      ir::MakeNode<ir::ops::AvgPool2d>(GetIrNode(), kernel_size, stride,
                                       padding, count_include_pad),
      GetDevice());
}

XLATensor XLATensor::mm(const XLATensor& input, const XLATensor& weight,
                        bool use_full_conv_precision) {
  return Create(ir::ops::MatMulOp(input.GetIrNode(), weight.GetIrNode(),
                                  use_full_conv_precision),
                input.GetDevice());
}

XLATensor XLATensor::avg_pool2d_backward(
    const XLATensor& out_backprop, const XLATensor& input,
    tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding,
    bool count_include_pad) {
  return Create(ir::MakeNode<ir::ops::AvgPool2dBackward>(
                    out_backprop.GetIrNode(), input.GetIrNode(), kernel_size,
                    stride, padding, count_include_pad),
                out_backprop.GetDevice());
}

XLATensor XLATensor::max_pool2d_backward(
    const XLATensor& out_backprop, const XLATensor& input,
    tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding) {
  return Create(ir::MakeNode<ir::ops::MaxPool2dBackward>(
                    out_backprop.GetIrNode(), input.GetIrNode(), kernel_size,
                    stride, padding),
                out_backprop.GetDevice());
}

std::tuple<XLATensor, XLATensor, XLATensor> XLATensor::conv2d_backward(
    const XLATensor& out_backprop, const XLATensor& input,
    const XLATensor& weight,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding,
    bool use_full_conv_precision) {
  ir::NodePtr node = ir::MakeNode<ir::ops::Conv2dBackward>(
      out_backprop.GetIrNode(), input.GetIrNode(), weight.GetIrNode(), stride,
      padding, use_full_conv_precision);
  XLATensor grad_input = Create(ir::Value(node, 0), out_backprop.GetDevice());
  XLATensor grad_weight = Create(ir::Value(node, 1), out_backprop.GetDevice());
  XLATensor grad_bias = Create(ir::Value(node, 2), out_backprop.GetDevice());
  return std::make_tuple(std::move(grad_input), std::move(grad_weight),
                         std::move(grad_bias));
}

XLATensor XLATensor::log_softmax_backward(const XLATensor& grad_output,
                                          const XLATensor& output,
                                          xla::int64 dim) {
  return Create(ir::MakeNode<ir::ops::LogSoftmaxBackward>(
                    grad_output.GetIrNode(), output.GetIrNode(), dim),
                grad_output.GetDevice());
}

XLATensor XLATensor::threshold_backward(const XLATensor& grad_output,
                                        const XLATensor& input,
                                        float threshold) {
  return Create(ir::MakeNode<ir::ops::ThresholdBackward>(
                    grad_output.GetIrNode(), input.GetIrNode(), threshold),
                grad_output.GetDevice());
}

XLATensor XLATensor::t() const {
  return Create(ir::ops::TransposeOp(GetIrNode()), GetDevice());
}

XLATensor XLATensor::view(
    tensorflow::gtl::ArraySlice<const xla::int64> output_size) const {
  if (data()->view != nullptr) {
    // Handle view of a view. This node is already a view, so use the view alias
    // to create the new IR Node.
    std::vector<xla::int64> complete_dimensions =
        GetCompleteShape(output_size, data()->view->shape.dimensions());
    xla::Shape shape = MakeArrayShapeFromDimensions(
        complete_dimensions, data()->view->shape.element_type(),
        GetDevice().hw_type);
    return Create(std::make_shared<View>(std::move(shape), data()->view->alias),
                  GetDevice());
  }
  // This node is not a view, and creating a view forks the current node into
  // becoming one itself. This means creating an alias with the current IR Node,
  // and using the same alias for the created IR Node.
  ir::Value ir_node = GetIrNode();
  std::shared_ptr<Alias> alias = std::make_shared<Alias>(ir_node);
  data()->view = std::make_shared<View>(ir_node.shape(), alias);

  std::vector<xla::int64> complete_dimensions =
      GetCompleteShape(output_size, ir_node.shape().dimensions());
  xla::Shape shape = MakeArrayShapeFromDimensions(
      complete_dimensions, ir_node.shape().element_type(), GetDevice().hw_type);
  return Create(std::make_shared<View>(std::move(shape), alias), GetDevice());
}

XLATensor XLATensor::log_softmax(xla::int64 dim) const {
  return Create(ir::MakeNode<ir::ops::LogSoftmax>(GetIrNode(), dim),
                GetDevice());
}

XLATensor XLATensor::nll_loss(const XLATensor& input, const XLATensor& target) {
  return Create(ir::ops::NllLossOp(input.GetIrNode(), target.GetIrNode()),
                input.GetDevice());
}

XLATensor XLATensor::nll_loss_backward(const XLATensor& input,
                                       const XLATensor& target) {
  return Create(
      ir::ops::NllLossBackwardOp(input.GetIrNode(), target.GetIrNode()),
      input.GetDevice());
}

XLATensor XLATensor::cross_replica_sum(
    const std::vector<std::vector<xla::int64>>& groups) const {
  ir::NodePtr crs = ir::ops::CrossReplicaSumOp(GetIrNode(), groups);
  return Create(std::move(crs), data()->device);
}

void XLATensor::ApplyPendingGraph() {
  // This method is called to ensure that the tensor data is available on
  // device, so that a call to CurrentXlaData() returns a valid pointer.
  if (CurrentXlaData() == nullptr) {
    ir::Value ir_node = CurrentIrNode();
    if (ir_node) {
      ir::LoweringContext lowering_ctx("ApplyPendingGraph");
      xla::XlaOp root = lowering_ctx.GetOutputOp(ir_node);
      xla::XlaComputation computation =
          lowering_ctx.Build(root).ConsumeValueOrDie();
      auto output_shape = shape();
      auto compiled_computation = xla::ComputationClient::Get()->Compile(
          std::move(computation), {GetDevice().ToString()},
          &output_shape.get());
      xla::ComputationClient::ExecuteComputationOptions options;
      options.explode_tuple = false;
      auto results = xla::ComputationClient::Get()->ExecuteComputation(
          *compiled_computation, lowering_ctx.GetParametersData(),
          compiled_computation->devices()[0], options);
      XLA_CHECK_EQ(results.size(), 1);
      SetXlaData(results.front());
    } else {
      // Otherwise it better be having at::Tensor data otherwise it will throw
      // an exception.
      XLA_CHECK(data()->tensor_data);
      data()->xla_data = TensorToXlaData(*data()->tensor_data, GetDevice());
    }
  }
}

std::vector<size_t> XLATensor::GetApplyOrder(
    const std::vector<XLATensor>& tensors) {
  std::vector<at::Tensor> at_tensors;
  std::vector<std::string> devices;
  std::vector<size_t> at_tensor_index;
  std::vector<size_t> order;
  order.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i].CurrentXlaData() == nullptr) {
      if (tensors[i].CurrentIrNode().node != nullptr) {
        // Add only tensors which need to be synced.
        order.push_back(i);
      } else {
        // The tensor only has at::Tensor data. We need to queue it for a device
        // upload.
        const c10::optional<at::Tensor>& tensor_data =
            tensors[i].CurrentTensorData();
        XLA_CHECK(tensor_data);
        at_tensors.push_back(*tensor_data);
        devices.push_back(tensors[i].GetDevice().ToString());
        at_tensor_index.push_back(i);
      }
    }
  }
  if (!at_tensors.empty()) {
    std::vector<std::shared_ptr<xla::ComputationClient::Data>> handles =
        CreateTensorsData(at_tensors, devices);
    for (size_t i = 0; i < handles.size(); ++i) {
      // If we are here, it means that the IR Node for the tensor is nullptr.
      // Also, we uploaded the at::Tensor data to the device, but such data is
      // still valid so we leave it live on the XLA tensor (so that a following
      // ToTensor() does not need to fetch it from device).
      tensors[at_tensor_index[i]].data()->xla_data = std::move(handles[i]);
    }
  }

  // Order the tensors based on their device and unique ID, so that we try to
  // mazimize the chances of creating the same XLA computation, and hence
  // hitting the compilation cache.
  std::sort(order.begin(), order.end(), [&tensors](size_t i1, size_t i2) {
    int device_compare =
        tensors[i1].GetDevice().compare(tensors[i2].GetDevice());
    if (device_compare != 0) {
      return device_compare < 0;
    }
    return tensors[i1].GetUniqueId() < tensors[i2].GetUniqueId();
  });
  return order;
}

bool XLATensor::RunCachedApply(std::vector<XLATensor>* tensors,
                               const ApplyContext& apply_context) {
  // Within the ApplyContext we saved the tensors unique IDs, and here we have
  // to map back the unique IDs to the tensor indices within the tensors vector.
  std::unordered_map<xla::int64, size_t> uid_index_map(tensors->size());
  for (size_t i = 0; i < tensors->size(); ++i) {
    uid_index_map[(*tensors)[i].GetUniqueId()] = i;
  }
  std::vector<std::vector<xla::ComputationClient::Data*>> parameters;
  parameters.reserve(apply_context.devices.size());
  for (auto& device_input_mapping : apply_context.input_mapping) {
    std::vector<xla::ComputationClient::Data*> device_parameters;
    device_parameters.reserve(device_input_mapping.size());
    for (auto uid : device_input_mapping) {
      auto it = uid_index_map.find(uid);
      if (it != uid_index_map.end()) {
        const std::shared_ptr<xla::ComputationClient::Data>& xla_data =
            (*tensors)[it->second].data()->xla_data;
        if (xla_data == nullptr) {
          // If we do not find real device data (we have a cached graph instead)
          // at the given tensor, it means the cached information does not apply
          // anymore.
          XLA_COUNTER("NoTensorDataForUid", 1);
          return false;
        }
        device_parameters.push_back(xla_data.get());
      } else {
        // If we have not found the unique ID of the parameter which is supposed
        // to feed data to the computation, the pending graph context changed,
        // and the apply_context is no more valid.
        XLA_COUNTER("NoTensorUid", 1);
        return false;
      }
    }
    parameters.push_back(std::move(device_parameters));
  }
  std::vector<std::vector<size_t>> index_mapping;
  index_mapping.reserve(apply_context.devices.size());
  for (auto& computation_index_mapping : apply_context.index_mapping) {
    std::vector<size_t> current_index_mapping;
    current_index_mapping.reserve(computation_index_mapping.size());
    for (auto uid : computation_index_mapping) {
      auto it = uid_index_map.find(uid);
      if (it != uid_index_map.end()) {
        current_index_mapping.push_back(it->second);
      } else {
        XLA_COUNTER("NoTensorUidForIndexMapping", 1);
        return false;
      }
    }
    index_mapping.push_back(std::move(current_index_mapping));
  }

  xla::ComputationClient::ExecuteParallelOptions options;
  auto results = xla::ComputationClient::Get()->ExecuteParallel(
      xla::util::GetConstSharedPointers(apply_context.computations), parameters,
      apply_context.devices, options);
  size_t device_index = 0;
  for (auto& computation_tuple_elements : results) {
    SetMulti(tensors, std::move(computation_tuple_elements),
             index_mapping[device_index]);
    ++device_index;
  }
  return true;
}

XLATensor::DataUidMap XLATensor::CreateDataUidMap(
    const std::vector<XLATensor>& tensors) {
  DataUidMap data_uid_map(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    std::shared_ptr<xla::ComputationClient::Data> xla_data =
        tensors[i].data()->xla_data;
    if (xla_data != nullptr) {
      auto it_inserted =
          data_uid_map.emplace(xla_data->unique_id(), tensors[i].GetUniqueId());
      if (!it_inserted.second) {
        // It can happen that two tensors references the same device data.
        // This is due to ReferenceDataFrom() API calls, which we use to
        // update the tensors (inputs, gradients,...) with the new data. In
        // that case select the tensor with lower unique ID (older), as the
        // newer one is very likely the new data provider which will be going
        // away soon (as soon as the last tensor reference will go away).
        it_inserted.first->second = std::min<xla::int64>(
            it_inserted.first->second, tensors[i].GetUniqueId());
        XLA_COUNTER("DuplicatedTensorData", 1);
      }
    }
  }
  return data_uid_map;
}

void XLATensor::ApplyPendingGraph(std::vector<XLATensor>* tensors,
                                  ApplyContext* apply_context) {
  struct DeviceContext {
    DeviceContext() : lowering_ctx("ApplyPendingGraph") {}

    ir::LoweringContext lowering_ctx;
    std::vector<size_t> index_mapping;
  };

  std::vector<size_t> order = GetApplyOrder(*tensors);
  std::vector<xla::int64> uid_order;
  uid_order.reserve(order.size());
  for (auto i : order) {
    uid_order.push_back((*tensors)[i].GetUniqueId());
  }
  DataUidMap data_uid_map;
  if (apply_context != nullptr) {
    // Does it look like the cached context still applies to the new run?
    if (apply_context->uid_order == uid_order &&
        RunCachedApply(tensors, *apply_context)) {
      XLA_COUNTER("CachedApplyGraph", 1);
      return;
    }
    XLA_COUNTER("UncachedApplyGraph", 1);
    data_uid_map = CreateDataUidMap(*tensors);
  }

  std::map<Device, DeviceContext> contexts_map;
  for (auto i : order) {
    DeviceContext* device_context = &contexts_map[(*tensors)[i].GetDevice()];
    device_context->index_mapping.push_back(i);
  }

  std::atomic<size_t> unknown_params(0);
  std::vector<std::vector<xla::ComputationClient::Data*>> parameters(
      contexts_map.size());
  std::vector<std::vector<xla::int64>> input_mapping(contexts_map.size());
  std::vector<std::vector<xla::int64>> index_mapping(contexts_map.size());
  std::vector<std::string> devices(contexts_map.size());
  std::vector<xla::Shape> shapes(contexts_map.size());
  xla::xla_util::MultiWait mwait(contexts_map.size());
  std::vector<xla::ComputationClient::CompileInstance> instances(
      contexts_map.size());
  size_t index = 0;
  for (auto& device_and_context : contexts_map) {
    const Device& device = device_and_context.first;
    DeviceContext* device_context = &device_and_context.second;

    auto generator = [&, device_context, index]() {
      std::vector<xla::int64> device_index_mapping;
      for (auto i : device_context->index_mapping) {
        ir::Value ir_node = (*tensors)[i].CurrentIrNode();
        xla::XlaOp root = device_context->lowering_ctx.GetOutputOp(ir_node);
        device_context->lowering_ctx.AddResult(root);
        device_index_mapping.push_back((*tensors)[i].GetUniqueId());
      }
      index_mapping[index] = std::move(device_index_mapping);

      xla::XlaComputation computation =
          device_context->lowering_ctx.Build().ConsumeValueOrDie();
      xla::ProgramShape program_shape =
          computation.GetProgramShape().ConsumeValueOrDie();
      shapes[index] =
          MakeShapeWithDeviceLayout(program_shape.result(), device.hw_type);
      devices[index] = device.ToString();
      instances[index] = {std::move(computation),
                          std::vector<std::string>({devices[index]}),
                          &shapes[index]};

      std::vector<xla::ComputationClient::Data*> parameters_data =
          device_context->lowering_ctx.GetParametersData();
      if (apply_context != nullptr) {
        std::vector<xla::int64> device_input_mapping;
        for (auto data : parameters_data) {
          auto it = data_uid_map.find(data->unique_id());
          if (it != data_uid_map.end()) {
            device_input_mapping.push_back(it->second);
          } else {
            XLA_COUNTER("UnknownTensorData", 1);
            unknown_params += 1;
          }
        }
        input_mapping[index] = std::move(device_input_mapping);
      }
      parameters[index] = std::move(parameters_data);
    };
    xla::xla_env::ScheduleClosure(mwait.Completer(std::move(generator)));
    ++index;
  }
  TF_CHECK_OK(mwait.Wait());

  std::vector<std::shared_ptr<xla::ComputationClient::Computation>>
      computations;
  if (!instances.empty()) {
    computations = xla::ComputationClient::Get()->Compile(std::move(instances));

    xla::ComputationClient::ExecuteParallelOptions options;
    auto results = xla::ComputationClient::Get()->ExecuteParallel(
        xla::util::GetConstSharedPointers(computations), parameters, devices,
        options);
    auto context_iterator = contexts_map.begin();
    for (auto& computation_tuple_elements : results) {
      // Replace destination's underlying data with the result of the
      // computation.
      SetMulti(tensors, std::move(computation_tuple_elements),
               context_iterator->second.index_mapping);
      ++context_iterator;
    }
  }
  if (apply_context != nullptr) {
    if (unknown_params == 0) {
      *apply_context = {std::move(computations), std::move(uid_order),
                        std::move(input_mapping), std::move(index_mapping),
                        std::move(devices)};
    } else {
      *apply_context = ApplyContext();
    }
  }
}

Device XLATensor::CommonDeviceForTensors(
    const std::vector<XLATensor>& tensors) {
  XLA_CHECK(!tensors.empty());
  const Device& device = tensors.front().GetDevice();
  for (const auto& tensor : tensors) {
    XLA_CHECK_EQ(device, tensor.GetDevice());
  }
  return device;
}

}  // namespace torch_xla
