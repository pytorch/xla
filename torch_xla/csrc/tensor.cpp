#include "tensor.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <mutex>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "helpers.h"
#include "lowering_context.h"
#include "ops/arithmetic_ir_ops.h"
#include "ops/avg_pool2d.h"
#include "ops/conv2d.h"
#include "ops/cross_replica_sum.h"
#include "ops/device_data.h"
#include "ops/generic.h"
#include "ops/infer_output_shape.h"
#include "ops/max_pool2d.h"
#include "ops/ops.h"
#include "ops/scalar.h"
#include "ops/softmax.h"
#include "ops/threshold.h"
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

// The tensors arena tracks all the XLA tensors which are currently live. This
// is used to create XLA computation "barriers" in order to flush pending
// operations and ensure the same XLA computations are created during the
// training loops.
class TensorsArena {
 public:
  static TensorsArena* Get() {
    static TensorsArena* arena = new TensorsArena();
    return arena;
  }

  std::shared_ptr<XLATensor> RegisterTensor(std::shared_ptr<XLATensor> tensor) {
    std::lock_guard<std::mutex> lock(lock_);
    tensors_map_.emplace(tensor.get(), tensor);
    return tensor;
  }

  void UnregisterTensor(XLATensor* tensor) {
    std::lock_guard<std::mutex> lock(lock_);
    tensors_map_.erase(tensor);
  }

  std::vector<std::shared_ptr<XLATensor>> GetTensors() {
    std::lock_guard<std::mutex> lock(lock_);
    std::vector<std::shared_ptr<XLATensor>> tensors;
    for (auto& ptr_wptr : tensors_map_) {
      std::shared_ptr<XLATensor> tensor = ptr_wptr.second.lock();
      if (tensor != nullptr) {
        tensors.push_back(std::move(tensor));
      }
    }
    return tensors;
  }

 private:
  std::mutex lock_;
  std::map<XLATensor*, std::weak_ptr<XLATensor>> tensors_map_;
};

void SetMulti(const std::vector<std::shared_ptr<XLATensor>>& dest_tuple,
              std::vector<std::shared_ptr<xla::ComputationClient::Data>>
                  new_dest_elements,
              const std::vector<size_t>& index_mapping) {
  XLA_CHECK_EQ(index_mapping.size(), new_dest_elements.size());
  // Replace the underlying data for the destination tensors with the data in
  // "new_dest_elements".
  for (size_t i = 0; i < new_dest_elements.size(); ++i) {
    size_t dest_tuple_index = index_mapping[i];
    dest_tuple[dest_tuple_index]->SetXlaData(std::move(new_dest_elements[i]));
  }
}

}  // namespace

std::shared_ptr<XLATensor> XLATensor::Create(const at::Tensor& tensor,
                                             const Device& device,
                                             bool requires_grad) {
  return TensorsArena::Get()->RegisterTensor(
      std::make_shared<XLATensor>(tensor, device, requires_grad));
}

std::shared_ptr<XLATensor> XLATensor::Create(
    std::shared_ptr<xla::ComputationClient::Data> xla_data,
    bool requires_grad) {
  return TensorsArena::Get()->RegisterTensor(
      std::make_shared<XLATensor>(std::move(xla_data), requires_grad));
}

std::shared_ptr<XLATensor> XLATensor::Create(ir::NodePtr ir_node,
                                             const Device& device) {
  return TensorsArena::Get()->RegisterTensor(
      std::make_shared<XLATensor>(std::move(ir_node), device));
}

std::shared_ptr<XLATensor> XLATensor::Create(std::shared_ptr<Data> data) {
  return TensorsArena::Get()->RegisterTensor(
      std::make_shared<XLATensor>(std::move(data)));
}

XLATensor::~XLATensor() { TensorsArena::Get()->UnregisterTensor(this); }

XLATensor::XLATensor(const at::Tensor& tensor, const Device& device,
                     bool requires_grad)
    : data_(std::make_shared<Data>(TensorToXlaData(tensor, device), device)),
      requires_grad_(requires_grad) {}

XLATensor::XLATensor(std::shared_ptr<xla::ComputationClient::Data> xla_data,
                     bool requires_grad)
    : data_(std::make_shared<Data>(xla_data, Device(xla_data->device()))),
      requires_grad_(requires_grad) {}

XLATensor::XLATensor(ir::NodePtr ir_node, const Device& device)
    : data_(std::make_shared<Data>(std::move(ir_node), device)) {
  TryLimitGraphSize();
}

std::shared_ptr<XLATensor> XLATensor::grad() const { return data_->grad; }

void XLATensor::SetGradient(std::shared_ptr<XLATensor> grad) {
  if (data_->grad == nullptr) {
    data_->grad = std::move(grad);
  } else {
    data_->grad->ReferenceDataFrom(*grad);
  }
}

at::ScalarType XLATensor::dtype() const {
  return TensorTypeFromXlaType(shape().get().element_type());
}

xla::util::MaybeRef<xla::Shape> XLATensor::shape() const {
  if (data_->xla_data != nullptr) {
    return data_->xla_data->shape();
  }
  if (data_->ir_node != nullptr) {
    return data_->ir_node->shape();
  }
  XLA_CHECK(data_->tensor_data);
  const Device& device = GetDevice();
  return MakeArrayShapeFromDimensions(
      data_->tensor_data->sizes(),
      XlaHelpers::MakeXlaPrimitiveType(data_->tensor_data->type().scalarType(),
                                       &device),
      device.hw_type);
}

const Device& XLATensor::GetDevice() const { return data_->device; }

xla::int64 XLATensor::GetUniqueId() const { return data_->unique_id; }

std::shared_ptr<xla::ComputationClient::Data> XLATensor::GetXlaData() {
  std::shared_ptr<xla::ComputationClient::Data> xla_data = CurrentXlaData();
  if (xla_data != nullptr) {
    return xla_data;
  }
  if (data_->ir_node != nullptr) {
    ApplyPendingGraph();
  } else {
    XLA_CHECK(data_->tensor_data);
    data_->xla_data = TensorToXlaData(*data_->tensor_data, GetDevice());
  }
  return data_->xla_data;
}

std::shared_ptr<xla::ComputationClient::Data> XLATensor::CurrentXlaData()
    const {
  if (data_->xla_data != nullptr) {
    // When we set a new Node for a tensor, we leave the XLA data pointer alive,
    // as it is needed in order for the cached tensor apply operation to work.
    // See comment in the SetIrNode() API.
    // In order to verify that that data is still valid as far as current tensor
    // data POV, we need to verify that the eventual IR Node is a DeviceData
    // node, and that its ComputationClient data pointer matches.
    const ir::NodePtr& ir_node = CurrentIrNode();
    if (ir_node == nullptr) {
      // If there is no IR node, then the XLA data is valid.
      return data_->xla_data;
    }
    const ir::ops::DeviceData* device_data =
        dynamic_cast<const ir::ops::DeviceData*>(ir_node.get());
    if (device_data != nullptr &&
        device_data->data().get() == data_->xla_data.get()) {
      return data_->xla_data;
    }
  }
  return nullptr;
}

std::string XLATensor::DumpGraphNodeComputation() const {
  std::string hlo_text;
  const ir::NodePtr& ir_node = CurrentIrNode();
  if (ir_node != nullptr) {
    ir::LoweringContext lowering_ctx("DumpGraphNodeComputation");
    XLA_CHECK_EQ(ir_node->num_outputs(), 1);
    xla::XlaOp root = lowering_ctx.GetOutputOp(ir::Output(ir_node.get(), 0));
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
  data_->xla_data = std::move(xla_data);
  data_->ir_node = nullptr;
  data_->tensor_data = c10::nullopt;
}

void XLATensor::SetIrNode(ir::NodePtr ir_node) {
  // We do not want to nullify that XLA data pointer here, as otherwise the
  // tensor apply computation caching will not work correctly.
  // If A is a tensor, a typical optimizer step computation will do:
  //  A' = F(A)
  // The cached apply computation will want to find the previous XLA data for
  // A's unique ID (as that data will be input to F()), but if setting A's IR
  // node nullify that, it will not be found.
  // We do have logic in CurrentXlaData() to verify that the XLA data pointer is
  // actually valid, as far as tensor value goes.
  data_->ir_node = std::move(ir_node);
  data_->tensor_data = c10::nullopt;
  TryLimitGraphSize();
}

void XLATensor::TryLimitGraphSize() {
  // If we are accumulating too many nodes in the pending graph, render the XLA
  // by executing the pending graph.
  static const size_t kMaxPendingGraphSize = 1000;
  if (data_->ir_node != nullptr &&
      data_->ir_node->graph_size() > kMaxPendingGraphSize) {
    ApplyPendingGraph();
  }
}

ir::NodePtr XLATensor::GetIrNode() const {
  const ir::NodePtr& ir_node = CurrentIrNode();
  if (ir_node != nullptr) {
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
    data_->ir_node = CreateTensorNode(xla_data);
    return data_->ir_node;
  }
  const c10::optional<at::Tensor>& tensor_data = CurrentTensorData();
  XLA_CHECK(tensor_data);
  // Now we have a tensor data. Do we force the creation of device memory, or we
  // generate an IR Node Constant for it?
  // TODO: For now force device data, but considerations about tensor size could
  // drive different logic.
  data_->xla_data = TensorToXlaData(*tensor_data, GetDevice());
  data_->ir_node = CreateTensorNode(data_->xla_data);
  return data_->ir_node;
}

const ir::NodePtr& XLATensor::CurrentIrNode() const { return data_->ir_node; }

void XLATensor::SetTensorData(at::Tensor tensor_data) {
  data_->tensor_data = std::move(tensor_data);
}

const c10::optional<at::Tensor>& XLATensor::CurrentTensorData() const {
  return data_->tensor_data;
}

void XLATensor::ReferenceDataFrom(const XLATensor& source) {
  XLA_CHECK_EQ(data_->device, source.data_->device);
  XLA_CHECK(xla::ShapeUtil::Equal(shape(), source.shape()))
      << shape() << " vs " << source.shape();

  data_->xla_data = source.data_->xla_data;
  data_->ir_node = source.data_->ir_node;
  data_->tensor_data = source.data_->tensor_data;
}

std::vector<int64_t> XLATensor::Size() const {
  const xla::Shape& tensor_shape = shape();
  return std::vector<int64_t>(tensor_shape.dimensions().begin(),
                              tensor_shape.dimensions().end());
}

at::Tensor XLATensor::ToTensor() {
  c10::optional<at::Tensor> tensor_data = CurrentTensorData();
  if (!tensor_data) {
    // The GetXlaData() call will trigger an ApplyPendingGraph() if an IR Node
    // is available on the tensor.
    std::vector<xla::Literal> literals =
        xla::ComputationClient::Get()->TransferFromServer({GetXlaData()});
    tensor_data = torch::autograd::make_variable(
        MakeTensorFromXlaLiteral(literals.front()), RequiresGrad());
    SetTensorData(*tensor_data);
  }
  return *tensor_data;
}

at::Tensor XLATensor::ToMutableTensor() {
  at::Tensor tensor_data = ToTensor();
  // In case of the ATEN Tensor data being possibly dirty, we do clear both the
  // IR Node and the XLA data. This API will be called to feed the tensor data
  // to ATEN APIs, and when we get to that point, we already lost the full XLA
  // fusion deal (and hence we do not need to keep the XLA data around for
  // caching computations).
  data_->xla_data = nullptr;
  data_->ir_node = nullptr;
  return tensor_data;
}

std::vector<std::shared_ptr<XLATensor>> XLATensor::GetLiveTensors() {
  return TensorsArena::Get()->GetTensors();
}

std::vector<at::Tensor> XLATensor::GetTensors(
    const std::vector<std::shared_ptr<XLATensor>>& tensors) {
  // TODO(dlibenzi): We do apply/compute and then fetch. Changing the API to
  // support getting handles and data might save a few pennies here.
  ApplyPendingGraph(tensors, /*apply_context=*/nullptr);

  std::vector<std::shared_ptr<xla::ComputationClient::Data>> tensors_data;
  for (auto& tensor : tensors) {
    if (!tensor->CurrentTensorData()) {
      tensors_data.push_back(tensor->GetXlaData());
    }
  }
  std::vector<xla::Literal> literals =
      xla::ComputationClient::Get()->TransferFromServer(tensors_data);
  std::vector<at::Tensor> results;
  size_t literals_index = 0;
  results.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    const c10::optional<at::Tensor>& tensor_data =
        tensors[i]->CurrentTensorData();
    if (tensor_data) {
      results.push_back(*tensor_data);
    } else {
      XLA_CHECK_LT(literals_index, literals.size());
      results.push_back(torch::autograd::make_variable(
          MakeTensorFromXlaLiteral(literals[literals_index]),
          tensors[i]->RequiresGrad()));
      ++literals_index;
    }
  }
  return results;
}

std::vector<std::shared_ptr<XLATensor>> XLATensor::CreateTensors(
    const std::vector<at::Tensor>& tensors,
    const std::vector<std::string>& devices) {
  std::vector<std::shared_ptr<xla::ComputationClient::Data>> handles =
      CreateTensorsData(tensors, devices);
  std::vector<std::shared_ptr<XLATensor>> xla_tensors;
  for (size_t i = 0; i < handles.size(); ++i) {
    xla_tensors.push_back(
        Create(std::move(handles[i]), tensors[i].requires_grad()));
  }
  return xla_tensors;
}

ir::NodePtr XLATensor::CreateTensorNode(
    std::shared_ptr<xla::ComputationClient::Data> data) {
  return ir::ops::DeviceDataOp(std::move(data));
}

xla::int64 XLATensor::GetNextTensorId() {
  static std::atomic<xla::int64>* id_generator = new std::atomic<xla::int64>(1);
  return id_generator->fetch_add(1);
}

std::shared_ptr<XLATensor> XLATensor::add(const XLATensor& other,
                                          const at::Scalar& alpha) {
  ir::NodePtr constant = ir::ops::ScalarOp(alpha.toDouble(), other.shape());
  return Create(GetIrNode() + other.GetIrNode() * constant, data_->device);
}

void XLATensor::add_(const XLATensor& other, const at::Scalar& alpha) {
  ir::NodePtr constant = ir::ops::ScalarOp(alpha.toDouble(), other.shape());
  SetIrNode(GetIrNode() + other.GetIrNode() * constant);
}

std::shared_ptr<XLATensor> XLATensor::mul(const XLATensor& other) {
  return Create(GetIrNode() * other.GetIrNode(), data_->device);
}

std::shared_ptr<XLATensor> XLATensor::mul(const at::Scalar& other) {
  ir::NodePtr constant = ir::ops::ScalarOp(other.toDouble(), shape());
  return Create(GetIrNode() * constant, data_->device);
}

void XLATensor::mul_(const XLATensor& other) {
  SetIrNode(GetIrNode() * other.GetIrNode());
}

void XLATensor::mul_(const at::Scalar& other) {
  ir::NodePtr constant = ir::ops::ScalarOp(other.toDouble(), shape());
  SetIrNode(GetIrNode() * constant);
}

std::shared_ptr<XLATensor> XLATensor::div(const XLATensor& other) {
  return Create(GetIrNode() / other.GetIrNode(), data_->device);
}

std::shared_ptr<XLATensor> XLATensor::div(const at::Scalar& other) {
  ir::NodePtr constant = ir::ops::ScalarOp(other.toDouble(), shape());
  return Create(GetIrNode() / constant, data_->device);
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
  ir::NodePtr div = tensor1.GetIrNode() / tensor2.GetIrNode();
  SetIrNode(GetIrNode() + div * constant);
}

void XLATensor::addcmul_(const at::Scalar& value, const XLATensor& tensor1,
                         const XLATensor& tensor2) {
  ir::NodePtr constant =
      ir::ops::ScalarOp(value.toDouble(), tensor1.shape().get().element_type());
  ir::NodePtr mul = tensor1.GetIrNode() * tensor2.GetIrNode();
  SetIrNode(GetIrNode() + mul * constant);
}

xla::int64 XLATensor::size(int dim) const {
  const xla::Shape& xla_shape = shape();
  int rank = xla_shape.dimensions_size();
  int min_shape_dim = -rank;
  int max_shape_dim = rank - 1;
  XLA_CHECK(min_shape_dim <= dim && dim <= max_shape_dim) << absl::StrCat(
      "Dimension out of range (expected to be in range of [", min_shape_dim,
      ", ", max_shape_dim, "], but got ", dim, ")");
  int dim_index = dim < 0 ? rank + dim : dim;
  XLA_CHECK_GE(dim_index, 0);
  XLA_CHECK_LT(dim_index, rank);
  return xla_shape.dimensions(dim_index);
}

std::shared_ptr<XLATensor> XLATensor::relu() {
  return Create(ir::ops::ReluOp(ir::NodeOperand(GetIrNode())), GetDevice());
}

std::shared_ptr<XLATensor> XLATensor::threshold(float threshold, float value) {
  return Create(std::make_shared<ir::ops::Threshold>(
                    ir::NodeOperand(GetIrNode()), threshold, value),
                GetDevice());
}

std::shared_ptr<XLATensor> XLATensor::conv2d(
    const std::shared_ptr<XLATensor>& weight,
    const std::shared_ptr<XLATensor>& bias, int stride, int padding,
    bool use_full_conv_precision) {
  std::shared_ptr<ir::ops::Conv2d> ir_node;
  if (bias) {
    ir_node = std::make_shared<ir::ops::Conv2d>(
        ir::NodeOperand(GetIrNode()), ir::NodeOperand(weight->GetIrNode()),
        ir::NodeOperand(bias->GetIrNode()), stride, padding,
        use_full_conv_precision);
  } else {
    ir_node = std::make_shared<ir::ops::Conv2d>(
        ir::NodeOperand(GetIrNode()), ir::NodeOperand(weight->GetIrNode()),
        stride, padding, use_full_conv_precision);
  }
  return Create(ir_node, GetDevice());
}

std::shared_ptr<XLATensor> XLATensor::addmm(const XLATensor& weight,
                                            const XLATensor& bias,
                                            bool use_full_conv_precision) {
  return Create(ir::ops::AddMatMulOp(ir::NodeOperand(GetIrNode()),
                                     ir::NodeOperand(weight.GetIrNode()),
                                     ir::NodeOperand(bias.GetIrNode()),
                                     use_full_conv_precision),
                GetDevice());
}

std::shared_ptr<XLATensor> XLATensor::max_pool2d(int kernel_size, int stride,
                                                 int padding) {
  return Create(std::make_shared<ir::ops::MaxPool2d>(
                    ir::NodeOperand(GetIrNode()), kernel_size, stride, padding),
                GetDevice());
}

std::shared_ptr<XLATensor> XLATensor::avg_pool2d(int kernel_size, int stride,
                                                 int padding,
                                                 bool count_include_pad) {
  return Create(std::make_shared<ir::ops::AvgPool2d>(
                    ir::NodeOperand(GetIrNode()), kernel_size, stride, padding,
                    count_include_pad),
                GetDevice());
}

std::shared_ptr<XLATensor> XLATensor::t() {
  return Create(ir::ops::TransposeOp(ir::NodeOperand(GetIrNode())),
                GetDevice());
}

std::shared_ptr<XLATensor> XLATensor::view(
    tensorflow::gtl::ArraySlice<const xla::int64> output_size) {
  return Create(std::make_shared<ir::ops::View>(ir::NodeOperand(GetIrNode()),
                                                output_size),
                GetDevice());
}

std::shared_ptr<XLATensor> XLATensor::log_softmax(xla::int64 dim) {
  return Create(
      std::make_shared<ir::ops::LogSoftmax>(ir::NodeOperand(GetIrNode()), dim),
      GetDevice());
}

std::shared_ptr<XLATensor> XLATensor::cross_replica_sum(
    const std::vector<std::vector<xla::int64>>& groups) {
  ir::NodePtr crs =
      ir::ops::CrossReplicaSumOp(ir::NodeOperand(GetIrNode()), groups);
  return Create(std::move(crs), data_->device);
}

void XLATensor::ApplyPendingGraph() {
  // This method is called to ensure that the tensor data is available on
  // device, so that a call to CurrentXlaData() returns a valid pointer.
  if (CurrentXlaData() == nullptr) {
    const ir::NodePtr& ir_node = CurrentIrNode();
    if (ir_node != nullptr) {
      ir::LoweringContext lowering_ctx("ApplyPendingGraph");
      XLA_CHECK_EQ(ir_node->num_outputs(), 1);
      xla::XlaOp root = lowering_ctx.GetOutputOp(ir::Output(ir_node.get(), 0));
      xla::XlaComputation computation =
          lowering_ctx.Build(root).ConsumeValueOrDie();
      auto compiled_computation = xla::ComputationClient::Get()->Compile(
          std::move(computation), {GetDevice().ToString()},
          /*output_shape=*/nullptr);
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
      XLA_CHECK(data_->tensor_data);
      data_->xla_data = TensorToXlaData(*data_->tensor_data, GetDevice());
    }
  }
}

std::vector<size_t> XLATensor::GetApplyOrder(
    const std::vector<std::shared_ptr<XLATensor>>& tensors) {
  std::vector<at::Tensor> at_tensors;
  std::vector<std::string> devices;
  std::vector<size_t> at_tensor_index;
  std::vector<size_t> order;
  order.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i]->CurrentXlaData() == nullptr) {
      if (tensors[i]->CurrentIrNode() != nullptr) {
        // Add only tensors which need to be synced.
        order.push_back(i);
      } else {
        // The tensor only has at::Tensor data. We need to queue it for a device
        // upload.
        const c10::optional<at::Tensor>& tensor_data =
            tensors[i]->CurrentTensorData();
        XLA_CHECK(tensor_data);
        at_tensors.push_back(*tensor_data);
        devices.push_back(tensors[i]->GetDevice().ToString());
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
      tensors[at_tensor_index[i]]->data_->xla_data = std::move(handles[i]);
    }
  }

  // Order the tensors based on their device and unique ID, so that we try to
  // mazimize the chances of creating the same XLA computation, and hence
  // hitting the compilation cache.
  std::sort(order.begin(), order.end(), [&tensors](size_t i1, size_t i2) {
    int device_compare =
        tensors[i1]->GetDevice().compare(tensors[i2]->GetDevice());
    if (device_compare != 0) {
      return device_compare < 0;
    }
    return tensors[i1]->GetUniqueId() < tensors[i2]->GetUniqueId();
  });
  return order;
}

bool XLATensor::RunCachedApply(
    const std::vector<std::shared_ptr<XLATensor>>& tensors,
    const ApplyContext& apply_context) {
  // Within the ApplyContext we saved the tensors unique IDs, and here we have
  // to map back the unique IDs to the tensor indices within the tensors vector.
  std::unordered_map<xla::int64, size_t> uid_index_map(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    uid_index_map[tensors[i]->GetUniqueId()] = i;
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
            tensors[it->second]->data_->xla_data;
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
    const std::vector<std::shared_ptr<XLATensor>>& tensors) {
  DataUidMap data_uid_map(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    std::shared_ptr<xla::ComputationClient::Data> xla_data =
        tensors[i]->data_->xla_data;
    if (xla_data != nullptr) {
      auto it_inserted = data_uid_map.emplace(xla_data->unique_id(),
                                              tensors[i]->GetUniqueId());
      if (!it_inserted.second) {
        // It can happen that two tensors references the same device data.
        // This is due to ReferenceDataFrom() API calls, which we use to
        // update the tensors (inputs, gradients,...) with the new data. In
        // that case select the tensor with lower unique ID (older), as the
        // newer one is very likely the new data provider which will be going
        // away soon (as soon as the last tensor reference will go away).
        it_inserted.first->second = std::min<xla::int64>(
            it_inserted.first->second, tensors[i]->GetUniqueId());
        XLA_COUNTER("DuplicatedTensorData", 1);
      }
    }
  }
  return data_uid_map;
}

void XLATensor::ApplyPendingGraph(
    const std::vector<std::shared_ptr<XLATensor>>& tensors,
    ApplyContext* apply_context) {
  struct DeviceContext {
    DeviceContext() : lowering_ctx("ApplyPendingGraph") {}

    ir::LoweringContext lowering_ctx;
    std::vector<size_t> index_mapping;
  };

  std::vector<size_t> order = GetApplyOrder(tensors);
  std::vector<xla::int64> uid_order;
  uid_order.reserve(order.size());
  for (auto i : order) {
    uid_order.push_back(tensors[i]->GetUniqueId());
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
    data_uid_map = CreateDataUidMap(tensors);
  }

  std::map<Device, DeviceContext> contexts_map;
  for (auto i : order) {
    DeviceContext* device_context = &contexts_map[tensors[i]->GetDevice()];
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
        const ir::NodePtr& ir_node = tensors[i]->CurrentIrNode();
        XLA_CHECK_EQ(ir_node->num_outputs(), 1);
        xla::XlaOp root = device_context->lowering_ctx.GetOutputOp(
            ir::Output(ir_node.get(), 0));
        device_context->lowering_ctx.AddResult(root);
        device_index_mapping.push_back(tensors[i]->GetUniqueId());
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
    const std::vector<std::shared_ptr<XLATensor>>& tensors) {
  XLA_CHECK(!tensors.empty());
  const Device& device = tensors.front()->GetDevice();
  for (const auto& tensor : tensors) {
    XLA_CHECK_EQ(device, tensor->GetDevice());
  }
  return device;
}

}  // namespace torch_xla
