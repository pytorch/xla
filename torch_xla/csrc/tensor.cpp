#include "torch_xla/csrc/tensor.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <mutex>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"
#include "tensorflow/compiler/xla/xla_client/thread_pool.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "torch/csrc/autograd/variable.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/adaptive_avg_pool2d.h"
#include "torch_xla/csrc/ops/all.h"
#include "torch_xla/csrc/ops/any.h"
#include "torch_xla/csrc/ops/arg_max.h"
#include "torch_xla/csrc/ops/arg_min.h"
#include "torch_xla/csrc/ops/arithmetic_ir_ops.h"
#include "torch_xla/csrc/ops/avg_pool2d.h"
#include "torch_xla/csrc/ops/avg_pool2d_backward.h"
#include "torch_xla/csrc/ops/batch_norm_forward.h"
#include "torch_xla/csrc/ops/bitwise_ir_ops.h"
#include "torch_xla/csrc/ops/cast.h"
#include "torch_xla/csrc/ops/cat.h"
#include "torch_xla/csrc/ops/constant.h"
#include "torch_xla/csrc/ops/constant_pad_nd.h"
#include "torch_xla/csrc/ops/conv2d.h"
#include "torch_xla/csrc/ops/conv2d_backward.h"
#include "torch_xla/csrc/ops/cross_replica_sum.h"
#include "torch_xla/csrc/ops/device_data.h"
#include "torch_xla/csrc/ops/diagonal.h"
#include "torch_xla/csrc/ops/dropout.h"
#include "torch_xla/csrc/ops/einsum.h"
#include "torch_xla/csrc/ops/expand.h"
#include "torch_xla/csrc/ops/gather.h"
#include "torch_xla/csrc/ops/generic.h"
#include "torch_xla/csrc/ops/index_op.h"
#include "torch_xla/csrc/ops/index_select.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/kth_value.h"
#include "torch_xla/csrc/ops/leaky_relu.h"
#include "torch_xla/csrc/ops/log_softmax.h"
#include "torch_xla/csrc/ops/log_softmax_backward.h"
#include "torch_xla/csrc/ops/masked_fill.h"
#include "torch_xla/csrc/ops/max_pool2d.h"
#include "torch_xla/csrc/ops/max_pool2d_backward.h"
#include "torch_xla/csrc/ops/mean.h"
#include "torch_xla/csrc/ops/native_batch_norm_backward.h"
#include "torch_xla/csrc/ops/native_batch_norm_forward.h"
#include "torch_xla/csrc/ops/not_supported.h"
#include "torch_xla/csrc/ops/ops.h"
#include "torch_xla/csrc/ops/permute.h"
#include "torch_xla/csrc/ops/prod.h"
#include "torch_xla/csrc/ops/repeat.h"
#include "torch_xla/csrc/ops/scalar.h"
#include "torch_xla/csrc/ops/scatter.h"
#include "torch_xla/csrc/ops/select.h"
#include "torch_xla/csrc/ops/slice.h"
#include "torch_xla/csrc/ops/softmax.h"
#include "torch_xla/csrc/ops/split.h"
#include "torch_xla/csrc/ops/squeeze.h"
#include "torch_xla/csrc/ops/stack.h"
#include "torch_xla/csrc/ops/sum.h"
#include "torch_xla/csrc/ops/svd.h"
#include "torch_xla/csrc/ops/threshold.h"
#include "torch_xla/csrc/ops/threshold_backward.h"
#include "torch_xla/csrc/ops/topk.h"
#include "torch_xla/csrc/ops/tril.h"
#include "torch_xla/csrc/ops/triu.h"
#include "torch_xla/csrc/ops/unsqueeze.h"
#include "torch_xla/csrc/ops/view.h"
#include "torch_xla/csrc/tensor_ops.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/translator.h"

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
    std::shared_ptr<xla::ComputationClient::Data> xla_data, bool requires_grad,
    c10::optional<at::ScalarType> logical_element_type) {
  XLATensor xtensor(std::move(xla_data), requires_grad, logical_element_type);
  TensorsArena::Get()->RegisterTensor(xtensor.data_ptr());
  return xtensor;
}

XLATensor XLATensor::Create(
    ir::Value ir_value, const Device& device,
    c10::optional<at::ScalarType> logical_element_type) {
  XLATensor xtensor(std::move(ir_value), device, logical_element_type);
  TensorsArena::Get()->RegisterTensor(xtensor.data_ptr());
  return xtensor;
}

XLATensor XLATensor::Create(
    std::shared_ptr<View> view, const Device& device,
    c10::optional<at::ScalarType> logical_element_type) {
  XLATensor xtensor(std::move(view), device, logical_element_type);
  TensorsArena::Get()->RegisterTensor(xtensor.data_ptr());
  return xtensor;
}

XLATensor::XLATensor(const at::Tensor& tensor, const Device& device,
                     bool requires_grad)
    : data_(std::make_shared<Data>(tensor, device)) {
  data()->requires_grad = requires_grad;
}

XLATensor::XLATensor(std::shared_ptr<xla::ComputationClient::Data> xla_data,
                     bool requires_grad,
                     c10::optional<at::ScalarType> logical_element_type)
    : data_(std::make_shared<Data>(xla_data, Device(xla_data->device()),
                                   logical_element_type)) {
  data()->requires_grad = requires_grad;
}

XLATensor::XLATensor(ir::Value ir_value, const Device& device,
                     c10::optional<at::ScalarType> logical_element_type)
    : data_(std::make_shared<Data>(std::move(ir_value), device,
                                   logical_element_type)) {
  TryLimitGraphSize();
}

XLATensor::XLATensor(std::shared_ptr<View> view, const Device& device,
                     c10::optional<at::ScalarType> logical_element_type)
    : data_(std::make_shared<Data>(std::move(view), device,
                                   logical_element_type)) {}

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
  return data()->logical_element_type
             ? *data()->logical_element_type
             : TensorTypeFromXlaType(shape().get().element_type());
}

xla::util::MaybeRef<xla::Shape> XLATensor::shape() const {
  if (data()->view != nullptr) {
    return data()->view->shape();
  }
  if (data()->xla_data != nullptr) {
    return data()->xla_data->shape();
  }
  const Device& device = GetDevice();
  if (data()->ir_value) {
    const xla::Shape& node_shape = data()->ir_value.shape();
    return MakeArrayShapeFromDimensions(
        node_shape.dimensions(), node_shape.element_type(), device.hw_type);
  }
  XLA_CHECK(data()->tensor_data);
  return MakeArrayShapeFromDimensions(
      data()->tensor_data->sizes(),
      MakeXlaPrimitiveType(data()->tensor_data->type().scalarType(), &device),
      device.hw_type);
}

const Device& XLATensor::GetDevice() const { return data()->device; }

xla::int64 XLATensor::GetUniqueId() const { return data()->unique_id; }

std::shared_ptr<xla::ComputationClient::Data> XLATensor::GetXlaData() {
  bool up_to_date = true;
  if (data()->view != nullptr) {
    View::IrNode ir_value_updated = data()->view->GetViewIrNode();
    if (ir_value_updated.updated || !data()->ir_value) {
      up_to_date = false;
      data()->ir_value = ir_value_updated.ir_value;
    }
  }
  if (up_to_date) {
    std::shared_ptr<xla::ComputationClient::Data> xla_data = CurrentXlaData();
    if (xla_data != nullptr) {
      return xla_data;
    }
  }
  if (data()->ir_value) {
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
    // See comment in the SetIrValue() API.
    // In order to verify that that data is still valid as far as current tensor
    // data POV, we need to verify that the eventual IR Node is a DeviceData
    // node, and that its ComputationClient data pointer matches.
    ir::Value ir_value = CurrentIrValue();
    if (!ir_value) {
      // If there is no IR node, then the XLA data is valid.
      return data()->xla_data;
    }
    const ir::ops::DeviceData* device_data =
        dynamic_cast<const ir::ops::DeviceData*>(ir_value.node.get());
    if (device_data != nullptr &&
        device_data->data().get() == data()->xla_data.get()) {
      return data()->xla_data;
    }
  }
  return nullptr;
}

std::string XLATensor::DumpGraphNodeComputation() const {
  std::string hlo_text;
  ir::Value ir_value = CurrentIrValue();
  if (ir_value) {
    ir::LoweringContext lowering_ctx("DumpGraphNodeComputation");
    xla::XlaOp root = lowering_ctx.GetOutputOp(ir_value);
    auto computation = ConsumeValue(lowering_ctx.Build(root));
    hlo_text = ConsumeValue(xla::xrt_util::GetComputationHloText(computation));
  }
  return hlo_text;
}

void XLATensor::SetXlaData(
    std::shared_ptr<xla::ComputationClient::Data> xla_data) {
  data()->xla_data = std::move(xla_data);
  data()->ir_value = ir::Value();
  data()->tensor_data = c10::nullopt;
}

void XLATensor::SetIrValue(ir::Value ir_value) {
  if (data()->view != nullptr) {
    // If we have an active view, and a SetIrValue() happens, it means we are
    // within an in-place execution context, and we need to update the view's
    // alias as well.
    data()->view->Update(ir_value);
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
  data()->ir_value = std::move(ir_value);
  data()->tensor_data = c10::nullopt;
  TryLimitGraphSize();
}

void XLATensor::TryLimitGraphSize() {
  // If we are accumulating too many nodes in the pending graph, render the XLA
  // by executing the pending graph.
  static const size_t kMaxPendingGraphSize = 1000;
  if (data()->ir_value &&
      data()->ir_value->graph_size() > kMaxPendingGraphSize) {
    ApplyPendingGraph();
  }
}

bool XLATensor::ShouldBeOnDevice(const at::Tensor& tensor) const {
  // Anything which is not a scalar goes on device for now.
  return tensor.numel() > 1;
}

ir::Value XLATensor::GetIrValue(const at::Tensor& tensor) const {
  if (ShouldBeOnDevice(tensor)) {
    data()->xla_data = TensorToXlaData(tensor, GetDevice());
    return CreateTensorNode(data()->xla_data);
  }
  xla::Shape tensor_shape =
      CreateComputationShapeFromTensor(tensor, &GetDevice());
  xla::Literal literal = GetTensorLiteral(tensor, &tensor_shape, &GetDevice());
  return ir::MakeNode<ir::ops::Constant>(std::move(literal));
}

ir::Value XLATensor::GetIrValue() const {
  ir::Value ir_value = CurrentIrValue();
  if (ir_value) {
    return ir_value;
  }
  std::shared_ptr<xla::ComputationClient::Data> xla_data = CurrentXlaData();
  if (xla_data != nullptr) {
    // In case of tensor node, we do not clear the XLA data when we set the IR
    // node. This because we want further calls to GetIrValue() to fetch the
    // same IR node, and not create new ones (even though the lowering context
    // will still collapse them all into a single XLA parameter op). So call
    // which wants the XLA data will still find it, w/out having to fetch it via
    // a computation client from-server call.
    data()->ir_value = CreateTensorNode(xla_data);
    return data()->ir_value;
  }
  c10::optional<at::Tensor> tensor_data = CurrentTensorData();
  XLA_CHECK(tensor_data);
  data()->ir_value = GetIrValue(*tensor_data);
  return data()->ir_value;
}

ir::Value XLATensor::CurrentIrValue() const {
  if (data()->view != nullptr) {
    return data()->view->GetViewIrNode().ir_value;
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

void XLATensor::ReferenceDataFrom(const XLATensor& source) {
  XLA_CHECK_EQ(data()->device, source.data()->device);
  XLA_CHECK(xla::ShapeUtil::Equal(shape(), source.shape()))
      << shape() << " vs " << source.shape();

  data()->xla_data = source.data()->xla_data;
  data()->ir_value = source.data()->ir_value;
  data()->tensor_data = source.data()->tensor_data;
}

at::Tensor XLATensor::ToTensor() {
  c10::optional<at::Tensor> tensor_data = CurrentTensorData();
  if (!tensor_data) {
    // The GetXlaData() call will trigger an ApplyPendingGraph() if an IR Node
    // is available on the tensor.
    std::vector<xla::Literal> literals =
        xla::ComputationClient::Get()->TransferFromServer({GetXlaData()});
    tensor_data = MakeTensorFromXlaLiteral(literals.front(), dtype());
    SetTensorData(*tensor_data);
  }
  return *tensor_data;
}

void XLATensor::SetScalarType(
    c10::optional<at::ScalarType> logical_element_type) {
  data()->logical_element_type = logical_element_type;
}

void XLATensor::DiscardXlaData() {
  XLA_CHECK(data()->tensor_data);
  data()->xla_data = nullptr;
  data()->ir_value = ir::Value();
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
      results.push_back(MakeTensorFromXlaLiteral(literals[literals_index],
                                                 (*tensors)[i].dtype()));
      ++literals_index;
    }
  }
  if (writeable != nullptr) {
    XLA_CHECK_EQ(tensors->size(), writeable->size());
    for (size_t i = 0; i < tensors->size(); ++i) {
      if ((*writeable)[i]) {
        // If all we have for this tensor is ATEN tensor data, we need to set it
        // before calling DiscardXlaData(), which will otherwise error out.
        if (!(*tensors)[i].CurrentTensorData()) {
          (*tensors)[i].SetTensorData(results[i]);
        }
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
    xla_tensors.push_back(Create(std::move(handles[i]),
                                 tensors[i].requires_grad(),
                                 tensors[i].scalar_type()));
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

std::vector<XLATensor> XLATensor::MakeOutputTensors(ir::NodePtr node) const {
  std::vector<XLATensor> tensors;
  tensors.reserve(node->num_outputs());
  for (size_t i = 0; i < node->num_outputs(); ++i) {
    tensors.push_back(CreateFrom(ir::Value(node, i)));
  }
  return tensors;
}

XLATensor XLATensor::add(const XLATensor& input, const XLATensor& other,
                         const at::Scalar& alpha) {
  ir::NodePtr constant = ir::ops::ScalarOp(alpha, other.shape());
  return input.CreateFrom(input.GetIrValue() + other.GetIrValue() * constant);
}

void XLATensor::add_(XLATensor& input, const XLATensor& other,
                     const at::Scalar& alpha) {
  ir::NodePtr constant = ir::ops::ScalarOp(alpha, other.shape());
  input.SetIrValue(input.GetIrValue() + other.GetIrValue() * constant);
}

XLATensor XLATensor::add(const XLATensor& input, const at::Scalar& other,
                         const at::Scalar& alpha) {
  ir::NodePtr other_constant = ir::ops::ScalarOp(other, input.shape());
  ir::NodePtr alpha_constant = ir::ops::ScalarOp(alpha, input.shape());
  return input.CreateFrom(input.GetIrValue() + other_constant * alpha_constant);
}

void XLATensor::add_(XLATensor& input, const at::Scalar& other,
                     const at::Scalar& alpha) {
  ir::NodePtr other_constant = ir::ops::ScalarOp(other, input.shape());
  ir::NodePtr alpha_constant = ir::ops::ScalarOp(alpha, input.shape());
  input.SetIrValue(input.GetIrValue() + other_constant * alpha_constant);
}

XLATensor XLATensor::sub(const XLATensor& input, const XLATensor& other,
                         const at::Scalar& alpha) {
  ir::NodePtr constant = ir::ops::ScalarOp(alpha, other.shape());
  return input.CreateFrom(input.GetIrValue() - other.GetIrValue() * constant);
}

void XLATensor::sub_(XLATensor& input, const XLATensor& other,
                     const at::Scalar& alpha) {
  ir::NodePtr constant = ir::ops::ScalarOp(alpha, other.shape());
  input.SetIrValue(input.GetIrValue() - other.GetIrValue() * constant);
}

XLATensor XLATensor::sub(const XLATensor& input, const at::Scalar& other,
                         const at::Scalar& alpha) {
  ir::NodePtr other_constant = ir::ops::ScalarOp(other, input.shape());
  ir::NodePtr alpha_constant = ir::ops::ScalarOp(alpha, input.shape());
  return input.CreateFrom(input.GetIrValue() - other_constant * alpha_constant);
}

void XLATensor::sub_(XLATensor& input, const at::Scalar& other,
                     const at::Scalar& alpha) {
  ir::NodePtr other_constant = ir::ops::ScalarOp(other, input.shape());
  ir::NodePtr alpha_constant = ir::ops::ScalarOp(alpha, input.shape());
  input.SetIrValue(input.GetIrValue() - other_constant * alpha_constant);
}

XLATensor XLATensor::mul(const XLATensor& input, const XLATensor& other) {
  return input.CreateFrom(input.GetIrValue() * other.GetIrValue());
}

XLATensor XLATensor::mul(const XLATensor& input, const at::Scalar& other) {
  ir::NodePtr constant = ir::ops::ScalarOp(other, input.shape());
  return input.CreateFrom(input.GetIrValue() * constant);
}

void XLATensor::mul_(XLATensor& input, const XLATensor& other) {
  input.SetIrValue(input.GetIrValue() * other.GetIrValue());
}

void XLATensor::mul_(XLATensor& input, const at::Scalar& other) {
  ir::NodePtr constant = ir::ops::ScalarOp(other, input.shape());
  input.SetIrValue(input.GetIrValue() * constant);
}

XLATensor XLATensor::div(const XLATensor& input, const XLATensor& other) {
  return input.CreateFrom(input.GetIrValue() / other.GetIrValue());
}

XLATensor XLATensor::div(const XLATensor& input, const at::Scalar& other) {
  ir::NodePtr constant = ir::ops::ScalarOp(other, input.shape());
  return input.CreateFrom(input.GetIrValue() / constant);
}

void XLATensor::div_(XLATensor& input, const XLATensor& other) {
  input.SetIrValue(input.GetIrValue() / other.GetIrValue());
}

void XLATensor::div_(XLATensor& input, const at::Scalar& other) {
  ir::NodePtr constant = ir::ops::ScalarOp(other, input.shape());
  input.SetIrValue(input.GetIrValue() / constant);
}

XLATensor XLATensor::fmod(const XLATensor& input, const XLATensor& other) {
  return input.CreateFrom(
      ir::ops::Fmod(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::fmod(const XLATensor& input, at::Scalar other) {
  ir::NodePtr constant = ir::ops::ScalarOp(other, input.shape());
  return input.CreateFrom(ir::ops::Fmod(input.GetIrValue(), constant));
}

void XLATensor::fmod_(XLATensor& input, at::Scalar other) {
  ir::NodePtr constant = ir::ops::ScalarOp(other, input.shape());
  input.SetIrValue(ir::ops::Fmod(input.GetIrValue(), constant));
}

void XLATensor::fmod_(XLATensor& input, const XLATensor& other) {
  input.SetIrValue(ir::ops::Fmod(input.GetIrValue(), other.GetIrValue()));
}

void XLATensor::zero_(XLATensor& input) {
  input.SetIrValue(ir::ops::ScalarOp(0.0, input.shape()));
}

void XLATensor::s_copy_(XLATensor& input, const XLATensor& src) {
  input.SetIrValue(src.GetIrValue());
}

XLATensor XLATensor::addcmul(const XLATensor& input, const at::Scalar& value,
                             const XLATensor& tensor1,
                             const XLATensor& tensor2) {
  ir::NodePtr constant =
      ir::ops::ScalarOp(value, tensor1.shape().get().element_type());
  ir::Value mul = tensor1.GetIrValue() * tensor2.GetIrValue();
  return input.CreateFrom(input.GetIrValue() + mul * constant);
}

void XLATensor::addcmul_(XLATensor& input, const at::Scalar& value,
                         const XLATensor& tensor1, const XLATensor& tensor2) {
  ir::NodePtr constant =
      ir::ops::ScalarOp(value, tensor1.shape().get().element_type());
  ir::Value mul = tensor1.GetIrValue() * tensor2.GetIrValue();
  input.SetIrValue(input.GetIrValue() + mul * constant);
}

XLATensor XLATensor::addcdiv(const XLATensor& input, const at::Scalar& value,
                             const XLATensor& tensor1,
                             const XLATensor& tensor2) {
  ir::NodePtr constant =
      ir::ops::ScalarOp(value.toDouble(), tensor1.shape().get().element_type());
  ir::Value div = tensor1.GetIrValue() / tensor2.GetIrValue();
  return input.CreateFrom(input.GetIrValue() + div * constant);
}

void XLATensor::addcdiv_(XLATensor& input, const at::Scalar& value,
                         const XLATensor& tensor1, const XLATensor& tensor2) {
  ir::NodePtr constant =
      ir::ops::ScalarOp(value.toDouble(), tensor1.shape().get().element_type());
  ir::Value div = tensor1.GetIrValue() / tensor2.GetIrValue();
  input.SetIrValue(input.GetIrValue() + div * constant);
}

xla::int64 XLATensor::size(xla::int64 dim) const {
  auto xla_shape = shape();
  int rank = xla_shape.get().rank();
  int dim_index = XlaHelpers::GetCanonicalDimensionIndex(dim, rank);
  return xla_shape.get().dimensions(dim_index);
}

XLATensor XLATensor::arange(at::Scalar start, at::Scalar end, at::Scalar step,
                            const Device& device, at::ScalarType scalar_type) {
  return Create(ir::ops::ARange(start, end, step, scalar_type), device,
                scalar_type);
}

XLATensor XLATensor::all(const XLATensor& input,
                         std::vector<xla::int64> dimensions,
                         bool keep_reduced_dimensions) {
  return input.CreateFrom(ir::MakeNode<ir::ops::All>(
      input.GetIrValue(), std::move(dimensions), keep_reduced_dimensions));
}

XLATensor XLATensor::any(const XLATensor& input,
                         std::vector<xla::int64> dimensions,
                         bool keep_reduced_dimensions) {
  return input.CreateFrom(ir::MakeNode<ir::ops::Any>(
      input.GetIrValue(), std::move(dimensions), keep_reduced_dimensions));
}

XLATensor XLATensor::ne(const XLATensor& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::ne, input, other);
}

void XLATensor::ne_(XLATensor& input, const at::Scalar& other) {
  ir::NodePtr cmp_result =
      ir::ops::ComparisonOp(at::aten::ne, input.GetIrValue(), other);
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::ne(const XLATensor& input, const XLATensor& other) {
  return DispatchComparisonOp(at::aten::ne, input, other);
}

void XLATensor::ne_(XLATensor& input, const XLATensor& other) {
  ir::NodePtr cmp_result = ir::ops::ComparisonOp(
      at::aten::ne, input.GetIrValue(), other.GetIrValue());
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::eq(const XLATensor& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::eq, input, other);
}

void XLATensor::eq_(XLATensor& input, const at::Scalar& other) {
  ir::NodePtr cmp_result =
      ir::ops::ComparisonOp(at::aten::eq, input.GetIrValue(), other);
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::eq(const XLATensor& input, const XLATensor& other) {
  return DispatchComparisonOp(at::aten::eq, input, other);
}

void XLATensor::eq_(XLATensor& input, const XLATensor& other) {
  ir::NodePtr cmp_result = ir::ops::ComparisonOp(
      at::aten::eq, input.GetIrValue(), other.GetIrValue());
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::ge(const XLATensor& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::ge, input, other);
}

void XLATensor::ge_(XLATensor& input, const at::Scalar& other) {
  ir::NodePtr cmp_result =
      ir::ops::ComparisonOp(at::aten::ge, input.GetIrValue(), other);
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::ge(const XLATensor& input, const XLATensor& other) {
  return DispatchComparisonOp(at::aten::ge, input, other);
}

void XLATensor::ge_(XLATensor& input, const XLATensor& other) {
  ir::NodePtr cmp_result = ir::ops::ComparisonOp(
      at::aten::ge, input.GetIrValue(), other.GetIrValue());
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::le(const XLATensor& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::le, input, other);
}

void XLATensor::le_(XLATensor& input, const at::Scalar& other) {
  ir::NodePtr cmp_result =
      ir::ops::ComparisonOp(at::aten::le, input.GetIrValue(), other);
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::le(const XLATensor& input, const XLATensor& other) {
  return DispatchComparisonOp(at::aten::le, input, other);
}

void XLATensor::le_(XLATensor& input, const XLATensor& other) {
  ir::NodePtr cmp_result = ir::ops::ComparisonOp(
      at::aten::le, input.GetIrValue(), other.GetIrValue());
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::gt(const XLATensor& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::gt, input, other);
}

void XLATensor::gt_(XLATensor& input, const at::Scalar& other) {
  ir::NodePtr cmp_result =
      ir::ops::ComparisonOp(at::aten::gt, input.GetIrValue(), other);
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::gt(const XLATensor& input, const XLATensor& other) {
  return DispatchComparisonOp(at::aten::gt, input, other);
}

void XLATensor::gt_(XLATensor& input, const XLATensor& other) {
  ir::NodePtr cmp_result = ir::ops::ComparisonOp(
      at::aten::gt, input.GetIrValue(), other.GetIrValue());
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::lt(const XLATensor& input, const at::Scalar& other) {
  return DispatchComparisonOp(at::aten::lt, input, other);
}

void XLATensor::lt_(XLATensor& input, const at::Scalar& other) {
  ir::NodePtr cmp_result =
      ir::ops::ComparisonOp(at::aten::lt, input.GetIrValue(), other);
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::lt(const XLATensor& input, const XLATensor& other) {
  return DispatchComparisonOp(at::aten::lt, input, other);
}

void XLATensor::lt_(XLATensor& input, const XLATensor& other) {
  ir::NodePtr cmp_result = ir::ops::ComparisonOp(
      at::aten::lt, input.GetIrValue(), other.GetIrValue());
  input.SetIrValue(ir::MakeNode<ir::ops::Cast>(cmp_result, input.dtype()));
}

XLATensor XLATensor::rsub(const XLATensor& input, const XLATensor& other,
                          const at::Scalar& alpha) {
  ir::NodePtr alpha_xla = ir::ops::ScalarOp(alpha, other.shape());
  return input.CreateFrom(other.GetIrValue() - alpha_xla * input.GetIrValue());
}

XLATensor XLATensor::rsub(const XLATensor& input, const at::Scalar& other,
                          const at::Scalar& alpha) {
  ir::NodePtr alpha_xla = ir::ops::ScalarOp(alpha, input.shape());
  ir::NodePtr other_xla = ir::ops::ScalarOp(other, input.shape());
  return input.CreateFrom(other_xla - alpha_xla * input.GetIrValue());
}

XLATensor XLATensor::__and__(const XLATensor& input, const at::Scalar& other) {
  XLA_CHECK(xla::ShapeUtil::ElementIsIntegral(input.shape()))
      << "Bitwise and is only supported for integer type tensors";
  ir::NodePtr other_broadcasted_ir = ir::ops::ScalarOp(other, input.shape());
  return input.CreateFrom(
      ir::ops::BitwiseAnd(input.GetIrValue(), other_broadcasted_ir));
}

XLATensor XLATensor::__and__(const XLATensor& input, const XLATensor& other) {
  XLA_CHECK(xla::ShapeUtil::ElementIsIntegral(input.shape()))
      << "Bitwise and is only supported for integer type tensors";
  return input.CreateFrom(
      ir::ops::BitwiseAnd(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::__or__(const XLATensor& input, const at::Scalar& other) {
  XLA_CHECK(xla::ShapeUtil::ElementIsIntegral(input.shape()))
      << "Bitwise or is only supported for integer type tensors";
  ir::NodePtr other_broadcasted_ir = ir::ops::ScalarOp(other, input.shape());
  return input.CreateFrom(
      ir::ops::BitwiseOr(input.GetIrValue(), other_broadcasted_ir));
}

XLATensor XLATensor::__or__(const XLATensor& input, const XLATensor& other) {
  XLA_CHECK(xla::ShapeUtil::ElementIsIntegral(input.shape()))
      << "Bitwise or is only supported for integer type tensors";
  return input.CreateFrom(
      ir::ops::BitwiseOr(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::__xor__(const XLATensor& input, const at::Scalar& other) {
  XLA_CHECK(xla::ShapeUtil::ElementIsIntegral(input.shape()))
      << "Bitwise xor is only supported for integer type tensors";
  ir::NodePtr other_broadcasted_ir = ir::ops::ScalarOp(other, input.shape());
  return input.CreateFrom(
      ir::ops::BitwiseXor(input.GetIrValue(), other_broadcasted_ir));
}

XLATensor XLATensor::__xor__(const XLATensor& input, const XLATensor& other) {
  XLA_CHECK(xla::ShapeUtil::ElementIsIntegral(input.shape()))
      << "Bitwise xor is only supported for integer type tensors";
  return input.CreateFrom(
      ir::ops::BitwiseXor(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::relu(const XLATensor& input) {
  return input.CreateFrom(ir::ops::ReluOp(input.GetIrValue()));
}

void XLATensor::relu_(XLATensor& input) {
  input.SetIrValue(ir::ops::ReluOp(input.GetIrValue()));
}

XLATensor XLATensor::leaky_relu(const XLATensor& input, double negative_slope) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::LeakyRelu>(input.GetIrValue(), negative_slope));
}

void XLATensor::leaky_relu_(XLATensor& input, double negative_slope) {
  input.SetIrValue(
      ir::MakeNode<ir::ops::LeakyRelu>(input.GetIrValue(), negative_slope));
}

XLATensor XLATensor::DispatchComparisonOp(c10::Symbol kind,
                                          const XLATensor& input,
                                          const at::Scalar& other) {
  ir::NodePtr node = ir::ops::ComparisonOp(kind, input.GetIrValue(), other);
  return Create(node, input.GetDevice(), at::ScalarType::Byte);
}

XLATensor XLATensor::DispatchComparisonOp(c10::Symbol kind,
                                          const XLATensor& input,
                                          const XLATensor& other) {
  ir::NodePtr node =
      ir::ops::ComparisonOp(kind, input.GetIrValue(), other.GetIrValue());
  return Create(node, input.GetDevice(), at::ScalarType::Byte);
}

XLATensor XLATensor::threshold(const XLATensor& input, float threshold,
                               float value) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Threshold>(input.GetIrValue(), threshold, value));
}

XLATensor XLATensor::conv2d(
    const XLATensor& input, const XLATensor& weight, const XLATensor& bias,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding) {
  ir::NodePtr ir_value =
      ir::MakeNode<ir::ops::Conv2d>(input.GetIrValue(), weight.GetIrValue(),
                                    bias.GetIrValue(), stride, padding);
  return input.CreateFrom(ir_value);
}

XLATensor XLATensor::conv2d(
    const XLATensor& input, const XLATensor& weight,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding) {
  ir::NodePtr ir_value = ir::MakeNode<ir::ops::Conv2d>(
      input.GetIrValue(), weight.GetIrValue(), stride, padding);
  return input.CreateFrom(ir_value);
}

XLATensor XLATensor::addmm(const XLATensor& input, const XLATensor& weight,
                           const XLATensor& bias) {
  return input.CreateFrom(ir::ops::AddMatMulOp(
      input.GetIrValue(), weight.GetIrValue(), bias.GetIrValue()));
}

XLATensor XLATensor::max_pool2d(
    const XLATensor& input,
    tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding) {
  return input.CreateFrom(ir::MakeNode<ir::ops::MaxPool2d>(
      input.GetIrValue(), kernel_size, stride, padding));
}

XLATensor XLATensor::avg_pool2d(
    const XLATensor& input,
    tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding,
    bool count_include_pad) {
  return input.CreateFrom(ir::MakeNode<ir::ops::AvgPool2d>(
      input.GetIrValue(), kernel_size, stride, padding, count_include_pad));
}

XLATensor XLATensor::full(tensorflow::gtl::ArraySlice<const xla::int64> size,
                          at::Scalar fill_value, const Device& device,
                          at::ScalarType scalar_type) {
  xla::Shape shape = MakeArrayShapeFromDimensions(
      size, MakeXlaPrimitiveType(scalar_type, &device), device.hw_type);
  return Create(ir::MakeNode<ir::ops::Scalar>(fill_value, std::move(shape)),
                device, scalar_type);
}

XLATensor XLATensor::full_like(const XLATensor& input, at::Scalar fill_value,
                               const Device& device,
                               c10::optional<at::ScalarType> scalar_type) {
  xla::Shape tensor_shape = input.shape();
  if (scalar_type) {
    tensor_shape.set_element_type(MakeXlaPrimitiveType(*scalar_type, &device));
  } else {
    scalar_type = input.dtype();
  }
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Scalar>(fill_value, tensor_shape), device,
      *scalar_type);
}

XLATensor XLATensor::select(const XLATensor& input, xla::int64 dim,
                            xla::int64 index) {
  auto input_shape = input.shape();
  dim = XlaHelpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  index = XlaHelpers::GetCanonicalPosition(input_shape.get().dimensions(), dim,
                                           index);
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Select>(input.GetIrValue(), dim, index));
}

std::tuple<XLATensor, XLATensor, XLATensor> XLATensor::svd(
    const XLATensor& input, bool some, bool compute_uv) {
  ir::NodePtr node =
      ir::MakeNode<ir::ops::SVD>(input.GetIrValue(), some, compute_uv);
  return std::make_tuple(input.CreateFrom(ir::Value(node, 0)),
                         input.CreateFrom(ir::Value(node, 1)),
                         input.CreateFrom(ir::Value(node, 2)));
}

std::tuple<XLATensor, XLATensor> XLATensor::kthvalue(const XLATensor& input,
                                                     xla::int64 k,
                                                     xla::int64 dim,
                                                     bool keepdim) {
  ir::NodePtr node =
      ir::MakeNode<ir::ops::KthValue>(input.GetIrValue(), k, dim, keepdim);
  return std::make_tuple(
      input.CreateFrom(ir::Value(node, 0)),
      input.CreateFrom(ir::Value(node, 1), at::ScalarType::Long));
}

std::tuple<XLATensor, XLATensor> XLATensor::topk(const XLATensor& input,
                                                 xla::int64 k, xla::int64 dim,
                                                 bool largest, bool sorted) {
  ir::NodePtr node =
      ir::MakeNode<ir::ops::TopK>(input.GetIrValue(), k, dim, largest, sorted);
  return std::make_tuple(
      input.CreateFrom(ir::Value(node, 0)),
      input.CreateFrom(ir::Value(node, 1), at::ScalarType::Long));
}

XLATensor XLATensor::dropout(const XLATensor& input, double p) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Dropout>(input.GetIrValue(), p));
}

XLATensor XLATensor::norm(const XLATensor& input, c10::optional<at::Scalar> p,
                          c10::optional<at::ScalarType> dtype,
                          at::IntArrayRef dim, bool keepdim) {
  return input.CreateFrom(
      ir::ops::Norm(input.GetIrValue(), p, dtype, dim, keepdim));
}

XLATensor XLATensor::neg(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Neg(input.GetIrValue()));
}

void XLATensor::neg_(XLATensor& input) {
  input.SetIrValue(ir::ops::Neg(input.GetIrValue()));
}

XLATensor XLATensor::sign(const XLATensor& input) {
  return input.CreateFrom(ir::ops::SignOp(input.GetIrValue()));
}

void XLATensor::sign_(XLATensor& input) {
  input.SetIrValue(ir::ops::SignOp(input.GetIrValue()));
}

XLATensor XLATensor::asin(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Asin(input.GetIrValue()));
}

void XLATensor::asin_(XLATensor& input) {
  input.SetIrValue(ir::ops::Asin(input.GetIrValue()));
}

XLATensor XLATensor::sin(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Sin(input.GetIrValue()));
}

void XLATensor::sin_(XLATensor& input) {
  input.SetIrValue(ir::ops::Sin(input.GetIrValue()));
}

XLATensor XLATensor::sinh(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Sinh(input.GetIrValue()));
}

void XLATensor::sinh_(XLATensor& input) {
  input.SetIrValue(ir::ops::Sinh(input.GetIrValue()));
}

XLATensor XLATensor::acos(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Acos(input.GetIrValue()));
}

void XLATensor::acos_(XLATensor& input) {
  input.SetIrValue(ir::ops::Acos(input.GetIrValue()));
}

XLATensor XLATensor::cos(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Cos(input.GetIrValue()));
}

void XLATensor::cos_(XLATensor& input) {
  input.SetIrValue(ir::ops::Cos(input.GetIrValue()));
}

XLATensor XLATensor::cosh(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Cosh(input.GetIrValue()));
}

void XLATensor::cosh_(XLATensor& input) {
  input.SetIrValue(ir::ops::Cosh(input.GetIrValue()));
}

XLATensor XLATensor::atan(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Atan(input.GetIrValue()));
}

void XLATensor::atan_(XLATensor& input) {
  input.SetIrValue(ir::ops::Atan(input.GetIrValue()));
}

XLATensor XLATensor::atan2(const XLATensor& input, const XLATensor& other) {
  return input.CreateFrom(
      ir::ops::Atan2(input.GetIrValue(), other.GetIrValue()));
}

void XLATensor::atan2_(XLATensor& input, const XLATensor& other) {
  input.SetIrValue(ir::ops::Atan2(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::tan(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Tan(input.GetIrValue()));
}

void XLATensor::tan_(XLATensor& input) {
  input.SetIrValue(ir::ops::Tan(input.GetIrValue()));
}

XLATensor XLATensor::tanh(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Tanh(input.GetIrValue()));
}

void XLATensor::tanh_(XLATensor& input) {
  input.SetIrValue(ir::ops::Tanh(input.GetIrValue()));
}

XLATensor XLATensor::abs(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Abs(input.GetIrValue()));
}

void XLATensor::abs_(XLATensor& input) {
  input.SetIrValue(ir::ops::Abs(input.GetIrValue()));
}

XLATensor XLATensor::clamp(const XLATensor& input,
                           c10::optional<at::Scalar> min,
                           c10::optional<at::Scalar> max) {
  return input.CreateFrom(ir::ops::Clamp(input.GetIrValue(), min, max));
}

void XLATensor::clamp_(XLATensor& input, c10::optional<at::Scalar> min,
                       c10::optional<at::Scalar> max) {
  input.SetIrValue(ir::ops::Clamp(input.GetIrValue(), min, max));
}

XLATensor XLATensor::constant_pad_nd(
    const XLATensor& input, tensorflow::gtl::ArraySlice<const xla::int64> pad,
    const at::Scalar& value) {
  std::vector<xla::int64> complete_pad(pad.begin(), pad.end());
  complete_pad.resize(2 * input.shape().get().rank());
  return input.CreateFrom(ir::MakeNode<ir::ops::ConstantPadNd>(
      input.GetIrValue(), complete_pad, value));
}

XLATensor XLATensor::ceil(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Ceil(input.GetIrValue()));
}

void XLATensor::ceil_(XLATensor& input) {
  input.SetIrValue(ir::ops::Ceil(input.GetIrValue()));
}

XLATensor XLATensor::floor(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Floor(input.GetIrValue()));
}

void XLATensor::floor_(XLATensor& input) {
  input.SetIrValue(ir::ops::Floor(input.GetIrValue()));
}

XLATensor XLATensor::trunc(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Trunc(input.GetIrValue()));
}

void XLATensor::trunc_(XLATensor& input) {
  input.SetIrValue(ir::ops::Trunc(input.GetIrValue()));
}

XLATensor XLATensor::frac(const XLATensor& input) {
  return input.CreateFrom(ir::ops::FracOp(input.GetIrValue()));
}

void XLATensor::frac_(XLATensor& input) {
  input.SetIrValue(ir::ops::FracOp(input.GetIrValue()));
}

XLATensor XLATensor::slice(const XLATensor& input, xla::int64 dim,
                           xla::int64 start, xla::int64 end, xla::int64 step) {
  auto input_shape = input.shape();
  dim = XlaHelpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  start = XlaHelpers::GetCanonicalPosition(input_shape.get().dimensions(), dim,
                                           start);
  end = XlaHelpers::GetCanonicalPosition(input_shape.get().dimensions(), dim,
                                         end);
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Slice>(input.GetIrValue(), dim, start, end, step));
}

XLATensor XLATensor::gather(const XLATensor& input, xla::int64 dim,
                            const XLATensor& index) {
  return input.CreateFrom(ir::MakeNode<ir::ops::Gather>(
      input.GetIrValue(), GetCanonicalDimension(input, dim),
      index.GetIrValue()));
}

void XLATensor::scatter_(XLATensor& input, xla::int64 dim,
                         const XLATensor& index, const XLATensor& src) {
  input.SetIrValue(ir::MakeNode<ir::ops::Scatter>(
      input.GetIrValue(), GetCanonicalDimension(input, dim), index.GetIrValue(),
      src.GetIrValue()));
}

XLATensor XLATensor::scatter(const XLATensor& input, xla::int64 dim,
                             const XLATensor& index, const XLATensor& src) {
  return input.CreateFrom(ir::MakeNode<ir::ops::Scatter>(
      input.GetIrValue(), GetCanonicalDimension(input, dim), index.GetIrValue(),
      src.GetIrValue()));
}

void XLATensor::scatter_(XLATensor& input, xla::int64 dim,
                         const XLATensor& index, at::Scalar value) {
  ir::NodePtr constant = ir::ops::ScalarOp(value, input.shape());
  input.SetIrValue(ir::MakeNode<ir::ops::Scatter>(
      input.GetIrValue(), GetCanonicalDimension(input, dim), index.GetIrValue(),
      constant));
}

XLATensor XLATensor::scatter(const XLATensor& input, xla::int64 dim,
                             const XLATensor& index, at::Scalar value) {
  ir::NodePtr constant = ir::ops::ScalarOp(value, input.shape());
  return input.CreateFrom(ir::MakeNode<ir::ops::Scatter>(
      input.GetIrValue(), GetCanonicalDimension(input, dim), index.GetIrValue(),
      constant));
}

XLATensor XLATensor::index_select(const XLATensor& input, xla::int64 dim,
                                  const XLATensor& index) {
  return input.CreateFrom(ir::MakeNode<ir::ops::IndexSelect>(
      input.GetIrValue(), GetCanonicalDimension(input, dim),
      index.GetIrValue()));
}

XLATensor XLATensor::expand(const XLATensor& input,
                            std::vector<xla::int64> size) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Expand>(input.GetIrValue(), std::move(size)));
}

XLATensor XLATensor::index(
    const XLATensor& input,
    tensorflow::gtl::ArraySlice<const XLATensor> indices) {
  return IndexByTensors(input, indices);
}

XLATensor XLATensor::stack(tensorflow::gtl::ArraySlice<const XLATensor> tensors,
                           xla::int64 dim) {
  XLA_CHECK_GT(tensors.size(), 0);
  std::vector<ir::Value> values;
  for (auto& tensor : tensors) {
    values.push_back(tensor.GetIrValue());
  }
  return tensors[0].CreateFrom(ir::MakeNode<ir::ops::Stack>(values, dim));
}

XLATensor XLATensor::cat(tensorflow::gtl::ArraySlice<const XLATensor> tensors,
                         xla::int64 dim) {
  XLA_CHECK_GT(tensors.size(), 0);
  std::vector<ir::Value> values;
  for (auto& tensor : tensors) {
    if (xla::ShapeUtil::ElementsIn(tensor.shape()) > 0) {
      dim = GetCanonicalDimension(tensor, dim);
      values.push_back(tensor.GetIrValue());
    }
  }
  return tensors[0].CreateFrom(ir::MakeNode<ir::ops::Cat>(values, dim));
}

std::vector<XLATensor> XLATensor::unbind(const XLATensor& input,
                                         xla::int64 dim) {
  xla::int64 dim_size = input.size(dim);
  std::vector<XLATensor> slices;
  slices.reserve(dim_size);
  for (xla::int64 index = 0; index < dim_size; ++index) {
    slices.push_back(select(input, dim, index));
  }
  return slices;
}

XLATensor XLATensor::mm(const XLATensor& input, const XLATensor& weight) {
  return input.CreateFrom(
      ir::ops::Dot(input.GetIrValue(), weight.GetIrValue()));
}

XLATensor XLATensor::matmul(const XLATensor& input, const XLATensor& other) {
  return input.CreateFrom(
      ir::ops::MatMul(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::bmm(const XLATensor& batch1, const XLATensor& batch2) {
  // Consistent with the checks in bmm_out_or_baddbmm_.
  std::string tag = "bmm";
  CheckRank(batch1, 3, tag, "batch1", 1);
  CheckRank(batch2, 3, tag, "batch2", 2);
  xla::int64 batch_size = batch1.size(0);
  CheckDimensionSize(batch2, 0, batch_size, tag, "batch2", 2);
  xla::int64 contraction_size = batch1.size(2);
  CheckDimensionSize(batch2, 1, contraction_size, tag, "batch2", 2);
  return matmul(batch1, batch2);
}

std::vector<XLATensor> XLATensor::broadcast_tensors(
    tensorflow::gtl::ArraySlice<const XLATensor> tensors) {
  XLA_CHECK(!tensors.empty()) << "broadcast_tensors cannot take an empty list";
  std::vector<ir::Value> tensor_ir_values;
  for (const auto& tensor : tensors) {
    tensor_ir_values.push_back(tensor.GetIrValue());
  }
  ir::NodePtr node = ir::ops::BroadcastTensors(tensor_ir_values);
  return tensors.front().MakeOutputTensors(node);
}

XLATensor XLATensor::einsum(
    const std::string& equation,
    tensorflow::gtl::ArraySlice<const XLATensor> tensors) {
  std::vector<ir::Value> tensor_ir_values;
  for (const auto& tensor : tensors) {
    tensor_ir_values.push_back(tensor.GetIrValue());
  }
  XLA_CHECK_EQ(tensors.size(), 2);
  return tensors[0].CreateFrom(
      ir::MakeNode<ir::ops::Einsum>(equation, tensor_ir_values));
}

XLATensor XLATensor::exp(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Exp(input.GetIrValue()));
}

void XLATensor::exp_(XLATensor& input) {
  input.SetIrValue(ir::ops::Exp(input.GetIrValue()));
}

XLATensor XLATensor::expm1(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Expm1(input.GetIrValue()));
}

void XLATensor::expm1_(XLATensor& input) {
  input.SetIrValue(ir::ops::Expm1(input.GetIrValue()));
}

XLATensor XLATensor::log(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Log(input.GetIrValue()));
}

void XLATensor::log_(XLATensor& input) {
  input.SetIrValue(ir::ops::Log(input.GetIrValue()));
}

XLATensor XLATensor::log_base(const XLATensor& input, ir::OpKind op,
                              double base) {
  return input.CreateFrom(ir::ops::LogBase(input.GetIrValue(), op, base));
}

void XLATensor::log_base_(XLATensor& input, ir::OpKind op, double base) {
  input.SetIrValue(ir::ops::LogBase(input.GetIrValue(), op, base));
}

XLATensor XLATensor::log1p(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Log1p(input.GetIrValue()));
}

void XLATensor::log1p_(XLATensor& input) {
  input.SetIrValue(ir::ops::Log1p(input.GetIrValue()));
}

XLATensor XLATensor::erf(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Erf(input.GetIrValue()));
}

void XLATensor::erf_(XLATensor& input) {
  input.SetIrValue(ir::ops::Erf(input.GetIrValue()));
}

XLATensor XLATensor::erfc(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Erfc(input.GetIrValue()));
}

void XLATensor::erfc_(XLATensor& input) {
  input.SetIrValue(ir::ops::Erfc(input.GetIrValue()));
}

XLATensor XLATensor::erfinv(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Erfinv(input.GetIrValue()));
}

void XLATensor::erfinv_(XLATensor& input) {
  input.SetIrValue(ir::ops::Erfinv(input.GetIrValue()));
}

XLATensor XLATensor::sqrt(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Sqrt(input.GetIrValue()));
}

void XLATensor::sqrt_(XLATensor& input) {
  input.SetIrValue(ir::ops::Sqrt(input.GetIrValue()));
}

XLATensor XLATensor::rsqrt(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Rsqrt(input.GetIrValue()));
}

void XLATensor::rsqrt_(XLATensor& input) {
  input.SetIrValue(ir::ops::Rsqrt(input.GetIrValue()));
}

XLATensor XLATensor::reciprocal(const XLATensor& input) {
  return input.CreateFrom(ir::ops::ReciprocalOp(input.GetIrValue()));
}

void XLATensor::reciprocal_(XLATensor& input) {
  input.SetIrValue(ir::ops::ReciprocalOp(input.GetIrValue()));
}

XLATensor XLATensor::pow(const XLATensor& input, at::Scalar exponent) {
  ir::NodePtr exponent_node = ir::ops::ScalarOp(exponent, input.shape());
  return input.CreateFrom(ir::ops::Pow(input.GetIrValue(), exponent_node));
}

XLATensor XLATensor::mean(const XLATensor& input,
                          std::vector<xla::int64> dimensions,
                          bool keep_reduced_dimensions,
                          c10::optional<at::ScalarType> dtype) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Mean>(input.GetIrValue(), std::move(dimensions),
                                  keep_reduced_dimensions, dtype));
}

XLATensor XLATensor::sum(const XLATensor& input,
                         std::vector<xla::int64> dimensions,
                         bool keep_reduced_dimensions,
                         c10::optional<at::ScalarType> dtype) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Sum>(input.GetIrValue(), std::move(dimensions),
                                 keep_reduced_dimensions, dtype));
}

XLATensor XLATensor::prod(const XLATensor& input,
                          std::vector<xla::int64> dimensions,
                          bool keep_reduced_dimensions,
                          c10::optional<at::ScalarType> dtype) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Prod>(input.GetIrValue(), std::move(dimensions),
                                  keep_reduced_dimensions, dtype));
}

XLATensor XLATensor::batch_norm(const XLATensor& input, const XLATensor& weight,
                                const XLATensor& bias,
                                const XLATensor& running_mean,
                                const XLATensor& running_var, double momentum,
                                double eps) {
  return input.CreateFrom(ir::MakeNode<ir::ops::BatchNormForward>(
      input.GetIrValue(), weight.GetIrValue(), bias.GetIrValue(),
      running_mean.GetIrValue(), running_var.GetIrValue(), momentum, eps));
}

std::tuple<XLATensor, XLATensor, XLATensor> XLATensor::native_batch_norm(
    const XLATensor& input, const XLATensor& weight, const XLATensor& bias,
    const XLATensor& running_mean, const XLATensor& running_var,
    double momentum, double eps) {
  ir::NodePtr node = ir::MakeNode<ir::ops::NativeBatchNormForward>(
      input.GetIrValue(), weight.GetIrValue(), bias.GetIrValue(),
      running_mean.GetIrValue(), running_var.GetIrValue(), momentum, eps);
  XLATensor output = input.CreateFrom(ir::Value(node, 0));
  XLATensor save_mean = input.CreateFrom(ir::Value(node, 1));
  XLATensor save_invstd = input.CreateFrom(ir::Value(node, 2));
  return std::make_tuple(std::move(output), std::move(save_mean),
                         std::move(save_invstd));
}

std::tuple<XLATensor, XLATensor, XLATensor>
XLATensor::native_batch_norm_backward(
    const XLATensor& grad_out, const XLATensor& input, const XLATensor& weight,
    const XLATensor& running_mean, const XLATensor& running_var,
    const XLATensor& save_mean, const XLATensor& save_invstd, double eps) {
  ir::NodePtr node = ir::MakeNode<ir::ops::NativeBatchNormBackward>(
      grad_out.GetIrValue(), input.GetIrValue(), weight.GetIrValue(),
      running_mean.GetIrValue(), running_var.GetIrValue(),
      save_mean.GetIrValue(), save_invstd.GetIrValue(), eps);
  XLATensor grad_input = input.CreateFrom(ir::Value(node, 0));
  XLATensor grad_weight = input.CreateFrom(ir::Value(node, 1));
  XLATensor grad_bias = input.CreateFrom(ir::Value(node, 2));
  return std::make_tuple(std::move(grad_input), std::move(grad_weight),
                         std::move(grad_bias));
}

XLATensor XLATensor::permute(
    const XLATensor& input,
    tensorflow::gtl::ArraySlice<const xla::int64> dims) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Permute>(input.GetIrValue(), dims));
}

XLATensor XLATensor::repeat(
    const XLATensor& input,
    tensorflow::gtl::ArraySlice<const xla::int64> repeats) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Repeat>(input.GetIrValue(), repeats));
}

std::vector<XLATensor> XLATensor::split(const XLATensor& input,
                                        xla::int64 split_size, xla::int64 dim) {
  auto input_shape = input.shape();
  int split_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  xla::int64 dim_size = input_shape.get().dimensions(split_dim);
  if (split_size == 0) {
    // Deal with 0 split size, it's a corner case which is only allowed when the
    // dimension size is 0 as well.
    XLA_CHECK_EQ(dim_size, 0);
    xla::Literal literal(input_shape.get());
    return {
        input.CreateFrom(ir::MakeNode<ir::ops::Constant>(std::move(literal)))};
  }
  std::vector<xla::int64> split_sizes;
  for (; dim_size > 0; dim_size -= split_size) {
    split_sizes.push_back(std::min<xla::int64>(dim_size, split_size));
  }
  ir::NodePtr node = ir::MakeNode<ir::ops::Split>(
      input.GetIrValue(), std::move(split_sizes), split_dim);
  return input.MakeOutputTensors(node);
}

std::vector<XLATensor> XLATensor::split_with_sizes(
    const XLATensor& input,
    tensorflow::gtl::ArraySlice<const xla::int64> split_size, xla::int64 dim) {
  auto input_shape = input.shape();
  int split_dim =
      XlaHelpers::GetCanonicalDimensionIndex(dim, input_shape.get().rank());
  ir::NodePtr node = ir::MakeNode<ir::ops::Split>(
      input.GetIrValue(), xla::util::ToVector<xla::int64>(split_size),
      split_dim);
  return input.MakeOutputTensors(node);
}

XLATensor XLATensor::squeeze(const XLATensor& input) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Squeeze>(input.GetIrValue(), -1));
}

XLATensor XLATensor::squeeze(const XLATensor& input, xla::int64 dim) {
  return input.CreateFrom(ir::MakeNode<ir::ops::Squeeze>(
      input.GetIrValue(), GetCanonicalDimension(input, dim)));
}

void XLATensor::squeeze_(XLATensor& input) {
  input.SetIrValue(ir::MakeNode<ir::ops::Squeeze>(input.GetIrValue(), -1));
}

void XLATensor::squeeze_(XLATensor& input, xla::int64 dim) {
  input.SetIrValue(ir::MakeNode<ir::ops::Squeeze>(
      input.GetIrValue(), GetCanonicalDimension(input, dim)));
}

XLATensor XLATensor::unsqueeze(const XLATensor& input, xla::int64 dim) {
  int squeeze_dim = XlaHelpers::GetCanonicalDimensionIndex(
      dim, input.shape().get().rank() + 1);
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Unsqueeze>(input.GetIrValue(), squeeze_dim));
}

void XLATensor::unsqueeze_(XLATensor& input, xla::int64 dim) {
  int squeeze_dim = XlaHelpers::GetCanonicalDimensionIndex(
      dim, input.shape().get().rank() + 1);
  input.SetIrValue(
      ir::MakeNode<ir::ops::Unsqueeze>(input.GetIrValue(), squeeze_dim));
}

XLATensor XLATensor::masked_fill(const XLATensor& input, const XLATensor& mask,
                                 const at::Scalar& value) {
  // Expand mask to be the same size as input.
  ir::NodePtr expanded_mask = ir::MakeNode<ir::ops::Expand>(
      mask.GetIrValue(), input.shape().get().dimensions());
  return input.CreateFrom(ir::MakeNode<ir::ops::MaskedFill>(
      input.GetIrValue(), expanded_mask, value));
}

void XLATensor::masked_fill_(XLATensor& input, const XLATensor& mask,
                             const at::Scalar& value) {
  // Expand mask to be the same size as input.
  ir::NodePtr expanded_mask = ir::MakeNode<ir::ops::Expand>(
      mask.GetIrValue(), input.shape().get().dimensions());
  input.SetIrValue(ir::MakeNode<ir::ops::MaskedFill>(input.GetIrValue(),
                                                     expanded_mask, value));
}

void XLATensor::fill_(XLATensor& input, const at::Scalar& value) {
  input.SetIrValue(ir::ops::ScalarOp(value, input.shape()));
}

XLATensor XLATensor::cross(const XLATensor& input, const XLATensor& other,
                           xla::int64 dim) {
  return tensor_ops::Cross(input, other, dim);
}

XLATensor XLATensor::eye(xla::int64 lines, xla::int64 cols,
                         const Device& device, at::ScalarType element_type) {
  return XLATensor::Create(
      ir::ops::Identity(lines, cols,
                        MakeXlaPrimitiveType(element_type, &device)),
      device, element_type);
}

XLATensor XLATensor::triu(const XLATensor& input, xla::int64 diagonal) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Triu>(input.GetIrValue(), diagonal));
}

void XLATensor::triu_(XLATensor& input, xla::int64 diagonal) {
  input.SetIrValue(ir::MakeNode<ir::ops::Triu>(input.GetIrValue(), diagonal));
}

XLATensor XLATensor::tril(const XLATensor& input, xla::int64 diagonal) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Tril>(input.GetIrValue(), diagonal));
}

void XLATensor::tril_(XLATensor& input, xla::int64 diagonal) {
  input.SetIrValue(ir::MakeNode<ir::ops::Tril>(input.GetIrValue(), diagonal));
}

XLATensor XLATensor::trace(const XLATensor& input) {
  auto input_shape_ref = input.shape();
  XLA_CHECK_EQ((*input_shape_ref).rank(), 2)
      << "invalid argument for trace: expected a matrix";
  ir::NodePtr eye = ir::ops::Identity((*input_shape_ref).dimensions(0),
                                      (*input_shape_ref).dimensions(1),
                                      (*input_shape_ref).element_type());
  return XLATensor::sum(input.CreateFrom(eye * input.GetIrValue()), {0, 1},
                        false, input.dtype());
}

XLATensor XLATensor::diagonal(const XLATensor& input, xla::int64 offset,
                              xla::int64 dim1, xla::int64 dim2) {
  xla::int64 rank = input.shape().get().rank();
  xla::int64 canonical_dim1 =
      XlaHelpers::GetCanonicalDimensionIndex(dim1, rank);
  xla::int64 canonical_dim2 =
      XlaHelpers::GetCanonicalDimensionIndex(dim2, rank);
  return input.CreateFrom(ir::MakeNode<ir::ops::Diagonal>(
      input.GetIrValue(), offset, canonical_dim1, canonical_dim2));
}

XLATensor XLATensor::where(const XLATensor& condition, const XLATensor& input,
                           const XLATensor& other) {
  return input.CreateFrom(ir::ops::Where(
      condition.GetIrValue(), input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::_adaptive_avg_pool2d(
    const XLATensor& input,
    tensorflow::gtl::ArraySlice<const xla::int64> output_size) {
  return input.CreateFrom(ir::MakeNode<ir::ops::AdaptiveAvgPool2d>(
      input.GetIrValue(), output_size));
}

XLATensor XLATensor::avg_pool2d_backward(
    const XLATensor& out_backprop, const XLATensor& input,
    tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding,
    bool count_include_pad) {
  return out_backprop.CreateFrom(ir::MakeNode<ir::ops::AvgPool2dBackward>(
      out_backprop.GetIrValue(), input.GetIrValue(), kernel_size, stride,
      padding, count_include_pad));
}

XLATensor XLATensor::_adaptive_avg_pool2d_backward(const XLATensor& grad_output,
                                                   const XLATensor& input) {
  return input.CreateFrom(ir::ops::AdaptiveAvgPool2dBackward(
      grad_output.GetIrValue(), input.GetIrValue()));
}

XLATensor XLATensor::max_pool2d_backward(
    const XLATensor& out_backprop, const XLATensor& input,
    tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding) {
  return out_backprop.CreateFrom(ir::MakeNode<ir::ops::MaxPool2dBackward>(
      out_backprop.GetIrValue(), input.GetIrValue(), kernel_size, stride,
      padding));
}

std::tuple<XLATensor, XLATensor, XLATensor> XLATensor::conv2d_backward(
    const XLATensor& out_backprop, const XLATensor& input,
    const XLATensor& weight,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding) {
  ir::NodePtr node = ir::MakeNode<ir::ops::Conv2dBackward>(
      out_backprop.GetIrValue(), input.GetIrValue(), weight.GetIrValue(),
      stride, padding);
  XLATensor grad_input = out_backprop.CreateFrom(ir::Value(node, 0));
  XLATensor grad_weight = out_backprop.CreateFrom(ir::Value(node, 1));
  XLATensor grad_bias = out_backprop.CreateFrom(ir::Value(node, 2));
  return std::make_tuple(std::move(grad_input), std::move(grad_weight),
                         std::move(grad_bias));
}

XLATensor XLATensor::cast(const XLATensor& input, at::ScalarType dtype) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Cast>(input.GetIrValue(), dtype), dtype);
}

XLATensor XLATensor::log_softmax_backward(const XLATensor& grad_output,
                                          const XLATensor& output,
                                          xla::int64 dim) {
  return grad_output.CreateFrom(ir::MakeNode<ir::ops::LogSoftmaxBackward>(
      grad_output.GetIrValue(), output.GetIrValue(), dim));
}

XLATensor XLATensor::threshold_backward(const XLATensor& grad_output,
                                        const XLATensor& input,
                                        float threshold) {
  return grad_output.CreateFrom(ir::MakeNode<ir::ops::ThresholdBackward>(
      grad_output.GetIrValue(), input.GetIrValue(), threshold));
}

XLATensor XLATensor::t(const XLATensor& input) {
  return input.CreateFrom(ir::ops::TransposeOp(input.GetIrValue()));
}

void XLATensor::t_(XLATensor& input) {
  input.SetIrValue(ir::ops::TransposeOp(input.GetIrValue()));
}

XLATensor XLATensor::transpose(const XLATensor& input, xla::int64 dim0,
                               xla::int64 dim1) {
  xla::int64 rank = input.shape().get().rank();
  xla::int64 canonical_dim0 =
      XlaHelpers::GetCanonicalDimensionIndex(dim0, rank);
  xla::int64 canonical_dim1 =
      XlaHelpers::GetCanonicalDimensionIndex(dim1, rank);
  auto permute_dims = xla::util::Iota<xla::int64>(rank);
  std::swap(permute_dims[canonical_dim0], permute_dims[canonical_dim1]);
  return permute(input, permute_dims);
}

void XLATensor::transpose_(XLATensor& input, xla::int64 dim0, xla::int64 dim1) {
  xla::int64 rank = input.shape().get().rank();
  xla::int64 canonical_dim0 =
      XlaHelpers::GetCanonicalDimensionIndex(dim0, rank);
  xla::int64 canonical_dim1 =
      XlaHelpers::GetCanonicalDimensionIndex(dim1, rank);
  auto permute_dims = xla::util::Iota<xla::int64>(rank);
  std::swap(permute_dims[canonical_dim0], permute_dims[canonical_dim1]);
  input.SetIrValue(
      ir::MakeNode<ir::ops::Permute>(input.GetIrValue(), permute_dims));
}

XLATensor XLATensor::reshape(
    const XLATensor& input,
    tensorflow::gtl::ArraySlice<const xla::int64> output_size) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::View>(input.GetIrValue(), output_size));
}

XLATensor XLATensor::view(
    const XLATensor& input,
    tensorflow::gtl::ArraySlice<const xla::int64> output_size) {
  auto input_shape = input.shape();
  std::vector<xla::int64> complete_dimensions =
      GetCompleteShape(output_size, input_shape.get().dimensions());
  xla::Shape shape = MakeArrayShapeFromDimensions(
      complete_dimensions, input_shape.get().element_type(),
      input.GetDevice().hw_type);
  ViewInfo view_info(std::move(shape), input_shape.get().dimensions());
  return input.CreateView(std::move(view_info));
}

XLATensor XLATensor::narrow(const XLATensor& input, xla::int64 dim,
                            xla::int64 start, xla::int64 length) {
  auto input_shape = input.shape();
  xla::Shape narrow_shape = input_shape;
  narrow_shape.set_dimensions(dim, length);
  ViewInfo view_info(std::move(narrow_shape), input_shape.get().dimensions());
  view_info.indices[dim] = start;
  return input.CreateView(std::move(view_info));
}

XLATensor XLATensor::CreateView(ViewInfo view_info) const {
  if (data()->view != nullptr) {
    return Create(data()->view->CreateSubView(view_info.shape, view_info),
                  GetDevice(), dtype());
  }
  // This node is not a view, and creating a view forks the current node into
  // becoming one itself. This means creating an alias with the current IR
  // Node, and using the same alias for the created IR Node.
  ir::Value ir_value = GetIrValue();
  std::shared_ptr<Alias> alias = std::make_shared<Alias>(ir_value);
  ViewInfo this_view_info(ir_value.shape(), ir_value.shape().dimensions());
  data()->view = std::make_shared<View>(ir_value.shape(), alias,
                                        std::move(this_view_info));
  return Create(std::make_shared<View>(view_info.shape, alias, view_info),
                GetDevice(), dtype());
}

XLATensor XLATensor::log_softmax(const XLATensor& input, xla::int64 dim) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::LogSoftmax>(input.GetIrValue(), dim));
}

XLATensor XLATensor::softmax(const XLATensor& input, xla::int64 dim) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::Softmax>(input.GetIrValue(), dim));
}

XLATensor XLATensor::sigmoid(const XLATensor& input) {
  return input.CreateFrom(ir::ops::Sigmoid(input.GetIrValue()));
}

void XLATensor::sigmoid_(XLATensor& input) {
  input.SetIrValue(ir::ops::Sigmoid(input.GetIrValue()));
}

XLATensor XLATensor::nll_loss(const XLATensor& input, const XLATensor& target) {
  return input.CreateFrom(
      ir::ops::NllLossOp(input.GetIrValue(), target.GetIrValue()));
}

XLATensor XLATensor::nll_loss_backward(const XLATensor& input,
                                       const XLATensor& target) {
  return input.CreateFrom(
      ir::ops::NllLossBackwardOp(input.GetIrValue(), target.GetIrValue()));
}

XLATensor XLATensor::smooth_l1_loss(const XLATensor& input,
                                    const XLATensor& target,
                                    xla::int64 reduction) {
  return tensor_ops::SmoothL1Loss(input, target, reduction);
}

XLATensor XLATensor::smooth_l1_loss_backward(const XLATensor& grad_output,
                                             const XLATensor& input,
                                             const XLATensor& target,
                                             xla::int64 reduction) {
  return tensor_ops::SmoothL1LossBackward(grad_output, input, target,
                                          reduction);
}

XLATensor XLATensor::min(const XLATensor& input, const XLATensor& other) {
  return input.CreateFrom(ir::ops::Min(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::max(const XLATensor& input, const XLATensor& other) {
  return input.CreateFrom(ir::ops::Max(input.GetIrValue(), other.GetIrValue()));
}

XLATensor XLATensor::argmax(const XLATensor& input, xla::int64 dim,
                            bool keepdim) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::ArgMax>(input.GetIrValue(), dim, keepdim),
      at::ScalarType::Long);
}

XLATensor XLATensor::argmin(const XLATensor& input, xla::int64 dim,
                            bool keepdim) {
  return input.CreateFrom(
      ir::MakeNode<ir::ops::ArgMin>(input.GetIrValue(), dim, keepdim),
      at::ScalarType::Long);
}

XLATensor XLATensor::not_supported(std::string description, xla::Shape shape,
                                   const Device& device) {
  return Create(ir::MakeNode<ir::ops::NotSupported>(std::move(description),
                                                    std::move(shape)),
                device);
}

XLATensor XLATensor::cross_replica_sum(
    const std::vector<std::vector<xla::int64>>& groups) const {
  ir::NodePtr crs = ir::ops::CrossReplicaSumOp(GetIrValue(), groups);
  return Create(std::move(crs), data()->device, dtype());
}

XLATensor XLATensor::CreateFrom(ir::Value ir_value) const {
  return Create(std::move(ir_value), GetDevice(), dtype());
}

XLATensor XLATensor::CreateFrom(ir::Value ir_value,
                                const Device& device) const {
  return Create(std::move(ir_value), device, dtype());
}

XLATensor XLATensor::CreateFrom(ir::Value ir_value,
                                at::ScalarType logical_element_type) const {
  return Create(std::move(ir_value), GetDevice(), logical_element_type);
}

XLATensor XLATensor::CreateFrom(ir::Value ir_value, const Device& device,
                                at::ScalarType logical_element_type) const {
  return Create(std::move(ir_value), device, logical_element_type);
}

void XLATensor::ApplyPendingGraph() {
  // This method is called to ensure that the tensor data is available on
  // device, so that a call to CurrentXlaData() returns a valid pointer.
  if (CurrentXlaData() == nullptr) {
    ir::Value ir_value = CurrentIrValue();
    if (ir_value) {
      ir::LoweringContext lowering_ctx("ApplyPendingGraph");
      xla::XlaOp root = lowering_ctx.GetOutputOp(ir_value);
      xla::XlaComputation computation = ConsumeValue(lowering_ctx.Build(root));
      xla::Shape output_shape = shape().get();
      const xla::Shape computation_shape =
          ConsumeValue(computation.GetProgramShape()).result();
      // Some in-place operations (e.g. squeeze) can change the shape.
      if (!xla::ShapeUtil::Compatible(computation_shape, output_shape)) {
        output_shape =
            MakeShapeWithDeviceLayout(computation_shape, GetDevice().hw_type);
      }
      auto compiled_computation = xla::ComputationClient::Get()->Compile(
          std::move(computation), {GetDevice().ToString()}, &output_shape);
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
      if (tensors[i].CurrentIrValue().node != nullptr) {
        // Add only tensors which need to be synced.
        order.push_back(i);
      } else {
        // The tensor only has at::Tensor data. We need to queue it for a
        // device upload.
        c10::optional<at::Tensor> tensor_data = tensors[i].CurrentTensorData();
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
      // still valid so we leave it live on the XLA tensor (so that a
      // following ToTensor() does not need to fetch it from device).
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
  // to map back the unique IDs to the tensor indices within the tensors
  // vector.
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
          // If we do not find real device data (we have a cached graph
          // instead) at the given tensor, it means the cached information
          // does not apply anymore.
          XLA_COUNTER("NoTensorDataForUid", 1);
          return false;
        }
        device_parameters.push_back(xla_data.get());
      } else {
        // If we have not found the unique ID of the parameter which is
        // supposed to feed data to the computation, the pending graph context
        // changed, and the apply_context is no more valid.
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
        ir::Value ir_value = (*tensors)[i].CurrentIrValue();
        xla::XlaOp root = device_context->lowering_ctx.GetOutputOp(ir_value);
        device_context->lowering_ctx.AddResult(root);
        device_index_mapping.push_back((*tensors)[i].GetUniqueId());
      }
      index_mapping[index] = std::move(device_index_mapping);

      xla::XlaComputation computation =
          ConsumeValue(device_context->lowering_ctx.Build());
      xla::ProgramShape program_shape =
          ConsumeValue(computation.GetProgramShape());
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

xla::int64 XLATensor::GetCanonicalDimension(const XLATensor& input,
                                            xla::int64 dim) {
  return XlaHelpers::GetCanonicalDimensionIndex(dim,
                                                input.shape().get().rank());
}

void XLATensor::CheckRank(const XLATensor& t, xla::int64 expected_rank,
                          const std::string& tag, const std::string& arg_name,
                          int arg_number) {
  xla::int64 actual_rank = t.shape().get().rank();
  XLA_CHECK_EQ(actual_rank, expected_rank)
      << "Expected " << expected_rank << "-dimensional tensor, but got "
      << actual_rank << "-dimensional tensor for "
      << "argument #" << arg_number << " '" << arg_name << "'"
      << " (while checking arguments for " << tag << ")";
}

void XLATensor::CheckDimensionSize(const XLATensor& t, xla::int64 dim,
                                   xla::int64 expected_size,
                                   const std::string& tag,
                                   const std::string& arg_name,
                                   int arg_number) {
  xla::int64 dim_size = t.size(dim);
  XLA_CHECK_EQ(t.size(dim), expected_size)
      << "Expected tensor to have size " << expected_size << " at dimension "
      << dim << ", but got size " << dim_size << " for "
      << "argument #" << arg_number << " '" << arg_name << "'"
      << " (while checking arguments for " << tag << ")";
}

}  // namespace torch_xla
