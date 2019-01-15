#include "tensor.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <list>
#include <mutex>
#include <numeric>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "helpers.h"
#include "lowering_context.h"
#include "ops/cross_replica_sum.h"
#include "ops/device_data.h"
#include "ops/generic.h"
#include "ops/scalar.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"
#include "tensorflow/compiler/xla/xla_client/thread_pool.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
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

// Creates a minor-to-major layout from given dimensions.
xla::Shape MakeTorchTensorLayout(const std::vector<xla::int64>& dimensions,
                                 const xla::PrimitiveType type) {
  return xla::ShapeUtil::MakeShapeWithDescendingLayout(type, dimensions);
}

xla::Shape MakeArrayShapeFromDimensions(
    const at::IntList& tensor_dimensions, const xla::PrimitiveType type,
    const XLATensor::DeviceType device_type) {
  const auto dimensions = XlaHelpers::I64List(tensor_dimensions);
  if (dimensions.size() == 4 && device_type == XLATensor::DeviceType::TPU) {
    // Use a TPU-compatible layout for 4D tensors -- batch and feature in minor
    // dimensions.
    std::vector<xla::int64> hwcn_layout{0, 1, 3, 2};
    return xla::ShapeUtil::MakeShapeWithLayout(type, dimensions, hwcn_layout);
  }
  return MakeTorchTensorLayout(dimensions, type);
}

// Copies n bytes from source to dest, with different stride values for source
// and destination.
template <typename S, typename D>
void StridedCopy(D* dest, xla::int64 dest_stride, const S* source,
                 xla::int64 source_stride, xla::int64 n) {
  for (; n > 0; --n, dest += dest_stride, source += source_stride) {
    *dest = static_cast<D>(*source);
  }
}

// Computes the offset of the value at a given index, assuming a contiguous/flat
// tensor data representation.
template <typename S>
xla::int64 GetFlatTensorOffset(const S& strides,
                               const std::vector<xla::int64>& indices) {
  xla::int64 base = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    base += indices[i] * strides[i];
  }
  return base;
}

std::vector<xla::int64> GetXlaStrides(const xla::Shape& shape) {
  std::vector<xla::int64> strides(shape.rank());
  xla::int64 stride = 1;
  for (auto dim : shape.layout().minor_to_major()) {
    strides[dim] = stride;
    stride *= shape.dimensions(dim);
  }
  return strides;
}

template <typename D, typename S>
void CopyData(D* dest, const S* source, xla::int64 n) {
  StridedCopy(dest, 1, source, 1, n);
}

template <>
void CopyData<float, float>(float* dest, const float* source, xla::int64 n) {
  std::copy(source, source + n, dest);
}

template <>
void CopyData<xla::int64, int64_t>(xla::int64* dest, const int64_t* source,
                                   xla::int64 n) {
  std::copy(source, source + n, dest);
}

std::vector<xla::int64> GetIterationDimensions(const xla::Shape& shape) {
  // Return the most minor dimension order, to iterate the literal memory in a
  // cache friendly way.
  // Another strategy could be to return the higher value dimension first, to
  // reduce the number of outer loops in TensorToLiteral(), but that leads to
  // StridedCopy() calls in which both source and destination are jumping off
  // memory locations.
  return std::vector<xla::int64>(shape.layout().minor_to_major().begin(),
                                 shape.layout().minor_to_major().end());
}

template <typename AtenNative, typename XlaNative>
xla::Literal TensorToLiteral(const at::Tensor& tensor,
                             const xla::Shape& shape) {
  const at::Tensor& contiguous_tensor = tensor.contiguous();
  auto contiguous_ptr = contiguous_tensor.data<AtenNative>();
  const auto& tensor_sizes = contiguous_tensor.sizes();
  XLA_CHECK_EQ(tensor_sizes.size(), shape.rank());
  xla::int64 total_elements =
      std::accumulate(tensor_sizes.begin(), tensor_sizes.end(), 1,
                      std::multiplies<xla::int64>());
  xla::Literal literal(shape);
  auto literal_data = literal.data<XlaNative>();
  XLA_CHECK_EQ(literal_data.size(), total_elements);
  if (total_elements == 1 ||
      xla::LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    // The Torch tensor is array layout, and so is the literal. We can issue a
    // fast copy of the elements.
    CopyData<XlaNative, AtenNative>(literal_data.data(), contiguous_ptr,
                                    total_elements);
  } else {
    const auto& tensor_strides = contiguous_tensor.strides();
    const auto& xla_tensor_strides = GetXlaStrides(shape);
    std::vector<xla::int64> indices(tensor_sizes.size());
    std::vector<xla::int64> iter_dims = GetIterationDimensions(shape);
    xla::int64 n = 0;
    while (n < tensor_sizes.size()) {
      StridedCopy(literal_data.data() +
                      GetFlatTensorOffset(xla_tensor_strides, indices),
                  xla_tensor_strides[iter_dims.front()],
                  contiguous_ptr + GetFlatTensorOffset(tensor_strides, indices),
                  tensor_strides[iter_dims.front()],
                  shape.dimensions(iter_dims.front()));
      // Compute the next index. Skip the lower iteration dimension, as we loop
      // over it using the StridedCopy() call above.
      for (n = 1; n < iter_dims.size(); ++n) {
        xla::int64 dim = iter_dims[n];
        indices[dim] += 1;
        if (indices[dim] < shape.dimensions(dim)) {
          break;
        }
        indices[dim] = 0;
      }
    }
  }
  return literal;
}

std::shared_ptr<xla::ComputationClient::Data> TensorToXla(
    const at::Tensor& param_tensor, const xla::Shape& param_shape,
    const XLATensor::Device& device, xla::ComputationClient* client) {
  xla::Literal literal = GetTensorLiteral(param_tensor, &param_shape);
  std::vector<xla::ComputationClient::LiteralDevice> literal_device;
  literal_device.emplace_back(std::move(literal), device.ToString());
  auto handles = client->TransferToServer(literal_device);
  XLA_CHECK_EQ(handles.size(), 1);
  return std::move(handles.front());
}

at::Tensor MakeTensorFromXlaLiteral(const xla::Literal& literal) {
  const xla::Literal* literal_ptr = &literal;
  xla::Literal f32_literal;
  if (literal_ptr->shape().element_type() == xla::PrimitiveType::BF16) {
    // If ever PyTorch will support BF16, remove this cast to F32.
    f32_literal = xla::LiteralUtil::ConvertBF16ToF32(*literal_ptr);
    literal_ptr = &f32_literal;
  }
  std::vector<int64_t> dimensions;
  for (const auto result_dimension : literal_ptr->shape().dimensions()) {
    dimensions.push_back(result_dimension);
  }
  xla::Shape torch_shape = MakeTorchTensorLayout(
      XlaHelpers::I64List(dimensions), literal_ptr->shape().element_type());
  xla::Literal literal_with_torch_layout;
  if (!xla::ShapeUtil::Equal(literal_ptr->shape(), torch_shape)) {
    literal_with_torch_layout = literal_ptr->Relayout(torch_shape);
    literal_ptr = &literal_with_torch_layout;
  }
  switch (literal_ptr->shape().element_type()) {
    case xla::PrimitiveType::F32: {
      const auto result_slice = literal_ptr->data<float>();
      at::Tensor result_tensor =
          at::empty(dimensions, at::TensorOptions(at::kFloat));
      std::copy(result_slice.begin(), result_slice.end(),
                result_tensor.data<float>());
      return result_tensor;
    }
    case xla::PrimitiveType::S64: {
      const auto result_slice = literal_ptr->data<xla::int64>();
      at::Tensor result_tensor =
          at::empty(dimensions, at::TensorOptions(at::kLong));
      std::copy(result_slice.begin(), result_slice.end(),
                result_tensor.data<int64_t>());
      return result_tensor;
    }
    default:
      AT_ERROR("Unsupported literal type");
  }
}

std::string DeviceTypeToString(const XLATensor::DeviceType hw_type) {
  switch (hw_type) {
    case XLATensor::DeviceType::CPU:
      return "CPU";
    case XLATensor::DeviceType::GPU:
      return "GPU";
    case XLATensor::DeviceType::TPU:
      return "TPU";
  }
}

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

std::string XLATensor::Device::ToString() const {
  return absl::StrCat(DeviceTypeToString(hw_type), ":", ordinal);
}

std::shared_ptr<XLATensor> XLATensor::Create(
    const torch::autograd::Variable& tensor, const Device& device) {
  return TensorsArena::Get()->RegisterTensor(
      std::make_shared<XLATensor>(tensor, device));
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

XLATensor::XLATensor(const torch::autograd::Variable& tensor,
                     const Device& device)
    : data_(std::make_shared<Data>(
          TensorToXla(
              tensor,
              MakeArrayShapeFromDimensions(
                  tensor.sizes(),
                  XlaHelpers::MakeXlaPrimitiveType(tensor.type().scalarType()),
                  device.hw_type),
              device, XlaGetClient()),
          device)),
      requires_grad_(tensor.requires_grad()) {}

XLATensor::XLATensor(std::shared_ptr<xla::ComputationClient::Data> xla_data,
                     bool requires_grad)
    : data_(std::make_shared<Data>(xla_data,
                                   DeviceFromString(xla_data->device()))),
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
  xla::PrimitiveType xla_type = shape().element_type();
  switch (xla_type) {
    case xla::PrimitiveType::F32:
      return at::ScalarType::Float;
    case xla::PrimitiveType::S64:
      return at::ScalarType::Long;
    default:
      TF_LOG(FATAL) << "XLA type not supported: " << xla_type;
  }
}

const xla::Shape& XLATensor::shape() const {
  return data_->xla_data ? data_->xla_data->shape() : data_->ir_node->shape();
}

const XLATensor::Device& XLATensor::GetDevice() const { return data_->device; }

xla::int64 XLATensor::GetUniqueId() const { return data_->unique_id; }

const std::shared_ptr<xla::ComputationClient::Data>& XLATensor::GetXlaData() {
  ApplyPendingGraph();
  return data_->xla_data;
}

const std::shared_ptr<xla::ComputationClient::Data>& XLATensor::CurrentXlaData()
    const {
  return data_->xla_data;
}

std::string XLATensor::DumpGraphNodeComputation() const {
  std::string hlo_text;
  const ir::NodePtr& ir_node = CurrentIrNode();
  if (ir_node != nullptr) {
    ir::LoweringContext lowering_ctx;
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
}

void XLATensor::SetXlaGraphNode(ir::NodePtr ir_node) {
  data_->ir_node = std::move(ir_node);
  TryLimitGraphSize();
}

void XLATensor::TryLimitGraphSize() {
  static long count = 0;
  if (++count % 1000 == 0) {
    ApplyPendingGraph();
  }

  // If we are accumulating too many nodes in the pending graph, render the XLA
  // by executing the pending graph.
  // static const xla::int64 kMaxPendingGraphSize = 1000;
  // if (data_->ir_node != nullptr &&
  //     data_->ir_node->graph_size() > kMaxPendingGraphSize) {
  //   ApplyPendingGraph();
  // }
}

ir::NodePtr XLATensor::GetIrNode() const {
  return data_->ir_node ? data_->ir_node : CreateTensorNode(data_->xla_data);
}

const ir::NodePtr& XLATensor::CurrentIrNode() const { return data_->ir_node; }

void XLATensor::ReferenceDataFrom(const XLATensor& source) {
  XLA_CHECK_EQ(data_->device, source.data_->device);
  XLA_CHECK(xla::ShapeUtil::Equal(shape(), source.shape()))
      << shape() << " vs " << source.shape();

  data_->xla_data = source.data_->xla_data;
  data_->ir_node = source.data_->ir_node;
}

std::vector<int64_t> XLATensor::Size() const {
  const xla::Shape& tensor_shape = shape();
  return std::vector<int64_t>(tensor_shape.dimensions().begin(),
                              tensor_shape.dimensions().end());
}

at::Tensor XLATensor::toTensor() {
  ApplyPendingGraph();

  std::vector<xla::Literal> literals =
      XlaGetClient()->TransferFromServer({GetXlaData()});
  return torch::autograd::make_variable(
      MakeTensorFromXlaLiteral(literals.front()), RequiresGrad());
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
    tensors_data.push_back(tensor->GetXlaData());
  }
  std::vector<xla::Literal> literals =
      XlaGetClient()->TransferFromServer(tensors_data);
  std::vector<at::Tensor> results;
  for (size_t i = 0; i < literals.size(); ++i) {
    results.push_back(torch::autograd::make_variable(
        MakeTensorFromXlaLiteral(literals[i]), tensors[i]->RequiresGrad()));
  }
  return results;
}

std::vector<std::shared_ptr<XLATensor>> XLATensor::CreateTensors(
    const std::vector<torch::autograd::Variable>& tensors,
    const std::vector<std::string>& devices) {
  XLA_CHECK_EQ(tensors.size(), devices.size());
  std::vector<xla::ComputationClient::LiteralDevice> literal_device;
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto converter = [&, i]() -> xla::Literal {
      Device device = DeviceFromString(devices[i]);
      xla::Shape shape = MakeArrayShapeFromDimensions(
          tensors[i].sizes(),
          XlaHelpers::MakeXlaPrimitiveType(tensors[i].type().scalarType()),
          device.hw_type);
      return GetTensorLiteral(tensors[i], &shape);
    };
    literal_device.emplace_back(std::move(converter), devices[i]);
  }
  auto handles = XlaGetClient()->TransferToServer(literal_device);
  std::vector<std::shared_ptr<XLATensor>> xla_tensors;
  for (size_t i = 0; i < handles.size(); ++i) {
    xla_tensors.push_back(
        Create(std::move(handles[i]), tensors[i].requires_grad()));
  }
  return xla_tensors;
}

ir::NodePtr XLATensor::CreateTensorNode(
    std::shared_ptr<xla::ComputationClient::Data> data) {
  return std::make_shared<ir::ops::DeviceData>(std::move(data));
}

xla::int64 XLATensor::GetNextTensorId() {
  static std::atomic<xla::int64>* id_generator = new std::atomic<xla::int64>(1);
  return id_generator->fetch_add(1);
}

ir::NodePtr XLATensor::CreateMulNode(const ir::NodePtr& node0,
                                     const ir::NodePtr& node1) {
  auto lower_fn = [](const ir::Node& node,
                     ir::LoweringContext* loctx) -> ir::XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedMul(op0, op1), loctx);
  };
  return std::make_shared<ir::ops::Generic>(
      ir::OpKind(at::aten::mul),
      ir::OpList{ir::NodeOperand(node0), ir::NodeOperand(node1)},
      XlaHelpers::GetPromotedShape(node0->shape(), node1->shape()),
      std::move(lower_fn));
}

ir::NodePtr XLATensor::CreateDivNode(const ir::NodePtr& node0,
                                     const ir::NodePtr& node1) {
  auto lower_fn = [](const ir::Node& node,
                     ir::LoweringContext* loctx) -> ir::XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedDiv(op0, op1), loctx);
  };
  return std::make_shared<ir::ops::Generic>(
      ir::OpKind(at::aten::div),
      ir::OpList{ir::NodeOperand(node0), ir::NodeOperand(node1)},
      XlaHelpers::GetPromotedShape(node0->shape(), node1->shape()),
      std::move(lower_fn));
}

ir::NodePtr XLATensor::CreateAddNode(const ir::NodePtr& node0,
                                     const ir::NodePtr& node1) {
  auto lower_fn = [](const ir::Node& node,
                     ir::LoweringContext* loctx) -> ir::XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedAdd(op0, op1), loctx);
  };
  return std::make_shared<ir::ops::Generic>(
      ir::OpKind(at::aten::add),
      ir::OpList{ir::NodeOperand(node0), ir::NodeOperand(node1)},
      XlaHelpers::GetPromotedShape(node0->shape(), node1->shape()),
      std::move(lower_fn));
}

ir::NodePtr XLATensor::CreateAddNode(const XLATensor& other,
                                     const at::Scalar& alpha) {
  ir::NodePtr constant =
      std::make_shared<ir::ops::Scalar>(alpha.toDouble(), other.shape());
  ir::NodePtr mul = CreateMulNode(other.GetIrNode(), constant);
  return CreateAddNode(GetIrNode(), mul);
}

std::shared_ptr<XLATensor> XLATensor::add(const XLATensor& other,
                                          const at::Scalar& alpha) {
  return Create(CreateAddNode(other, alpha), data_->device);
}

void XLATensor::add_(const XLATensor& other, const at::Scalar& alpha) {
  SetXlaGraphNode(CreateAddNode(other, alpha));
}

std::shared_ptr<XLATensor> XLATensor::mul(const XLATensor& other) {
  return Create(CreateMulNode(GetIrNode(), other.GetIrNode()), data_->device);
}

std::shared_ptr<XLATensor> XLATensor::mul(const at::Scalar& other) {
  ir::NodePtr constant =
      std::make_shared<ir::ops::Scalar>(other.toDouble(), shape());
  return Create(CreateMulNode(GetIrNode(), constant), data_->device);
}

void XLATensor::mul_(const XLATensor& other) {
  SetXlaGraphNode(CreateMulNode(GetIrNode(), other.GetIrNode()));
}

void XLATensor::mul_(const at::Scalar& other) {
  ir::NodePtr constant =
      std::make_shared<ir::ops::Scalar>(other.toDouble(), shape());
  SetXlaGraphNode(CreateMulNode(GetIrNode(), constant));
}

std::shared_ptr<XLATensor> XLATensor::div(const XLATensor& other) {
  return Create(CreateDivNode(GetIrNode(), other.GetIrNode()), data_->device);
}

std::shared_ptr<XLATensor> XLATensor::div(const at::Scalar& other) {
  ir::NodePtr constant =
      std::make_shared<ir::ops::Scalar>(other.toDouble(), shape());
  return Create(CreateDivNode(GetIrNode(), constant), data_->device);
}

void XLATensor::div_(const XLATensor& other) {
  SetXlaGraphNode(CreateDivNode(GetIrNode(), other.GetIrNode()));
}

void XLATensor::div_(const at::Scalar& other) {
  ir::NodePtr constant =
      std::make_shared<ir::ops::Scalar>(other.toDouble(), shape());
  SetXlaGraphNode(CreateDivNode(GetIrNode(), constant));
}

void XLATensor::zero_() {
  SetXlaGraphNode(std::make_shared<ir::ops::Scalar>(0.0, shape()));
}

void XLATensor::addcdiv_(const at::Scalar& value, const XLATensor& tensor1,
                         const XLATensor& tensor2) {
  ir::NodePtr constant = std::make_shared<ir::ops::Scalar>(
      value.toDouble(), tensor1.shape().element_type());
  ir::NodePtr div = CreateDivNode(tensor1.GetIrNode(), tensor2.GetIrNode());
  ir::NodePtr scaled = CreateMulNode(div, constant);
  SetXlaGraphNode(CreateAddNode(GetIrNode(), scaled));
}

void XLATensor::addcmul_(const at::Scalar& value, const XLATensor& tensor1,
                         const XLATensor& tensor2) {
  ir::NodePtr constant = std::make_shared<ir::ops::Scalar>(
      value.toDouble(), tensor1.shape().element_type());
  ir::NodePtr div = CreateMulNode(tensor1.GetIrNode(), tensor2.GetIrNode());
  ir::NodePtr scaled = CreateMulNode(div, constant);
  SetXlaGraphNode(CreateAddNode(GetIrNode(), scaled));
}

std::shared_ptr<XLATensor> XLATensor::cross_replica_sum(
    const std::vector<std::vector<xla::int64>>& groups) {
  ir::NodePtr crs = std::make_shared<ir::ops::CrossReplicaSum>(
      ir::NodeOperand(GetIrNode()), groups);
  return Create(std::move(crs), data_->device);
}

void XLATensor::ApplyPendingGraph() {
  const ir::NodePtr& ir_node = CurrentIrNode();
  if (ir_node != nullptr) {
    ir::LoweringContext lowering_ctx;
    XLA_CHECK_EQ(ir_node->num_outputs(), 1);
    xla::XlaOp root = lowering_ctx.GetOutputOp(ir::Output(ir_node.get(), 0));
    xla::XlaComputation computation =
        lowering_ctx.Build(root).ConsumeValueOrDie();
    auto compiled_computation = XlaGetClient()->Compile(
        std::move(computation), {GetDevice().ToString()},
        /*output_shape=*/nullptr);
    xla::ComputationClient::ExecuteComputationOptions options;
    options.explode_tuple = false;
    auto results = XlaGetClient()->ExecuteComputation(
        *compiled_computation, lowering_ctx.GetParametersData(),
        compiled_computation->devices()[0], options);
    XLA_CHECK_EQ(results.size(), 1);
    SetXlaData(results.front());
  }
}

std::vector<size_t> XLATensor::GetApplyOrder(
    const std::vector<std::shared_ptr<XLATensor>>& tensors) {
  std::vector<size_t> order;
  order.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i]->CurrentIrNode() != nullptr) {
      // Add only tensors which need to be synced.
      order.push_back(i);
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
        auto& xla_data = tensors[it->second]->CurrentXlaData();
        if (xla_data == nullptr) {
          // If we do not find real device data (we have a cached graph instead)
          // at the given tensor, it means the cached information does not apply
          // anymore.
          return false;
        }
        device_parameters.push_back(xla_data.get());
      } else {
        // If we have not found the unique ID of the parameter which is supposed
        // to feed data to the computation, the pending graph context changed,
        // and the apply_context is no more valid.
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
        return false;
      }
    }
    index_mapping.push_back(std::move(current_index_mapping));
  }

  xla::ComputationClient::ExecuteParallelOptions options;
  auto results = XlaGetClient()->ExecuteParallel(
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
    auto& xla_data = tensors[i]->CurrentXlaData();
    if (xla_data != nullptr) {
      auto it_inserted =
          data_uid_map.emplace(xla_data.get(), tensors[i]->GetUniqueId());
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
          auto it = data_uid_map.find(data);
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
    computations = XlaGetClient()->Compile(std::move(instances));

    xla::ComputationClient::ExecuteParallelOptions options;
    auto results = XlaGetClient()->ExecuteParallel(
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

XLATensor::Device XLATensor::DeviceFromString(const std::string& device_spec) {
  if (device_spec.empty()) {
    const std::string default_device_spec = XlaGetClient()->GetDefaultDevice();
    XLA_CHECK(!default_device_spec.empty());
    return DeviceFromString(default_device_spec);
  }
  if (device_spec[0] == ':') {
    const std::string default_device_spec = XlaGetClient()->GetDefaultDevice();
    auto pos = default_device_spec.find(':');
    XLA_CHECK_NE(pos, std::string::npos) << default_device_spec;
    return DeviceFromString(default_device_spec.substr(0, pos) + device_spec);
  }
  std::vector<std::string> device_spec_parts = absl::StrSplit(device_spec, ':');
  std::string invalid_device_error =
      "Invalid device specification: " + device_spec;
  if (device_spec_parts.size() != 2) {
    AT_ERROR(invalid_device_error);
  }
  int device_ordinal = std::stoi(device_spec_parts[1]);
  std::string device_hw_type = device_spec_parts[0];
  if (device_hw_type == "CPU") {
    return {XLATensor::DeviceType::CPU, device_ordinal};
  }
  if (device_hw_type == "GPU") {
    return {XLATensor::DeviceType::GPU, device_ordinal};
  }
  if (device_hw_type == "TPU") {
    return {XLATensor::DeviceType::TPU, device_ordinal};
  }
  AT_ERROR(invalid_device_error);
}

XLATensor::Device XLATensor::CommonDeviceForTensors(
    const std::vector<std::shared_ptr<XLATensor>>& tensors) {
  XLA_CHECK(!tensors.empty());
  const XLATensor::Device& device = tensors.front()->GetDevice();
  for (const auto& tensor : tensors) {
    const XLATensor::Device& tensor_device = tensor->GetDevice();
    if (tensor_device != device) {
      AT_ERROR("All input tensors should have the same device");
    }
  }
  return device;
}

xla::Literal GetTensorLiteral(const at::Tensor& tensor,
                              const xla::Shape* shape) {
  xla::Shape computed_shape;
  if (shape == nullptr) {
    auto dimensions = XlaHelpers::I64List(tensor.sizes());
    computed_shape = MakeTorchTensorLayout(
        dimensions,
        XlaHelpers::MakeXlaPrimitiveType(tensor.type().scalarType()));
    shape = &computed_shape;
  }
  switch (tensor.type().scalarType()) {
    case at::ScalarType::Float:
      if (shape->element_type() == xla::PrimitiveType::BF16) {
        return TensorToLiteral<float, tensorflow::bfloat16>(tensor, *shape);
      }
      return TensorToLiteral<float, float>(tensor, *shape);
    case at::ScalarType::Long:
      return TensorToLiteral<int64_t, xla::int64>(tensor, *shape);
    default:
      TF_LOG(FATAL) << "Tensor type not supported";
  }
}

std::vector<xla::Shape> GetComponentShapes(const xla::Shape& shape) {
  std::vector<xla::Shape> component_shapes;
  if (shape.IsTuple()) {
    for (const xla::Shape& component_shape : shape.tuple_shapes()) {
      XLA_CHECK(!component_shape.IsTuple());
      component_shapes.push_back(component_shape);
    }
  } else {
    component_shapes.push_back(shape);
  }
  return component_shapes;
}

xla::Shape MakeShapeWithDeviceLayout(const xla::Shape& shape,
                                     const XLATensor::DeviceType device_type) {
  std::vector<xla::Shape> shape_components = GetComponentShapes(shape);
  std::vector<xla::Shape> shape_components_with_layout;
  XLA_CHECK(!shape_components.empty());
  for (const auto& shape_component : shape_components) {
    std::vector<int64_t> shape_component_dimensions(
        shape_component.dimensions().begin(),
        shape_component.dimensions().end());
    shape_components_with_layout.push_back(MakeArrayShapeFromDimensions(
        shape_component_dimensions, shape_component.element_type(),
        device_type));
  }
  return shape_components_with_layout.size() > 1
             ? xla::ShapeUtil::MakeTupleShape(shape_components_with_layout)
             : shape_components_with_layout.front();
}

}  // namespace torch_xla
