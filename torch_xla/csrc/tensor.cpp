#include "tensor.h"

#include <algorithm>
#include <functional>
#include <list>
#include <mutex>
#include <numeric>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "helpers.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"
#include "torch/csrc/autograd/variable.h"
#include "translator.h"

namespace torch {
namespace jit {

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
  std::vector<xla::int64> strides(xla::ShapeUtil::Rank(shape));
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
  XLA_CHECK_EQ(tensor_sizes.size(), xla::ShapeUtil::Rank(shape));
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
              std::vector<std::shared_ptr<xla::ComputationClient::Data>>&
                  new_dest_elements,
              const std::vector<xla::int64>& index_mapping) {
  XLA_CHECK_EQ(index_mapping.size(), new_dest_elements.size());
  // Replace the underlying data for the destination tensors with the data in
  // "new_dest_elements".
  for (size_t i = 0; i < new_dest_elements.size(); ++i) {
    xla::int64 dest_tuple_index = index_mapping[i];
    dest_tuple[dest_tuple_index]->SetXlaData(std::move(new_dest_elements[i]));
  }
}

}  // namespace

std::string XLATensor::Device::ToString() const {
  return absl::StrCat(DeviceTypeToString(hw_type), ":", ordinal);
}

std::shared_ptr<XLATensor> XLATensor::Create(const autograd::Variable& tensor,
                                             const Device& device) {
  return TensorsArena::Get()->RegisterTensor(
      std::make_shared<XLATensor>(tensor, device));
}

std::shared_ptr<XLATensor> XLATensor::Create(
    std::shared_ptr<xla::ComputationClient::Data> xla_data,
    bool requires_grad) {
  return TensorsArena::Get()->RegisterTensor(
      std::make_shared<XLATensor>(std::move(xla_data), requires_grad));
}

std::shared_ptr<XLATensor> XLATensor::Create(
    std::shared_ptr<XlaGraphNode> xla_graph_node, const Device& device) {
  return TensorsArena::Get()->RegisterTensor(
      std::make_shared<XLATensor>(std::move(xla_graph_node), device));
}

std::shared_ptr<XLATensor> XLATensor::Create(std::shared_ptr<Data> data) {
  return TensorsArena::Get()->RegisterTensor(
      std::make_shared<XLATensor>(std::move(data)));
}

XLATensor::~XLATensor() { TensorsArena::Get()->UnregisterTensor(this); }

XLATensor::XLATensor(const autograd::Variable& tensor, const Device& device)
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

XLATensor::XLATensor(std::shared_ptr<XlaGraphNode> xla_graph_node,
                     const Device& device)
    : data_(std::make_shared<Data>(std::move(xla_graph_node), device)) {
  TryLimitGraphSize();
}

void XLATensor::MulAddMulti(
    const double scale_dest,
    const std::vector<std::shared_ptr<XLATensor>>& dest_tuple,
    const double alpha,
    const std::vector<std::shared_ptr<XLATensor>>& source_tuple) {
  XLA_CHECK_EQ(dest_tuple.size(), source_tuple.size());
  XlaGraphContext xla_graph_ctx(/*collate_parameters=*/true);
  for (size_t i = 0; i < dest_tuple.size(); ++i) {
    auto dest_node = dest_tuple[i]->GetXlaGraphNode();
    auto source_node = source_tuple[i]->GetXlaGraphNode();
    auto dest_node_op = dest_node->Generate(&xla_graph_ctx);
    auto source_node_op = source_node->Generate(&xla_graph_ctx);
    if (alpha != 1) {
      const auto alpha_source = XlaHelpers::ScalarBroadcast<float>(
          alpha, source_tuple[i]->shape(), xla_graph_ctx.builder());
      source_node_op = xla::Mul(source_node_op, alpha_source);
    }
    if (scale_dest != 1) {
      const auto scale_dest_broadcast = XlaHelpers::ScalarBroadcast<float>(
          scale_dest, dest_tuple[i]->shape(), xla_graph_ctx.builder());
      dest_node_op = xla::Mul(dest_node_op, scale_dest_broadcast);
    }
    xla_graph_ctx.AddResult(xla::Add(dest_node_op, source_node_op));
  }
  std::vector<xla::int64> index_mapping(dest_tuple.size());
  std::iota(index_mapping.begin(), index_mapping.end(), 0);
  ComputeAndDistribute(&xla_graph_ctx, index_mapping, dest_tuple);
}

void XLATensor::ZeroMulti(
    const std::vector<std::shared_ptr<XLATensor>>& dest_tuple) {
  if (dest_tuple.empty()) {
    return;
  }
  // Create a computation which returns zeroes shaped the same as tensors in
  // "dest_tuple".
  XlaGraphContext xla_graph_ctx(/*collate_parameters=*/true);
  for (auto& dest : dest_tuple) {
    const auto dest_shape = dest->shape();
    const auto zero =
        xla::ConstantLiteral(xla_graph_ctx.builder(),
                             xla::LiteralUtil::Zero(dest_shape.element_type()));
    xla_graph_ctx.AddResult(
        Broadcast(zero, XlaHelpers::ShapeSizes(dest_shape)));
  }
  std::vector<xla::int64> index_mapping(dest_tuple.size());
  std::iota(index_mapping.begin(), index_mapping.end(), 0);
  ComputeAndDistribute(&xla_graph_ctx, index_mapping, dest_tuple);
}

std::shared_ptr<XLATensor> XLATensor::grad() const { return data_->grad; }

void XLATensor::setGrad(std::shared_ptr<XLATensor> grad) {
  data_->grad = std::move(grad);
}

at::ScalarType XLATensor::dtype() const {
  xla::PrimitiveType xla_type = shape().element_type();
  switch (xla_type) {
    case xla::PrimitiveType::F32:
      return at::ScalarType::Float;
    case xla::PrimitiveType::S64:
      return at::ScalarType::Long;
    default:
      LOG(FATAL) << "XLA type not supported: " << xla_type;
  }
}

const xla::Shape& XLATensor::shape() const {
  return data_->xla_data ? data_->xla_data->shape()
                         : data_->xla_graph_node->shape();
}

const XLATensor::Device& XLATensor::GetDevice() const { return data_->device; }

const std::shared_ptr<xla::ComputationClient::Data>& XLATensor::GetXlaData() {
  ApplyPendingGraph();
  return data_->xla_data;
}

std::string XLATensor::DumpGraphNodeComputation() const {
  std::string hlo_text;
  auto& xla_graph_node = current_xla_graph_node();
  if (xla_graph_node != nullptr) {
    XlaGraphContext xla_graph_ctx(/*collate_parameters=*/true);
    auto root = xla_graph_node->Generate(&xla_graph_ctx);
    auto computation = xla_graph_ctx.Build(root).ConsumeValueOrDie();
    hlo_text =
        xla::xrt_util::GetComputationHloText(computation).ConsumeValueOrDie();
  }
  return hlo_text;
}

void XLATensor::SetXlaData(
    std::shared_ptr<xla::ComputationClient::Data> xla_data) {
  XLA_CHECK(xla::ShapeUtil::Equal(shape(), xla_data->shape()))
      << xla::ShapeUtil::HumanStringWithLayout(shape()) << " vs "
      << xla::ShapeUtil::HumanStringWithLayout(xla_data->shape()) << "\n"
      << DumpGraphNodeComputation();
  data_->xla_data = std::move(xla_data);
  data_->xla_graph_node = nullptr;
}

void XLATensor::SetXlaGraphNode(std::shared_ptr<XlaGraphNode> xla_graph_node) {
  data_->xla_graph_node = std::move(xla_graph_node);
  TryLimitGraphSize();
}

void XLATensor::TryLimitGraphSize() {
  // If we are accumulating too many nodes in the pending graph, render the XLA
  // by executing the pending graph.
  static const xla::int64 kMaxPendingGraphSize = 1000;
  if (data_->xla_graph_node != nullptr &&
      data_->xla_graph_node->graph_size() > kMaxPendingGraphSize) {
    ApplyPendingGraph();
  }
}

std::shared_ptr<XlaGraphNode> XLATensor::GetXlaGraphNode() const {
  return data_->xla_graph_node ? data_->xla_graph_node
                               : CreateTensorNode(data_->xla_data);
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
  return autograd::make_variable(MakeTensorFromXlaLiteral(literals.front()),
                                 RequiresGrad());
}

std::vector<std::shared_ptr<XLATensor>> XLATensor::GetLiveTensors() {
  return TensorsArena::Get()->GetTensors();
}

std::vector<at::Tensor> XLATensor::GetTensors(
    const std::vector<std::shared_ptr<XLATensor>>& tensors) {
  // TODO(dlibenzi): We do apply/compute and then fetch. Changing the API to
  // support getting handles and data might save a few pennies here.
  ApplyPendingGraph(tensors);

  std::vector<std::shared_ptr<xla::ComputationClient::Data>> tensors_data;
  for (auto& tensor : tensors) {
    tensors_data.push_back(tensor->GetXlaData());
  }
  std::vector<xla::Literal> literals =
      XlaGetClient()->TransferFromServer(tensors_data);
  std::vector<at::Tensor> results;
  for (size_t i = 0; i < literals.size(); ++i) {
    results.push_back(autograd::make_variable(
        MakeTensorFromXlaLiteral(literals[i]), tensors[i]->RequiresGrad()));
  }
  return results;
}

std::vector<std::shared_ptr<XLATensor>> XLATensor::CreateTensors(
    const std::vector<autograd::Variable>& tensors,
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

std::shared_ptr<XlaGraphNode> XLATensor::CreateTensorNode(
    std::shared_ptr<xla::ComputationClient::Data> data) {
  auto generator = [data](XlaGraphContext* ctx,
                          const XlaGraphNode&) -> xla::XlaOp {
    return ctx->GetParameter(data);
  };
  return XlaGraphNode::New(std::move(generator), data->shape(), {});
}

std::shared_ptr<XlaGraphNode> XLATensor::CreateMulNode(XLATensor& other) {
  auto generator = [](XlaGraphContext* ctx,
                      const XlaGraphNode& node) -> xla::XlaOp {
    auto node_op = node.input(0)->Generate(ctx);
    auto other_node_op = node.input(1)->Generate(ctx);
    return node_op * other_node_op;
  };
  return XlaGraphNode::New(std::move(generator), shape(),
                           {GetXlaGraphNode(), other.GetXlaGraphNode()});
}

std::shared_ptr<XlaGraphNode> XLATensor::CreateMulNode(
    const at::Scalar& other) {
  auto generator = [other](XlaGraphContext* ctx,
                           const XlaGraphNode& node) -> xla::XlaOp {
    auto node_op = node.input(0)->Generate(ctx);
    return node_op * XlaHelpers::ScalarBroadcast<float>(other.toDouble(),
                                                        node.input(0)->shape(),
                                                        ctx->builder());
  };
  return XlaGraphNode::New(std::move(generator), shape(), {GetXlaGraphNode()});
}

std::shared_ptr<XlaGraphNode> XLATensor::CreateDivNode(XLATensor& other) {
  auto generator = [](XlaGraphContext* ctx,
                      const XlaGraphNode& node) -> xla::XlaOp {
    auto node_op = node.input(0)->Generate(ctx);
    auto other_node_op = node.input(1)->Generate(ctx);
    return node_op / other_node_op;
  };
  return XlaGraphNode::New(std::move(generator), shape(),
                           {GetXlaGraphNode(), other.GetXlaGraphNode()});
}

std::shared_ptr<XlaGraphNode> XLATensor::CreateDivNode(
    const at::Scalar& other) {
  auto generator = [other](XlaGraphContext* ctx,
                           const XlaGraphNode& node) -> xla::XlaOp {
    auto node_op = node.input(0)->Generate(ctx);
    return node_op / XlaHelpers::ScalarBroadcast<float>(other.toDouble(),
                                                        node.input(0)->shape(),
                                                        ctx->builder());
  };
  return XlaGraphNode::New(std::move(generator), shape(), {GetXlaGraphNode()});
}

std::shared_ptr<XlaGraphNode> XLATensor::CreateAddNode(
    XLATensor& other, const at::Scalar& alpha) {
  auto generator = [alpha](XlaGraphContext* ctx,
                           const XlaGraphNode& node) -> xla::XlaOp {
    auto node_op = node.input(0)->Generate(ctx);
    auto other_node_op = node.input(1)->Generate(ctx);
    return node_op + other_node_op * XlaHelpers::ScalarBroadcast<float>(
                                         alpha.toDouble(),
                                         node.input(0)->shape(),
                                         ctx->builder());
  };
  return XlaGraphNode::New(std::move(generator), shape(),
                           {GetXlaGraphNode(), other.GetXlaGraphNode()});
}

std::shared_ptr<XLATensor> XLATensor::add(XLATensor& other,
                                          const at::Scalar& alpha) {
  return Create(CreateAddNode(other, alpha), data_->device);
}

void XLATensor::add_(XLATensor& other, const at::Scalar& alpha) {
  SetXlaGraphNode(CreateAddNode(other, alpha));
}

std::shared_ptr<XLATensor> XLATensor::mul(XLATensor& other) {
  return Create(CreateMulNode(other), data_->device);
}

std::shared_ptr<XLATensor> XLATensor::mul(const at::Scalar& other) {
  return Create(CreateMulNode(other), data_->device);
}

void XLATensor::mul_(XLATensor& other) {
  SetXlaGraphNode(CreateMulNode(other));
}

void XLATensor::mul_(const at::Scalar& other) {
  SetXlaGraphNode(CreateMulNode(other));
}

std::shared_ptr<XLATensor> XLATensor::div(XLATensor& other) {
  return Create(CreateDivNode(other), data_->device);
}

std::shared_ptr<XLATensor> XLATensor::div(const at::Scalar& other) {
  return Create(CreateDivNode(other), data_->device);
}

void XLATensor::div_(XLATensor& other) {
  SetXlaGraphNode(CreateDivNode(other));
}

void XLATensor::div_(const at::Scalar& other) {
  SetXlaGraphNode(CreateDivNode(other));
}

void XLATensor::zero_() {
  xla::Shape tensor_shape = shape();
  auto generator = [tensor_shape](XlaGraphContext* ctx,
                                  const XlaGraphNode&) -> xla::XlaOp {
    auto zero_literal = xla::LiteralUtil::Zero(tensor_shape.element_type());
    auto const_zero = xla::ConstantLiteral(ctx->builder(), zero_literal);
    return xla::Broadcast(const_zero, XlaHelpers::ShapeSizes(tensor_shape));
  };
  SetXlaGraphNode(XlaGraphNode::New(std::move(generator), tensor_shape, {}));
}

std::shared_ptr<XLATensor> XLATensor::cross_replica_sum(
    const std::vector<std::vector<xla::int64>>& groups) {
  auto generator = [groups](XlaGraphContext* ctx,
                            const XlaGraphNode& node) -> xla::XlaOp {
    std::vector<xla::ReplicaGroup> crs_groups;
    for (auto& group : groups) {
      xla::ReplicaGroup rgroup;
      for (auto replica_id : group) {
        rgroup.add_replica_ids(replica_id);
      }
      crs_groups.push_back(std::move(rgroup));
    }
    auto node_op = node.input(0)->Generate(ctx);
    return xla::CrossReplicaSum(node_op, crs_groups);
  };
  auto crs_node =
      XlaGraphNode::New(std::move(generator), shape(), {GetXlaGraphNode()});
  return Create(std::move(crs_node), data_->device);
}

void XLATensor::ApplyPendingGraph() {
  auto& xla_graph_node = current_xla_graph_node();
  if (xla_graph_node != nullptr) {
    XlaGraphContext xla_graph_ctx(/*collate_parameters=*/true);
    auto root = xla_graph_node->Generate(&xla_graph_ctx);
    auto computation = xla_graph_ctx.Build(root).ConsumeValueOrDie();
    SetXlaData(XlaGetClient()->ExecuteComputation(
        computation, xla_graph_ctx.GetParametersData(), GetDevice().ToString(),
        nullptr));
  }
}

void XLATensor::ComputeAndDistribute(
    XlaGraphContext* xla_graph_ctx,
    const std::vector<xla::int64>& index_mapping,
    const std::vector<std::shared_ptr<XLATensor>>& tensors) {
  auto computation = xla_graph_ctx->Build().ValueOrDie();
  auto program_shape = computation.GetProgramShape().ValueOrDie();
  const auto device = CommonDeviceForTensors(tensors);
  const auto multi_shape =
      MakeShapeWithDeviceLayout(program_shape.result(), device.hw_type);
  auto client = XlaGetClient();
  auto result_tuple = client->ExecuteComputation(
      computation, xla_graph_ctx->GetParametersData(), device.ToString(),
      &multi_shape);
  auto new_dest_elements = client->DeconstructTuple({result_tuple});
  // Replace destination's underlying data with the result of the computation.
  SetMulti(tensors, new_dest_elements.front(), index_mapping);
}

std::vector<size_t> XLATensor::GetTensorsOrder(
    const std::vector<std::shared_ptr<XLATensor>>& tensors) {
  // The ApplyPendingGraph() API is getting a bunch of tensors, and needs to
  // create a computation to sync the pending XLA operations on device memory.
  // The tensors passed ApplyPendingGraph() tends to be logically the same, but
  // different generations of them.
  // In order to avoid creating different XLA computations every time, we need
  // to find a stable order on the tensors. This works correctly if the tensors
  // being passed are really logically the same but of different generations.
  struct TensorMetadata {
    TensorMetadata(std::string data, size_t index)
        : hash(std::hash<std::string>()(data)),
          index(index),
          data(std::move(data)) {}
    size_t hash;
    size_t index;
    std::string data;
  };
  std::vector<TensorMetadata> tensor_meta;
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto& xla_graph_node = tensors[i]->current_xla_graph_node();
    if (xla_graph_node != nullptr) {
      XlaGraphContext xla_graph_ctx(/*collate_parameters=*/false);
      auto root = xla_graph_node->Generate(&xla_graph_ctx);
      auto computation = xla_graph_ctx.Build(root).ConsumeValueOrDie();
      std::string data;
      XLA_CHECK(tensorflow::SerializeToStringDeterministic(computation.proto(),
                                                           &data));
      tensor_meta.emplace_back(std::move(data), i);
    }
  }
  std::sort(tensor_meta.begin(), tensor_meta.end(),
            [](const TensorMetadata& tm1, const TensorMetadata& tm2) {
              if (tm1.hash != tm2.hash) {
                return tm1.hash < tm2.hash;
              }
              return tm1.data < tm2.data;
            });
  std::vector<size_t> order;
  for (auto& meta : tensor_meta) {
    order.push_back(meta.index);
  }
  return order;
}

void XLATensor::ApplyPendingGraph(
    const std::vector<std::shared_ptr<XLATensor>>& tensors) {
  struct DeviceContext {
    DeviceContext() : xla_graph_ctx(/*collate_parameters=*/false) {}

    XlaGraphContext xla_graph_ctx;
    std::vector<xla::int64> index_mapping;
  };
  std::map<Device, DeviceContext> contexts_map;
  std::vector<size_t> order = GetTensorsOrder(tensors);
  for (auto i : order) {
    auto& xla_graph_node = tensors[i]->current_xla_graph_node();
    if (xla_graph_node != nullptr) {
      DeviceContext* device_context = &contexts_map[tensors[i]->GetDevice()];
      auto root = xla_graph_node->Generate(&device_context->xla_graph_ctx);
      device_context->xla_graph_ctx.AddResult(root);
      device_context->index_mapping.push_back(i);
    }
  }
  if (!contexts_map.empty()) {
    std::vector<xla::XlaComputation> computations;
    std::vector<std::vector<xla::ComputationClient::Data*>> parameters;
    std::list<xla::Shape> shapes;
    std::vector<std::string> devices;
    std::vector<const xla::Shape*> output_shapes;
    for (auto& device_context : contexts_map) {
      computations.push_back(
          device_context.second.xla_graph_ctx.Build().ConsumeValueOrDie());
      auto program_shape = computations.back().GetProgramShape().ValueOrDie();
      shapes.push_back(MakeShapeWithDeviceLayout(program_shape.result(),
                                                 device_context.first.hw_type));
      output_shapes.push_back(&shapes.back());
      devices.push_back(device_context.first.ToString());
      parameters.push_back(
          device_context.second.xla_graph_ctx.GetParametersData());
    }
    auto client = XlaGetClient();
    auto result_tuples = client->ExecuteParallel(computations, parameters,
                                                 devices, output_shapes);
    auto result_tuple_elements = client->DeconstructTuple(result_tuples);
    auto context_iterator = contexts_map.begin();
    for (auto& computation_tuple_elements : result_tuple_elements) {
      // Replace destination's underlying data with the result of the
      // computation.
      SetMulti(tensors, computation_tuple_elements,
               context_iterator->second.index_mapping);
      ++context_iterator;
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
      LOG(FATAL) << "Tensor type not supported";
  }
}

std::vector<xla::Shape> GetComponentShapes(const xla::Shape& shape) {
  std::vector<xla::Shape> component_shapes;
  if (xla::ShapeUtil::IsTuple(shape)) {
    for (const xla::Shape& component_shape : shape.tuple_shapes()) {
      XLA_CHECK(!xla::ShapeUtil::IsTuple(component_shape));
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

}  // namespace jit
}  // namespace torch
