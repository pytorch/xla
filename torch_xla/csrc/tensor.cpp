#include "tensor.h"

#include <list>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "c10/util/Exception.h"
#include "helpers.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "torch/csrc/autograd/variable.h"
#include "translator.h"

namespace torch {
namespace jit {

namespace {

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

template <class NativeT>
std::vector<NativeT> LinearizeTensor(const at::Tensor& t,
                                     const size_t total_elements);

template <>
std::vector<float> LinearizeTensor<float>(const at::Tensor& t,
                                          const size_t total_elements) {
  const at::Tensor& cont_t = t.contiguous();
  return std::vector<float>(cont_t.data<float>(),
                            cont_t.data<float>() + total_elements);
}

template <>
std::vector<xla::int64> LinearizeTensor<xla::int64>(
    const at::Tensor& t, const size_t total_elements) {
  const at::Tensor& cont_t = t.contiguous();
  return std::vector<xla::int64>(cont_t.data<int64_t>(),
                                 cont_t.data<int64_t>() + total_elements);
}

template <class NativeT>
std::shared_ptr<xla::ComputationClient::Data> TensorToXlaImpl(
    const at::Tensor& param_tensor, const xla::Shape& param_shape,
    const XLATensor::Device& device, xla::ComputationClient* client) {
  size_t total_elements = 1;
  std::vector<xla::int64> dimension_sizes;
  for (const auto dimension_size : param_tensor.sizes()) {
    dimension_sizes.push_back(dimension_size);
    total_elements *= dimension_size;
  }
  xla::Array<NativeT> parameter_xla_array(dimension_sizes);
  parameter_xla_array.SetValues(
      LinearizeTensor<NativeT>(param_tensor, total_elements));
  xla::Literal literal(param_shape);
  literal.PopulateFromArray(parameter_xla_array);
  return client->TransferParameterToServer(literal,
                                           /*device=*/device.ToString());
}

std::shared_ptr<xla::ComputationClient::Data> TensorToXla(
    const at::Tensor& param_tensor, const xla::Shape& param_shape,
    const XLATensor::Device& device, xla::ComputationClient* client) {
  switch (param_tensor.type().scalarType()) {
    case at::ScalarType::Float:
      return TensorToXlaImpl<float>(param_tensor, param_shape, device, client);
    case at::ScalarType::Long:
      return TensorToXlaImpl<xla::int64>(param_tensor, param_shape, device,
                                         client);
    default:
      LOG(FATAL) << "Tensor type not supported";
  }
}

at::Tensor MakeTensorFromXlaLiteral(const xla::Literal& literal) {
  const auto& result_shape = literal.shape();
  std::vector<int64_t> dimensions;
  for (const auto result_dimension : result_shape.dimensions()) {
    dimensions.push_back(result_dimension);
  }
  auto literal_type = result_shape.element_type();
  const auto torch_layout =
      MakeTorchTensorLayout(XlaHelpers::I64List(dimensions), literal_type);
  const auto literal_with_torch_layout = literal.Relayout(torch_layout);
  switch (literal_type) {
    case xla::PrimitiveType::F32: {
      const auto result_slice = literal_with_torch_layout.data<float>();
      at::Tensor result_tensor =
          at::empty(dimensions, at::TensorOptions(at::kFloat));
      std::copy(result_slice.begin(), result_slice.end(),
                result_tensor.data<float>());
      return result_tensor;
    }
    case xla::PrimitiveType::S64: {
      const auto result_slice = literal_with_torch_layout.data<xla::int64>();
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
  CHECK_EQ(index_mapping.size(), new_dest_elements.size());
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

XLATensor::XLATensor(const autograd::Variable& tensor, const Device& device)
    : data_(std::make_shared<Data>(
          TensorToXla(
              tensor,
              MakeArrayShapeFromDimensions(
                  tensor.sizes(),
                  XlaHelpers::MakeXlaPrimitiveType(tensor.type().scalarType()),
                  device.hw_type),
              device, XlaGetClient()),
          device, 0)),
      requires_grad_(tensor.requires_grad()) {}

XLATensor::XLATensor(std::shared_ptr<xla::ComputationClient::Data> xla_data,
                     uint64_t module_id)
    : data_(std::make_shared<Data>(
          xla_data, DeviceFromString(xla_data->device()), module_id)) {}

XLATensor::XLATensor(std::shared_ptr<XlaGraphNode> xla_graph_node,
                     const Device& device, uint64_t module_id)
    : data_(std::make_shared<Data>(std::move(xla_graph_node), device,
                                   module_id)) {
  TryLimitGraphSize();
}

void XLATensor::MulAddMulti(
    const double scale_dest,
    const std::vector<std::shared_ptr<XLATensor>>& dest_tuple,
    const double alpha,
    const std::vector<std::shared_ptr<XLATensor>>& source_tuple) {
  CHECK_EQ(dest_tuple.size(), source_tuple.size());
  XlaGraphContext xla_graph_ctx;
  for (size_t i = 0; i < dest_tuple.size(); ++i) {
    auto dest_node = dest_tuple[i]->GetXlaGraphNode();
    auto source_node = source_tuple[i]->GetXlaGraphNode();
    auto dest_node_op = dest_node->Generate(&xla_graph_ctx).ValueOrDie();
    auto source_node_op = source_node->Generate(&xla_graph_ctx).ValueOrDie();
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
  XlaGraphContext xla_graph_ctx;
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

const xla::Shape& XLATensor::shape() const {
  return data_->xla_data ? data_->xla_data->shape()
                         : data_->xla_graph_node->shape();
}

const XLATensor::Device& XLATensor::GetDevice() const { return data_->device; }

const std::shared_ptr<xla::ComputationClient::Data>& XLATensor::GetXlaData() {
  ApplyPendingGraph();
  return data_->xla_data;
}

void XLATensor::SetXlaData(
    std::shared_ptr<xla::ComputationClient::Data> xla_data) {
  data_->xla_data = std::move(xla_data);
  data_->xla_graph_node = nullptr;
  // A modified tensor doesn't come directly from a module forward call.
  data_->module_id = 0;
}

void XLATensor::SetXlaGraphNode(std::shared_ptr<XlaGraphNode> xla_graph_node) {
  data_->xla_graph_node = std::move(xla_graph_node);
  // A modified tensor doesn't come directly from a module forward call.
  data_->module_id = 0;
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

uint64_t XLATensor::ForwardModuleId() const { return data_->module_id; }

at::Tensor XLATensor::toTensor() {
  ApplyPendingGraph();
  // Because there's no transferToClient, we'll define an `identity` graph, and
  // execute it.
  xla::XlaBuilder b("identity");
  xla::GetTupleElement(xla::Tuple(&b, {xla::Parameter(&b, 0, shape(), "x")}),
                       0);
  xla::XlaComputation identity = b.Build().ValueOrDie();

  auto client = XlaGetClient();
  auto result_literal = client->ExecuteComputationAndTransfer(
      identity, {data_->xla_data.get()}, nullptr);
  auto return_tensor = MakeTensorFromXlaLiteral(*result_literal);
  return autograd::make_variable(return_tensor, requires_grad_);
}

std::shared_ptr<XlaGraphNode> XLATensor::CreateTensorNode(
    std::shared_ptr<xla::ComputationClient::Data> data) {
  auto generator = [data](XlaGraphContext* ctx,
                          const XlaGraphNode&) -> xla::StatusOr<xla::XlaOp> {
    return ctx->GetParameter(data);
  };
  return XlaGraphNode::New(std::move(generator), data->shape(), {});
}

std::shared_ptr<XlaGraphNode> XLATensor::CreateMulNode(XLATensor& other) {
  auto generator = [](XlaGraphContext* ctx,
                      const XlaGraphNode& node) -> xla::StatusOr<xla::XlaOp> {
    TF_ASSIGN_OR_RETURN(auto node_op, node.input(0)->Generate(ctx));
    TF_ASSIGN_OR_RETURN(auto other_node_op, node.input(1)->Generate(ctx));
    return node_op * other_node_op;
  };
  return XlaGraphNode::New(std::move(generator), shape(),
                           {GetXlaGraphNode(), other.GetXlaGraphNode()});
}

std::shared_ptr<XlaGraphNode> XLATensor::CreateMulNode(
    const at::Scalar& other) {
  auto generator = [other](
                       XlaGraphContext* ctx,
                       const XlaGraphNode& node) -> xla::StatusOr<xla::XlaOp> {
    TF_ASSIGN_OR_RETURN(auto node_op, node.input(0)->Generate(ctx));
    return node_op * XlaHelpers::ScalarBroadcast<float>(other.toDouble(),
                                                        node.input(0)->shape(),
                                                        ctx->builder());
  };
  return XlaGraphNode::New(std::move(generator), shape(), {GetXlaGraphNode()});
}

std::shared_ptr<XlaGraphNode> XLATensor::CreateDivNode(XLATensor& other) {
  auto generator = [](XlaGraphContext* ctx,
                      const XlaGraphNode& node) -> xla::StatusOr<xla::XlaOp> {
    TF_ASSIGN_OR_RETURN(auto node_op, node.input(0)->Generate(ctx));
    TF_ASSIGN_OR_RETURN(auto other_node_op, node.input(1)->Generate(ctx));
    return node_op / other_node_op;
  };
  return XlaGraphNode::New(std::move(generator), shape(),
                           {GetXlaGraphNode(), other.GetXlaGraphNode()});
}

std::shared_ptr<XlaGraphNode> XLATensor::CreateDivNode(
    const at::Scalar& other) {
  auto generator = [other](
                       XlaGraphContext* ctx,
                       const XlaGraphNode& node) -> xla::StatusOr<xla::XlaOp> {
    TF_ASSIGN_OR_RETURN(auto node_op, node.input(0)->Generate(ctx));
    return node_op / XlaHelpers::ScalarBroadcast<float>(other.toDouble(),
                                                        node.input(0)->shape(),
                                                        ctx->builder());
  };
  return XlaGraphNode::New(std::move(generator), shape(), {GetXlaGraphNode()});
}

std::shared_ptr<XlaGraphNode> XLATensor::CreateAddNode(
    XLATensor& other, const at::Scalar& alpha) {
  auto generator = [alpha](
                       XlaGraphContext* ctx,
                       const XlaGraphNode& node) -> xla::StatusOr<xla::XlaOp> {
    TF_ASSIGN_OR_RETURN(auto node_op, node.input(0)->Generate(ctx));
    TF_ASSIGN_OR_RETURN(auto other_node_op, node.input(1)->Generate(ctx));
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
  return std::make_shared<XLATensor>(CreateAddNode(other, alpha), data_->device,
                                     0);
}

void XLATensor::add_(XLATensor& other, const at::Scalar& alpha) {
  SetXlaGraphNode(CreateAddNode(other, alpha));
}

std::shared_ptr<XLATensor> XLATensor::mul(XLATensor& other) {
  return std::make_shared<XLATensor>(CreateMulNode(other), data_->device, 0);
}

std::shared_ptr<XLATensor> XLATensor::mul(const at::Scalar& other) {
  return std::make_shared<XLATensor>(CreateMulNode(other), data_->device, 0);
}

void XLATensor::mul_(XLATensor& other) {
  SetXlaGraphNode(CreateMulNode(other));
}

void XLATensor::mul_(const at::Scalar& other) {
  SetXlaGraphNode(CreateMulNode(other));
}

std::shared_ptr<XLATensor> XLATensor::div(XLATensor& other) {
  return std::make_shared<XLATensor>(CreateDivNode(other), data_->device, 0);
}

std::shared_ptr<XLATensor> XLATensor::div(const at::Scalar& other) {
  return std::make_shared<XLATensor>(CreateDivNode(other), data_->device, 0);
}

void XLATensor::div_(XLATensor& other) {
  SetXlaGraphNode(CreateDivNode(other));
}

void XLATensor::div_(const at::Scalar& other) {
  SetXlaGraphNode(CreateDivNode(other));
}

void XLATensor::zero_() {
  xla::Shape tensor_shape = shape();
  auto generator = [tensor_shape](
                       XlaGraphContext* ctx,
                       const XlaGraphNode&) -> xla::StatusOr<xla::XlaOp> {
    auto zero_literal = xla::LiteralUtil::Zero(tensor_shape.element_type());
    auto const_zero = xla::ConstantLiteral(ctx->builder(), zero_literal);
    return xla::Broadcast(const_zero, XlaHelpers::ShapeSizes(tensor_shape));
  };
  SetXlaGraphNode(XlaGraphNode::New(std::move(generator), tensor_shape, {}));
}

std::shared_ptr<XLATensor> XLATensor::cross_replica_sum(
    const std::vector<std::vector<xla::int64>>& groups) {
  auto generator = [groups](
                       XlaGraphContext* ctx,
                       const XlaGraphNode& node) -> xla::StatusOr<xla::XlaOp> {
    std::vector<xla::ReplicaGroup> crs_groups;
    for (auto& group : groups) {
      xla::ReplicaGroup rgroup;
      for (auto replica_id : group) {
        rgroup.add_replica_ids(replica_id);
      }
      crs_groups.push_back(std::move(rgroup));
    }
    TF_ASSIGN_OR_RETURN(auto node_op, node.input(0)->Generate(ctx));
    return xla::CrossReplicaSum(node_op, crs_groups);
  };
  auto crs_node =
      XlaGraphNode::New(std::move(generator), shape(), {GetXlaGraphNode()});
  return std::make_shared<XLATensor>(std::move(crs_node), data_->device, 0);
}

void XLATensor::ApplyPendingGraph() {
  auto& xla_graph_node = current_xla_graph_node();
  if (xla_graph_node != nullptr) {
    XlaGraphContext xla_graph_ctx;
    auto root = xla_graph_node->Generate(&xla_graph_ctx).ValueOrDie();
    auto computation = xla_graph_ctx.Build(root).ValueOrDie();
    SetXlaData(XlaGetClient()->ExecuteComputation(
        computation, xla_graph_ctx.GetParametersData(), nullptr));
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
      computation, xla_graph_ctx->GetParametersData(), &multi_shape);
  auto new_dest_elements = client->DeconstructTuple(*result_tuple).ValueOrDie();
  // Replace destination's underlying data with the result of the computation.
  SetMulti(tensors, new_dest_elements, index_mapping);
}

void XLATensor::ApplyPendingGraph(
    const std::vector<std::shared_ptr<XLATensor>>& tensors) {
  struct DeviceContext {
    XlaGraphContext xla_graph_ctx;
    std::vector<xla::int64> index_mapping;
  };
  std::map<Device, DeviceContext> contexts_map;
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto& xla_graph_node = tensors[i]->current_xla_graph_node();
    if (xla_graph_node != nullptr) {
      DeviceContext* device_context = &contexts_map[tensors[i]->GetDevice()];
      auto root =
          xla_graph_node->Generate(&device_context->xla_graph_ctx).ValueOrDie();
      device_context->xla_graph_ctx.AddResult(root);
      device_context->index_mapping.push_back(i);
    }
  }
  if (!contexts_map.empty()) {
    std::vector<xla::XlaComputation> computations;
    std::vector<std::vector<xla::ComputationClient::Data*>> parameters;
    std::list<xla::Shape> shapes;
    std::vector<const xla::Shape*> output_shapes;
    for (auto& device_context : contexts_map) {
      computations.push_back(
          device_context.second.xla_graph_ctx.Build().ValueOrDie());
      auto program_shape = computations.back().GetProgramShape().ValueOrDie();
      shapes.push_back(MakeShapeWithDeviceLayout(program_shape.result(),
                                                 device_context.first.hw_type));
      output_shapes.push_back(&shapes.back());
      parameters.push_back(
          device_context.second.xla_graph_ctx.GetParametersData());
    }
    auto client = XlaGetClient();
    auto result_tuples =
        client->ExecuteParallel(computations, parameters, output_shapes);
    auto context_iterator = contexts_map.begin();
    for (size_t i = 0; i < computations.size(); ++i, ++context_iterator) {
      auto new_dest_elements =
          client->DeconstructTuple(*result_tuples[i]).ValueOrDie();
      // Replace destination's underlying data with the result of the
      // computation.
      SetMulti(tensors, new_dest_elements,
               context_iterator->second.index_mapping);
    }
  }
}

XLATensor::Device XLATensor::DeviceFromString(const std::string& device_spec) {
  if (device_spec.empty()) {
    const std::string default_device_spec = XlaGetClient()->GetDefaultDevice();
    CHECK(!default_device_spec.empty());
    return DeviceFromString(default_device_spec);
  }
  if (device_spec[0] == ':') {
    const std::string default_device_spec = XlaGetClient()->GetDefaultDevice();
    auto pos = default_device_spec.find(':');
    CHECK_NE(pos, std::string::npos) << default_device_spec;
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
  CHECK(!tensors.empty());
  const XLATensor::Device& device = tensors.front()->GetDevice();
  for (const auto& tensor : tensors) {
    const XLATensor::Device& tensor_device = tensor->GetDevice();
    if (tensor_device != device) {
      AT_ERROR("All input tensors should have the same device");
    }
  }
  return device;
}

std::vector<xla::Shape> GetComponentShapes(const xla::Shape& shape) {
  std::vector<xla::Shape> component_shapes;
  if (xla::ShapeUtil::IsTuple(shape)) {
    for (const xla::Shape& component_shape : shape.tuple_shapes()) {
      CHECK(!xla::ShapeUtil::IsTuple(component_shape));
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
  CHECK(!shape_components.empty());
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
