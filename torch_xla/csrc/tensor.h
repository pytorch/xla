#pragma once

#include "graph_context.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/ir.h"

namespace torch {
namespace jit {

class XLATensor {
  struct Data;

 public:
  TH_DISALLOW_COPY_AND_ASSIGN(XLATensor);

  enum class DeviceType { CPU, GPU, TPU };

  struct Device {
    Device() = default;
    Device(DeviceType hw_type, int ordinal)
        : hw_type(hw_type), ordinal(ordinal) {}

    bool operator==(const Device& other) const {
      return hw_type == other.hw_type && ordinal == other.ordinal;
    }

    bool operator!=(const Device& other) const { return !(*this == other); }

    bool operator<(const Device& rhs) const {
      if (hw_type != rhs.hw_type) {
        return hw_type < rhs.hw_type;
      }
      return ordinal < rhs.ordinal;
    }

    std::string ToString() const;

    DeviceType hw_type = DeviceType::CPU;
    int ordinal = 0;
  };

  XLATensor(const autograd::Variable& tensor, const Device& device);
  XLATensor(std::shared_ptr<xla::ComputationClient::Data> xla_data,
            uint64_t module_id, bool requires_grad);
  XLATensor(std::shared_ptr<XlaGraphNode> xla_graph_node, const Device& device,
            uint64_t module_id);
  XLATensor(std::shared_ptr<Data> data) : data_(std::move(data)) {}

  // Creates a new XLA tensor sharing the core tensor data structure, with
  // require-gradients disabled.
  std::shared_ptr<XLATensor> Clone() const {
    return std::make_shared<XLATensor>(data_);
  }

  bool RequiresGrad() const { return requires_grad_; }

  void detach_() { requires_grad_ = false; }

  at::Tensor toTensor();

  std::shared_ptr<XLATensor> grad() const;
  void setGrad(std::shared_ptr<XLATensor> grad);

  const xla::Shape& shape() const;
  const Device& GetDevice() const;
  const std::shared_ptr<xla::ComputationClient::Data>& GetXlaData();
  void SetXlaData(std::shared_ptr<xla::ComputationClient::Data> xla_data);
  std::shared_ptr<XlaGraphNode> GetXlaGraphNode() const;
  std::vector<int64_t> Size() const;
  uint64_t ForwardModuleId() const;

  // Basic tensor operations used by the optimizers.
  std::shared_ptr<XLATensor> add(XLATensor& other, const at::Scalar& alpha);
  void add_(XLATensor& other, const at::Scalar& alpha);

  std::shared_ptr<XLATensor> mul(XLATensor& other);
  std::shared_ptr<XLATensor> mul(const at::Scalar& other);
  void mul_(XLATensor& other);
  void mul_(const at::Scalar& other);

  std::shared_ptr<XLATensor> div(XLATensor& other);
  std::shared_ptr<XLATensor> div(const at::Scalar& other);
  void div_(XLATensor& other);
  void div_(const at::Scalar& other);

  void zero_();

  std::shared_ptr<XLATensor> cross_replica_sum(
      const std::vector<std::vector<xla::int64>>& groups);

  // Applies the queue of operations in preparation for using the data.
  void ApplyPendingGraph();

  // Converts the given "device_spec" string to a device. The format is
  // <hw_type>:<ordinal>, where hw_type is one of TPU, CPU or GPU and ordinal is
  // an integer.
  static Device DeviceFromString(const std::string& device_spec);

  // Returns the common device for "tensors". Throws if not all tensors have the
  // same device.
  static Device CommonDeviceForTensors(
      const std::vector<std::shared_ptr<XLATensor>>& tensors);

  // In place scale and add for multiple tensors. The operation applies to all
  // tensors "dest" in "dest_tuple" and is:
  //   dest = scale_dest * dest + alpha * source
  // where "source" is the corresponding tensor in "source_tuple".
  // This is a (temporary) building block for manually batched SGD optimizer. We
  // have ways to automatically batch the optimizer application to all weights
  // in the model; for expediency, we'll instead do this to minimize the number
  // of moving parts needed to achieve better usability.
  static void MulAddMulti(
      const double scale_dest,
      const std::vector<std::shared_ptr<XLATensor>>& dest_tuple,
      const double alpha,
      const std::vector<std::shared_ptr<XLATensor>>& source_tuple);

  // Zero all the tensors in "dest_tuple", it exists for the same reason as
  // "MulAddMulti".
  static void ZeroMulti(
      const std::vector<std::shared_ptr<XLATensor>>& dest_tuple);

  // Applies the queue of operations for a list of tensors.
  static void ApplyPendingGraph(
      const std::vector<std::shared_ptr<XLATensor>>& tensors);

  // Retrieves the PyTorch tensors behind the XLA tensors.
  static std::vector<at::Tensor> GetTensors(
      const std::vector<std::shared_ptr<XLATensor>>& tensors);

  // Operation which creates XLA tensors out of autograd variable by batching
  // the requests to the computation servers.
  static std::vector<std::shared_ptr<XLATensor>> CreateTensors(
      const std::vector<autograd::Variable>& tensors,
      const std::vector<std::string>& devices);

 private:
  struct Data {
    Data(std::shared_ptr<xla::ComputationClient::Data> xla_data,
         const Device& device, uint64_t module_id)
        : xla_data(std::move(xla_data)), device(device), module_id(module_id) {}
    Data(std::shared_ptr<XlaGraphNode> xla_graph_node, const Device& device,
         uint64_t module_id)
        : xla_graph_node(std::move(xla_graph_node)),
          device(device),
          module_id(module_id) {}

    std::shared_ptr<xla::ComputationClient::Data> xla_data;
    std::shared_ptr<XLATensor> grad;
    std::shared_ptr<XlaGraphNode> xla_graph_node;
    Device device;
    uint64_t module_id = 0;
  };

  void SetXlaGraphNode(std::shared_ptr<XlaGraphNode> xla_graph_node);

  const std::shared_ptr<XlaGraphNode>& current_xla_graph_node() const {
    return data_->xla_graph_node;
  }

  // We build an XLA graph accumulating XLA operations, but at a given point we
  // need to force a rendering, otherwise the graph can grow without control.
  // Think:
  //   for i in range(0, 100000):
  //     a = a + b
  void TryLimitGraphSize();

  std::shared_ptr<XlaGraphNode> CreateAddNode(XLATensor& other,
                                              const at::Scalar& alpha);
  std::shared_ptr<XlaGraphNode> CreateMulNode(XLATensor& other);
  std::shared_ptr<XlaGraphNode> CreateMulNode(const at::Scalar& other);
  std::shared_ptr<XlaGraphNode> CreateDivNode(XLATensor& other);
  std::shared_ptr<XlaGraphNode> CreateDivNode(const at::Scalar& other);

  static void ComputeAndDistribute(
      XlaGraphContext* xla_graph_ctx,
      const std::vector<xla::int64>& index_mapping,
      const std::vector<std::shared_ptr<XLATensor>>& tensors);

  static std::shared_ptr<XlaGraphNode> CreateTensorNode(
      std::shared_ptr<xla::ComputationClient::Data> data);

  std::shared_ptr<Data> data_;
  bool requires_grad_ = false;
};

// If "shape" is a tuple, return the element shapes, otherwise return a
// singleton list containing the original shape.
std::vector<xla::Shape> GetComponentShapes(const xla::Shape& shape);

// Create a shape with "device_type" compatible layout from the given "shape".
xla::Shape MakeShapeWithDeviceLayout(const xla::Shape& shape,
                                     const XLATensor::DeviceType device_type);

}  // namespace jit
}  // namespace torch
