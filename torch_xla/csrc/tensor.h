#pragma once

#include <iostream>
#include <string>
#include <unordered_map>

#include "device.h"
#include "ir.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/ir.h"

namespace torch_xla {

class XLATensor {
  struct Data;

 public:
  TH_DISALLOW_COPY_AND_ASSIGN(XLATensor);

  // The context used by the ApplyPendingGraph() API, in order to allow it speed
  // up operations in case the new tensors graph apply matches the one stored
  // within the apply context.
  struct ApplyContext {
    std::vector<std::shared_ptr<xla::ComputationClient::Computation>>
        computations;
    std::vector<xla::int64> uid_order;
    std::vector<std::vector<xla::int64>> input_mapping;
    std::vector<std::vector<xla::int64>> index_mapping;
    std::vector<std::string> devices;
  };

  static std::shared_ptr<XLATensor> Create(const at::Tensor& tensor,
                                           const Device& device,
                                           bool requires_grad);
  static std::shared_ptr<XLATensor> Create(
      std::shared_ptr<xla::ComputationClient::Data> xla_data,
      bool requires_grad);
  static std::shared_ptr<XLATensor> Create(ir::NodePtr ir_node,
                                           const Device& device);
  static std::shared_ptr<XLATensor> Create(std::shared_ptr<Data> data);

  // NOTE: These direct constructors should not be used, and the Create() APIs
  // above should be used instead. These are not private because the hacks
  // necessary in order to use std::make_shared<> are worse than having those
  // public. And it is good to save the double allocation required by a normal
  // naked pointer std::shared_ptr<> creation.
  XLATensor(const at::Tensor& tensor, const Device& device, bool requires_grad);
  XLATensor(std::shared_ptr<xla::ComputationClient::Data> xla_data,
            bool requires_grad);
  XLATensor(ir::NodePtr ir_node, const Device& device);
  XLATensor(std::shared_ptr<Data> data) : data_(std::move(data)) {}

  ~XLATensor();

  // Creates a new XLA tensor sharing the core tensor data structure, with
  // require-gradients disabled.
  std::shared_ptr<XLATensor> Clone() const { return Create(data_); }

  bool RequiresGrad() const { return requires_grad_; }

  void detach_() { requires_grad_ = false; }

  at::Tensor ToTensor();

  // This API should be called instead of ToTensor() when the tensor is passed
  // to other ATEN APIs which will modify its value.
  at::Tensor ToMutableTensor();

  std::shared_ptr<XLATensor> grad() const;
  void SetGradient(std::shared_ptr<XLATensor> grad);

  at::ScalarType dtype() const;
  xla::util::MaybeRef<xla::Shape> shape() const;
  const Device& GetDevice() const;
  xla::int64 GetUniqueId() const;

  // Fetches the XLA data behind the tensor. If the tensor has a graph defining
  // its current value, executes the graph and fetches the XLA data result.
  std::shared_ptr<xla::ComputationClient::Data> GetXlaData();

  // Fetches the current value of the XLA data, which can be missing (nullptr)
  // in case the tensor has a graph defining its current value,
  std::shared_ptr<xla::ComputationClient::Data> CurrentXlaData() const;

  void SetXlaData(std::shared_ptr<xla::ComputationClient::Data> xla_data);

  // Retrieves the current IR Node, or nullptr in case no active IR Node is
  // available.
  const ir::NodePtr& CurrentIrNode() const;

  // Retrieves the IR Node representing this XLATensor. One will be created if
  // missing. Note that although this is a const API, it actually changes the
  // internal state ofthe object.
  ir::NodePtr GetIrNode() const;

  const c10::optional<at::Tensor>& CurrentTensorData() const;

  // Makes the data references from the current tensor, point to the ones from
  // the source tensor.
  void ReferenceDataFrom(const XLATensor& source);

  std::vector<int64_t> Size() const;

  // Basic tensor operations used by the optimizers.
  std::shared_ptr<XLATensor> add(const XLATensor& other,
                                 const at::Scalar& alpha);
  void add_(const XLATensor& other, const at::Scalar& alpha);

  std::shared_ptr<XLATensor> mul(const XLATensor& other);
  std::shared_ptr<XLATensor> mul(const at::Scalar& other);
  void mul_(const XLATensor& other);
  void mul_(const at::Scalar& other);

  std::shared_ptr<XLATensor> div(const XLATensor& other);
  std::shared_ptr<XLATensor> div(const at::Scalar& other);
  void div_(const XLATensor& other);
  void div_(const at::Scalar& other);

  void zero_();

  void addcdiv_(const at::Scalar& value, const XLATensor& tensor1,
                const XLATensor& tensor2);
  void addcmul_(const at::Scalar& value, const XLATensor& tensor1,
                const XLATensor& tensor2);

  // Additional operations which are part of the PyTorch Tensor functionality.
  xla::int64 size(int dim) const;

  std::shared_ptr<XLATensor> relu();

  std::shared_ptr<XLATensor> threshold(float threshold, float value);

  std::shared_ptr<XLATensor> conv2d(const std::shared_ptr<XLATensor>& weight,
                                    const std::shared_ptr<XLATensor>& bias,
                                    int stride, int padding,
                                    bool use_full_conv_precision);

  std::shared_ptr<XLATensor> addmm(const XLATensor& weight,
                                   const XLATensor& bias,
                                   bool use_full_conv_precision);

  std::shared_ptr<XLATensor> max_pool2d(int kernel_size, int stride,
                                        int padding);

  std::shared_ptr<XLATensor> avg_pool2d(int kernel_size, int stride,
                                        int padding, bool count_include_pad);

  std::shared_ptr<XLATensor> t();

  std::shared_ptr<XLATensor> view(
      tensorflow::gtl::ArraySlice<const xla::int64> output_size);

  std::shared_ptr<XLATensor> log_softmax(xla::int64 dim);

  std::shared_ptr<XLATensor> cross_replica_sum(
      const std::vector<std::vector<xla::int64>>& groups);

  // Applies the queue of operations in preparation for using the data.
  void ApplyPendingGraph();

  // Dumps the XLA HLO text of the computation accumulated in the graph node
  // which is attached to this tensor.
  std::string DumpGraphNodeComputation() const;

  // Returns the common device for "tensors". Throws if not all tensors have the
  // same device.
  static Device CommonDeviceForTensors(
      const std::vector<std::shared_ptr<XLATensor>>& tensors);

  // Retrieves the set of XLA tensors which are currently live in the system.
  static std::vector<std::shared_ptr<XLATensor>> GetLiveTensors();

  // Applies the queue of operations for a list of tensors. The context of the
  // apply operation will be saved within the apply_context pointer, if not
  // nullptr. The ApplyPendingGraph() API will try to guess whether the current
  // apply operation matches the previously cached one in apply_context, and
  // eventually uses the cached XLA compiled computations to run the apply.
  static void ApplyPendingGraph(
      const std::vector<std::shared_ptr<XLATensor>>& tensors,
      ApplyContext* apply_context);

  // Retrieves the PyTorch tensors behind the XLA tensors.
  static std::vector<at::Tensor> GetTensors(
      const std::vector<std::shared_ptr<XLATensor>>& tensors);

  // Operation which creates XLA tensors out of autograd variable by batching
  // the requests to the computation servers.
  static std::vector<std::shared_ptr<XLATensor>> CreateTensors(
      const std::vector<at::Tensor>& tensors,
      const std::vector<std::string>& devices);

 private:
  // Maps from ComputationClient Data unique ID to XLA tensor unique ID.
  using DataUidMap = std::unordered_map<xla::int64, xla::int64>;

  struct Data {
    Data(std::shared_ptr<xla::ComputationClient::Data> xla_data,
         const Device& device)
        : xla_data(std::move(xla_data)),
          device(device),
          unique_id(GetNextTensorId()) {}
    Data(ir::NodePtr ir_node, const Device& device)
        : ir_node(std::move(ir_node)),
          device(device),
          unique_id(GetNextTensorId()) {}
    Data(at::Tensor tensor_data, const Device& device)
        : tensor_data(std::move(tensor_data)),
          device(device),
          unique_id(GetNextTensorId()) {}

    std::shared_ptr<xla::ComputationClient::Data> xla_data;
    ir::NodePtr ir_node;
    c10::optional<at::Tensor> tensor_data;
    Device device;
    xla::int64 unique_id = 0;
    std::shared_ptr<XLATensor> grad;
  };

  void SetIrNode(ir::NodePtr ir_node);

  void SetTensorData(at::Tensor tensor_data);

  // We build an XLA graph accumulating XLA operations, but at a given point we
  // need to force a rendering, otherwise the graph can grow without control.
  // Think:
  //   for i in range(0, 100000):
  //     a = a + b
  void TryLimitGraphSize();

  // Create the mapping from computation client Data pointers to the XLA tensors
  // unique ID which are holding it.
  static DataUidMap CreateDataUidMap(
      const std::vector<std::shared_ptr<XLATensor>>& tensors);

  // Tries to run a cached ApplyPendingGraph() with the information in
  // apply_context. Returns whether the cached run could be completed
  // successfully.
  static bool RunCachedApply(
      const std::vector<std::shared_ptr<XLATensor>>& tensors,
      const ApplyContext& apply_context);

  // Returns a permutation which represents an ordering by tensor device and
  // unique ID, of all the tensors which needs sync (the ones which have a graph
  // backing their value). The tensors which are already sync, will not be
  // returned within the permutation. If a tensor has at::Tensor data only, the
  // at::Tensor data will be uploaded to the device and the tensor will receive
  // new XLA data.
  static std::vector<size_t> GetApplyOrder(
      const std::vector<std::shared_ptr<XLATensor>>& tensors);

  static ir::NodePtr CreateTensorNode(
      std::shared_ptr<xla::ComputationClient::Data> data);

  static xla::int64 GetNextTensorId();

  std::shared_ptr<Data> data_;
  bool requires_grad_ = false;
  bool has_aliases_ = false;
};

}  // namespace torch_xla
