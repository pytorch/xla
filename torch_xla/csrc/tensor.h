#pragma once

#include <iostream>
#include <string>
#include <unordered_map>

#include "device.h"
#include "ir.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch/csrc/autograd/variable.h"

namespace torch_xla {

class XLATensor {
  class TensorsArena;
  struct Data;
  struct View;

 public:
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

  static XLATensor Create(const at::Tensor& tensor, const Device& device,
                          bool requires_grad);
  static XLATensor Create(
      std::shared_ptr<xla::ComputationClient::Data> xla_data,
      bool requires_grad);

  // Creates an empty/null tensor.
  XLATensor() = default;

  bool RequiresGrad() const { return data()->requires_grad; }

  void detach_() { data()->requires_grad = false; }

  bool is_null() const { return data_ptr() == nullptr; }

  XLATensor alias() const { return XLATensor(data_ptr()); }

  at::Tensor ToTensor();

  // This API should be called instead of ToTensor() when the tensor is passed
  // to other ATEN APIs which will modify its value.
  at::Tensor ToMutableTensor();

  c10::optional<XLATensor> grad() const;
  void SetGradient(const XLATensor& grad);

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
  ir::NodeOperand CurrentIrNode() const;

  // Retrieves the IR Node representing this XLATensor. One will be created if
  // missing. Note that although this is a const API, it actually changes the
  // internal state ofthe object.
  ir::NodeOperand GetIrNode() const;

  const c10::optional<at::Tensor>& CurrentTensorData() const;

  // Makes the data references from the current tensor, point to the ones from
  // the source tensor.
  void ReferenceDataFrom(const XLATensor& source);

  std::vector<int64_t> DimensionSizes() const;

  // Basic tensor operations used by the optimizers.
  XLATensor add(const XLATensor& other, const at::Scalar& alpha) const;
  void add_(const XLATensor& other, const at::Scalar& alpha);

  XLATensor mul(const XLATensor& other) const;
  XLATensor mul(const at::Scalar& other) const;
  void mul_(const XLATensor& other);
  void mul_(const at::Scalar& other);

  XLATensor div(const XLATensor& other) const;
  XLATensor div(const at::Scalar& other) const;
  void div_(const XLATensor& other);
  void div_(const at::Scalar& other);

  void zero_();

  void addcdiv_(const at::Scalar& value, const XLATensor& tensor1,
                const XLATensor& tensor2);
  void addcmul_(const at::Scalar& value, const XLATensor& tensor1,
                const XLATensor& tensor2);

  // Additional operations which are part of the PyTorch Tensor functionality.
  xla::int64 size(int dim) const;

  XLATensor relu() const;

  XLATensor threshold(float threshold, float value) const;

  XLATensor conv2d(const XLATensor& weight, const XLATensor& bias,
                   tensorflow::gtl::ArraySlice<const xla::int64> stride,
                   tensorflow::gtl::ArraySlice<const xla::int64> padding,
                   bool use_full_conv_precision) const;

  XLATensor conv2d(const XLATensor& weight,
                   tensorflow::gtl::ArraySlice<const xla::int64> stride,
                   tensorflow::gtl::ArraySlice<const xla::int64> padding,
                   bool use_full_conv_precision) const;

  XLATensor addmm(const XLATensor& weight, const XLATensor& bias,
                  bool use_full_conv_precision) const;

  XLATensor max_pool2d(
      tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
      tensorflow::gtl::ArraySlice<const xla::int64> stride,
      tensorflow::gtl::ArraySlice<const xla::int64> padding) const;

  XLATensor avg_pool2d(
      tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
      tensorflow::gtl::ArraySlice<const xla::int64> stride,
      tensorflow::gtl::ArraySlice<const xla::int64> padding,
      bool count_include_pad) const;

  XLATensor t() const;

  XLATensor view(
      tensorflow::gtl::ArraySlice<const xla::int64> output_size) const;

  XLATensor log_softmax(xla::int64 dim) const;

  static XLATensor avg_pool2d_backward(
      const XLATensor& out_backprop, const XLATensor& input,
      tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
      tensorflow::gtl::ArraySlice<const xla::int64> stride,
      tensorflow::gtl::ArraySlice<const xla::int64> padding,
      bool count_include_pad);

  static std::tuple<XLATensor, XLATensor, XLATensor> conv2d_backward(
      const XLATensor& out_backprop, const XLATensor& input,
      const XLATensor& weight,
      tensorflow::gtl::ArraySlice<const xla::int64> stride,
      tensorflow::gtl::ArraySlice<const xla::int64> padding,
      bool use_full_conv_precision);

  XLATensor cross_replica_sum(
      const std::vector<std::vector<xla::int64>>& groups) const;

  // Applies the queue of operations in preparation for using the data.
  void ApplyPendingGraph();

  // Dumps the XLA HLO text of the computation accumulated in the graph node
  // which is attached to this tensor.
  std::string DumpGraphNodeComputation() const;

  // Returns the common device for "tensors". Throws if not all tensors have the
  // same device.
  static Device CommonDeviceForTensors(const std::vector<XLATensor>& tensors);

  // Retrieves the set of XLA tensors which are currently live in the system.
  static std::vector<XLATensor> GetLiveTensors();

  // Applies the queue of operations for a list of tensors. The context of the
  // apply operation will be saved within the apply_context pointer, if not
  // nullptr. The ApplyPendingGraph() API will try to guess whether the current
  // apply operation matches the previously cached one in apply_context, and
  // eventually uses the cached XLA compiled computations to run the apply.
  static void ApplyPendingGraph(std::vector<XLATensor>* tensors,
                                ApplyContext* apply_context);

  // Retrieves the PyTorch tensors behind the XLA tensors. If the writeable
  // vector is not nullptr, it must be the same size as tensors, and the
  // corresponding bool tells whether the ATEN tensor to be retrieved should the
  // a writeable copy.
  static std::vector<at::Tensor> GetTensors(std::vector<XLATensor>* tensors,
                                            const std::vector<bool>* writeable);

  // Operation which creates XLA tensors out of autograd variable by batching
  // the requests to the computation servers.
  static std::vector<XLATensor> CreateTensors(
      const std::vector<at::Tensor>& tensors,
      const std::vector<std::string>& devices);

 private:
  // Maps from ComputationClient Data unique ID to XLA tensor unique ID.
  using DataUidMap = std::unordered_map<xla::int64, xla::int64>;

  // When a "view" (capture by reference) is taken on a node, an Alias object is
  // created on the captured node itself, with its current IR Node value.
  // Inplace operations using the SetIrNode() API to update the current value,
  // will notice the presence of the alias, and also update the Alias ir_node.
  struct Alias {
    explicit Alias(ir::NodeOperand ir_node) : ir_node(std::move(ir_node)) {}

    ir::NodeOperand ir_node;
  };

  // A view represents a state of an XLA tensor in which its current value is a
  // view/reference of another tensor (IR Node). A View is fed by an Alias,
  // which captures the current value of the input tensor.
  struct View {
    View(xla::Shape shape, std::shared_ptr<Alias> alias)
        : shape(std::move(shape)), alias(std::move(alias)) {}

    xla::Shape shape;
    std::shared_ptr<Alias> alias;
    ir::NodeOperand ir_node;
  };

  struct ViewIrNode {
    ir::NodeOperand ir_node;
    bool updated;
  };

  // This is the core XLA tensor data structure where all the tensor data is
  // held. The XLA tensor is nothing more than a shared pointer to a Data
  // object.
  struct Data {
    Data(std::shared_ptr<xla::ComputationClient::Data> xla_data,
         const Device& device)
        : xla_data(std::move(xla_data)),
          device(device),
          unique_id(GetNextTensorId()) {}
    Data(ir::NodeOperand ir_node, const Device& device)
        : ir_node(std::move(ir_node)),
          device(device),
          unique_id(GetNextTensorId()) {}
    Data(std::shared_ptr<View> view, const Device& device)
        : view(std::move(view)), device(device), unique_id(GetNextTensorId()) {}
    Data(at::Tensor tensor_data, const Device& device)
        : tensor_data(std::move(tensor_data)),
          device(device),
          unique_id(GetNextTensorId()) {}

    ~Data();

    std::shared_ptr<xla::ComputationClient::Data> xla_data;
    ir::NodeOperand ir_node;
    std::shared_ptr<View> view;
    c10::optional<at::Tensor> tensor_data;
    Device device;
    xla::int64 unique_id = 0;
    std::shared_ptr<XLATensor> grad;
    bool requires_grad = false;
  };

  XLATensor(const at::Tensor& tensor, const Device& device, bool requires_grad);
  XLATensor(std::shared_ptr<xla::ComputationClient::Data> xla_data,
            bool requires_grad);
  XLATensor(ir::NodeOperand ir_node, const Device& device);
  XLATensor(std::shared_ptr<View> view, const Device& device);
  XLATensor(std::shared_ptr<Data> data);

  static XLATensor Create(ir::NodeOperand ir_node, const Device& device);
  static XLATensor Create(std::shared_ptr<View> view, const Device& device);

  Data* data() const;

  std::shared_ptr<Data> data_ptr() const { return data_; }

  void SetIrNode(ir::NodeOperand ir_node);

  void SetTensorData(at::Tensor tensor_data);

  // Discards all the XLA and IR data, by making the ATEN tensor one the only
  // source for this XLA tensor. An error is generated if the XLA tensor does
  // not have ATEN tensors data.
  void DiscardXlaData();

  // We build an XLA graph accumulating XLA operations, but at a given point we
  // need to force a rendering, otherwise the graph can grow without control.
  // Think:
  //   for i in range(0, 100000):
  //     a = a + b
  void TryLimitGraphSize();

  // Extracts the current IR Node out of a view, into a ViewIrNode structure
  // where the updated fields tells whether a new IR Node has been created, or
  // the cached one returned.
  static ViewIrNode GetViewIrNode(View* view);

  // Create the mapping from computation client Data pointers to the XLA tensors
  // unique ID which are holding it.
  static DataUidMap CreateDataUidMap(const std::vector<XLATensor>& tensors);

  // Tries to run a cached ApplyPendingGraph() with the information in
  // apply_context. Returns whether the cached run could be completed
  // successfully.
  static bool RunCachedApply(std::vector<XLATensor>* tensors,
                             const ApplyContext& apply_context);

  // Returns a permutation which represents an ordering by tensor device and
  // unique ID, of all the tensors which needs sync (the ones which have a graph
  // backing their value). The tensors which are already sync, will not be
  // returned within the permutation. If a tensor has at::Tensor data only, the
  // at::Tensor data will be uploaded to the device and the tensor will receive
  // new XLA data.
  static std::vector<size_t> GetApplyOrder(
      const std::vector<XLATensor>& tensors);

  static ir::NodeOperand CreateTensorNode(
      std::shared_ptr<xla::ComputationClient::Data> data);

  static xla::int64 GetNextTensorId();

  std::shared_ptr<Data> data_;
};

}  // namespace torch_xla
