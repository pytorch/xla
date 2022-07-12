#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/async_task.h"
#include "tensorflow/compiler/xla/xla_client/cache.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/lazy/core/ir_util.h"
#include "torch_xla/csrc/computation.h"
#include "torch_xla/csrc/cross_replica_reduces.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ir_util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/dynamic_ir.h"
#include "torch_xla/csrc/view.h"
#include "torch_xla/csrc/xla_sharding_util.h"

namespace torch_xla {

class XLATensor;
using XLATensorPtr = c10::intrusive_ptr<XLATensor>;

class XLATensor : public c10::intrusive_ptr_target {
  class DeviceContextArena;
  struct Data;

 public:
  static XLATensorPtr Create(const at::Tensor& tensor,
                             const torch::lazy::BackendDevice& device);
  static XLATensorPtr Create(
      torch::lazy::BackendDataPtr xla_data,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static XLATensorPtr Create(
      torch::lazy::Value ir_value, const torch::lazy::BackendDevice& device,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  // The default ctor previously created a null LazyTensor (one with no 'data'
  // obj). Creating a null XLATensor is no longer possible, since the same can
  // be achieved by creating a null LazyTensorPtr and it is way too confusing to
  // have to check both lazy_tensor_ptr && *lazy_tensor_ptr, so everywhere that
  // used to rely on a LazyTensor obj with a null Data can now rely on a null
  // LazyTensorPtr instead.
  XLATensor() = delete;

  size_t generation() const { return data()->generation; }

  XLATensorPtr alias() const {
    return c10::make_intrusive<XLATensor>(XLATensor(data_ptr()));
  }

  int64_t size(int64_t dim) const;

  at::Tensor ToTensor(bool detached);

  void ShallowCopyTo(XLATensorPtr dest) const;

  // Assigns the tensor value to the XLA tensor.
  void SetTensor(at::Tensor tensor);

  void UpdateFromTensor(at::Tensor tensor, bool sync);
  void UpdateFromTensorOut(at::Tensor tensor);
  void UpdateFromTensorOut(const XLATensorPtr& tensor);

  at::ScalarType dtype() const;
  c10::optional<at::ScalarType> dtype_optional() const;

  // Set logical_element_type which is visible to upstream PyTorch.
  void SetScalarType(c10::optional<at::ScalarType> logical_element_type);

  xla::util::MaybeRef<xla::Shape> shape() const;
  xla::Shape shape_with_layout() const;

  const torch::lazy::BackendDevice& GetDevice() const;
  int64_t GetUniqueId() const;

  // Retrieves an opaque ID of the alias object upon which the tensor's view is
  // rooted, or 0 if this tensor is not a view.
  std::ptrdiff_t GetViewAliasId() const;

  // Fetches the XLA data behind the tensor. If the tensor has a graph defining
  // its current value, executes the graph and fetches the XLA data result.
  torch::lazy::BackendDataPtr GetXlaData();

  // Fetches the current value of the XLA data, which can be missing (nullptr)
  // in case the tensor has a graph defining its current value,
  torch::lazy::BackendDataPtr CurrentXlaData() const;

  void SetXlaData(torch::lazy::BackendDataPtr xla_data);

  // Retrieves the current IR XlaNode, or nullptr in case no active IR XlaNode
  // is available.
  torch::lazy::Value CurrentIrValue() const;

  // Retrieves the IR Node representing this XLATensor. One will be created if
  // missing. Note that although this is a const API, it actually changes the
  // internal state ofthe object.
  torch::lazy::Value GetIrValue() const;

  c10::optional<at::Tensor> CurrentTensorData() const;

  // Applies the queue of operations in preparation for using the data.
  void ApplyPendingGraph();

  // This method just syncs the tensors passed as argument. This method is
  // called at two places:
  // 1. Creating tensor from IR value. This is where an output tensor is created
  // from an IR computation
  // 2. SetIRValue(). This is where the IR value of in place operations are
  // updated. Note: We do not sync the output of ViewTensors. This is because:
  // 1. The operations that generate the ViewTensor would be re-done when its
  // base tensor is updated. When the base tensor is updated, torch-xla would
  // apply all the views on it and hence the operations would be repeated.
  // Hence, we don't sync the ViewTensors and in case users want to print them,
  // they can still do it and will incur a small graph compile. This way we
  // avoid some extra compiles. This makes it lazy just for view operations.
  // Note: ViewTensors do not share the same storage as the input tensor. This
  // is by design. Currently, to respect the definitions of view tensors,
  // different view relationships between tensors is tracked and update all the
  // tensors to make it look as if they share same storage. Hence, the
  // operations on view tensor would be repeated when we try to sync the tensor
  // that is affected by the view tensor.
  static void ApplyEagerSync(std::vector<XLATensorPtr>& tensors);

  static torch::lazy::Value GetDeviceDataIrValue(
      const at::Scalar& value, xla::PrimitiveType type,
      const torch::lazy::BackendDevice& device);
  // Use with caution, constant will cause more frequent recompilation
  // compared to the device_data.
  static torch::lazy::Value GetIrValueForConstant(const at::Scalar& value,
                                                  const xla::Shape& shape);
  static torch::lazy::Value GetIrValueForScalar(
      const at::Scalar& value, xla::PrimitiveType type,
      const torch::lazy::BackendDevice& device);
  static torch::lazy::Value GetIrValueForScalar(
      const at::Scalar& value, const torch::lazy::BackendDevice& device);
  static torch::lazy::Value GetIrValueForScalar(
      const at::Scalar& value, xla::PrimitiveType type,
      absl::Span<const int64_t> dimensions,
      const torch::lazy::BackendDevice& device);
  static torch::lazy::Value GetIrValueForScalar(
      const at::Scalar& value, const xla::Shape& shape,
      const torch::lazy::BackendDevice& device);
  static torch::lazy::Value GetIrValueForScalar(
      const at::Scalar& value, const xla::Shape& shape,
      c10::optional<at::ScalarType> logical_element_type,
      const torch::lazy::BackendDevice& device);

  static torch::lazy::Value GetRngSeed(
      const torch::lazy::BackendDevice& device);

  static void SetRngSeed(const torch::lazy::BackendDevice& device,
                         uint64_t seed);

  static uint64_t GetRunningSeed(const torch::lazy::BackendDevice& device);

  // Dispatches a comparison operator, setting the logical type of the result
  // appropriately.
  static XLATensorPtr DispatchComparisonOp(c10::Symbol kind,
                                           const XLATensorPtr& input,
                                           const at::Scalar& other);

  // Same as above, with the second input a tensor as well.
  static XLATensorPtr DispatchComparisonOp(c10::Symbol kind,
                                           const XLATensorPtr& input,
                                           const XLATensorPtr& other);

  // Dumps the XLA HLO text of the computation accumulated in the graph which is
  // attached the tensors.
  static std::string DumpHloComputation(
      const std::vector<XLATensorPtr>& tensors);

  // Retrieves the set of XLA tensors which are currently live in the system,
  // for the given device. If device is nullptr, the live tensors for all
  // devices will be returned. Returned tensors are sorted by device as primary
  // key, and by unique ID as secondary key.
  static std::vector<XLATensorPtr> GetLiveTensors(
      const torch::lazy::BackendDevice* device);

  // Applies all the pending IR operations queued over the input tensors. All
  // the tensors must be on the same device. If wait is true, the sync operation
  // will be run synchronously. The devices argument, if not empty, tells the
  // devices which should be participating into the replicated computation.
  static void SyncTensorsGraph(std::vector<XLATensorPtr>* tensors,
                               absl::Span<const std::string> devices, bool wait,
                               bool sync_xla_data);

  // Makes sure that any outstanding IR operation accumulated over live tensors,
  // gets turned into device data. If wait is true, the sync operation will be
  // run synchronously. The devices argument, if not empty, tells the devices
  // which should be participating into the replicated computation.
  static void SyncLiveTensorsGraph(const torch::lazy::BackendDevice* device,
                                   absl::Span<const std::string> devices,
                                   bool wait);

  // Marks an execution step, which allows the tensor framework to understand
  // the computation boundaries.
  static void MarkStep(const torch::lazy::BackendDevice& device);

  // Waits for all the outstanding operations on all the supplied devices.
  // If devices is empty, the wait will happen for all local devices.
  static void WaitDeviceOps(absl::Span<const std::string> devices);

  // Retrieves the PyTorch CPU tensors behind the XLA tensors IR operations.
  // All the tensors must be on the same device.
  static std::vector<at::Tensor> GetTensors(std::vector<XLATensorPtr>* tensors);

  // Operation which creates XLA tensors out of PyTorch CPU tensors by batching
  // the requests to the computation servers.
  static std::vector<XLATensorPtr> CreateTensors(
      const std::vector<at::Tensor>& tensors,
      const std::vector<std::string>& devices);

  //////////////////////////////////////////////////////////////////////////////
  // XLA dedicated operators follows here, listed in alphabetical order.
  //////////////////////////////////////////////////////////////////////////////
  static std::pair<XLATensorPtr, torch::lazy::Value> all_reduce(
      const XLATensorPtr& input, const torch::lazy::Value& token,
      AllReduceType reduce_type, double scale,
      std::vector<std::vector<int64_t>> groups, bool pin_layout);

  static torch::lazy::Value all_reduce_(
      XLATensorPtr& input, const torch::lazy::Value& token,
      AllReduceType reduce_type, double scale,
      std::vector<std::vector<int64_t>> groups, bool pin_layout);

  static torch::lazy::Value all_reduce(std::vector<XLATensorPtr>* inputs,
                                       const torch::lazy::Value& token,
                                       AllReduceType reduce_type, double scale,
                                       std::vector<std::vector<int64_t>> groups,
                                       bool pin_layout);

  static std::pair<XLATensorPtr, torch::lazy::Value> reduce_scatter(
      const XLATensorPtr& input, const torch::lazy::Value& token,
      AllReduceType reduce_type, double scale, int64_t scatter_dim,
      int64_t shard_count, std::vector<std::vector<int64_t>> groups,
      bool pin_layout);

  static torch::lazy::Value reduce_scatter_out(
      XLATensorPtr& output, const XLATensorPtr& input,
      const torch::lazy::Value& token, AllReduceType reduce_type, double scale,
      int64_t scatter_dim, int64_t shard_count,
      std::vector<std::vector<int64_t>> groups, bool pin_layout);

  static std::pair<XLATensorPtr, torch::lazy::Value> all_to_all(
      const XLATensorPtr& input, const torch::lazy::Value& token,
      int64_t split_dimension, int64_t concat_dimension, int64_t split_count,
      std::vector<std::vector<int64_t>> groups, bool pin_layout);

  static std::pair<XLATensorPtr, torch::lazy::Value> all_gather(
      const XLATensorPtr& input, const torch::lazy::Value& token, int64_t dim,
      int64_t shard_count, std::vector<std::vector<int64_t>> groups,
      bool pin_layout);

  static torch::lazy::Value all_gather_out(
      XLATensorPtr& output, const XLATensorPtr& input,
      const torch::lazy::Value& token, int64_t dim, int64_t shard_count,
      std::vector<std::vector<int64_t>> groups, bool pin_layout);

  static std::pair<XLATensorPtr, torch::lazy::Value> collective_permute(
      const XLATensorPtr& input, const torch::lazy::Value& token,
      std::vector<std::pair<int64_t, int64_t>> source_target_pairs);

  static XLATensorPtr get_dimensions_size(const XLATensorPtr& input,
                                          std::vector<int64_t> dimensions);

  static std::pair<XLATensorPtr, torch::lazy::Value> recv(
      XLATensorPtr& output, const torch::lazy::Value& token,
      int64_t channel_id);

  static std::pair<XLATensorPtr, torch::lazy::Value> send(
      const XLATensorPtr& input, const torch::lazy::Value& token,
      int64_t channel_id);

  static void sgd_optimizer_step_(const XLATensorPtr& found_inf,
                                  XLATensorPtr& step, XLATensorPtr& param,
                                  XLATensorPtr& buf, const XLATensorPtr& d_p,
                                  double weight_decay, double momentum,
                                  double lr, double dampening, bool nesterov,
                                  bool maximize);

  static void adam_optimizer_step_(
      const XLATensorPtr& found_inf, XLATensorPtr& step, XLATensorPtr& param,
      const XLATensorPtr& grad, XLATensorPtr& exp_avg, XLATensorPtr& exp_avg_sq,
      XLATensorPtr& max_exp_avg_sq, double beta1, double beta2, double lr,
      double weight_decay, double eps, bool amsgrad, bool maximize,
      bool use_adamw);

  static std::vector<XLATensorPtr> user_computation(
      const std::string& opname, absl::Span<const XLATensorPtr> inputs,
      ComputationPtr computation);

  //////////////////////////////////////////////////////////////////////////////
  // ATEN operators follows here, listed in alphabetical order.
  //////////////////////////////////////////////////////////////////////////////
  static void __ilshift__(XLATensorPtr& input, const at::Scalar& other);
  static void __ilshift__(XLATensorPtr& input, const XLATensorPtr& other);

  static void __irshift__(XLATensorPtr& input, const at::Scalar& other);
  static void __irshift__(XLATensorPtr& input, const XLATensorPtr& other);

  static XLATensorPtr __lshift__(
      const XLATensorPtr& input, const at::Scalar& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static XLATensorPtr __lshift__(
      const XLATensorPtr& input, const XLATensorPtr& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static XLATensorPtr __rshift__(
      const XLATensorPtr& input, const at::Scalar& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static XLATensorPtr __rshift__(
      const XLATensorPtr& input, const XLATensorPtr& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static std::tuple<XLATensorPtr, XLATensorPtr> adaptive_max_pool2d(
      const XLATensorPtr& input, std::vector<int64_t> output_size);

  static XLATensorPtr adaptive_max_pool2d_backward(
      const XLATensorPtr& grad_output, const XLATensorPtr& input);

  static XLATensorPtr adaptive_avg_pool3d(const XLATensorPtr& input,
                                          std::vector<int64_t> output_size);

  static XLATensorPtr adaptive_avg_pool3d_backward(
      const XLATensorPtr& grad_output, const XLATensorPtr& input);

  static XLATensorPtr _adaptive_avg_pool2d(const XLATensorPtr& input,
                                           std::vector<int64_t> output_size);

  static XLATensorPtr _adaptive_avg_pool2d_backward(
      const XLATensorPtr& grad_output, const XLATensorPtr& input);

  static void _amp_foreach_non_finite_check_and_unscale_(
      std::vector<XLATensorPtr> self, XLATensorPtr& found_inf,
      const XLATensorPtr& inv_scale);

  static void _amp_update_scale_(XLATensorPtr& current_scale,
                                 XLATensorPtr& growth_tracker,
                                 const XLATensorPtr& found_inf,
                                 double scale_growth_factor,
                                 double scale_backoff_factor,
                                 int growth_interval);

  static XLATensorPtr abs(const XLATensorPtr& input);

  static XLATensorPtr add(
      const XLATensorPtr& input, const XLATensorPtr& other,
      const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static XLATensorPtr add(
      const XLATensorPtr& input, const at::Scalar& other,
      const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static XLATensorPtr addcdiv(const XLATensorPtr& input,
                              const at::Scalar& value,
                              const XLATensorPtr& tensor1,
                              const XLATensorPtr& tensor2);
  static void addcdiv_(XLATensorPtr& input, const at::Scalar& value,
                       const XLATensorPtr& tensor1,
                       const XLATensorPtr& tensor2);

  static XLATensorPtr addcmul(const XLATensorPtr& input,
                              const at::Scalar& value,
                              const XLATensorPtr& tensor1,
                              const XLATensorPtr& tensor2);

  static XLATensorPtr addmm(const XLATensorPtr& input,
                            const XLATensorPtr& weight,
                            const XLATensorPtr& bias);

  static XLATensorPtr all(const XLATensorPtr& input,
                          std::vector<int64_t> dimensions,
                          bool keep_reduced_dimensions);

  static XLATensorPtr amax(const XLATensorPtr& input,
                           std::vector<int64_t> dimensions,
                           bool keep_reduced_dimensions);

  static XLATensorPtr amin(const XLATensorPtr& input,
                           std::vector<int64_t> dimensions,
                           bool keep_reduced_dimensions);

  static XLATensorPtr any(const XLATensorPtr& input,
                          std::vector<int64_t> dimensions,
                          bool keep_reduced_dimensions);

  static void arange_out(XLATensorPtr& out, const at::Scalar& start,
                         const at::Scalar& end, const at::Scalar& step,
                         at::ScalarType scalar_type);

  static XLATensorPtr argmax(const XLATensorPtr& input, int64_t dim,
                             bool keepdim);
  static XLATensorPtr argmax(const XLATensorPtr& input);

  static XLATensorPtr argmin(const XLATensorPtr& input, int64_t dim,
                             bool keepdim);
  static XLATensorPtr argmin(const XLATensorPtr& input);

  // Takes a slice from the input as R1 at the specified offset and reshapes it
  // into the provided size.
  static XLATensorPtr as_strided(const XLATensorPtr& input,
                                 std::vector<int64_t> size,
                                 std::vector<int64_t> stride,
                                 c10::optional<int64_t> storage_offset);

  // In-place version of the method above.
  static void as_strided_(XLATensorPtr& input, std::vector<int64_t> size,
                          std::vector<int64_t> stride,
                          c10::optional<int64_t> storage_offset);

  static XLATensorPtr atan2(
      const XLATensorPtr& input, const XLATensorPtr& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static XLATensorPtr avg_pool_nd(const XLATensorPtr& input,
                                  int64_t spatial_dim_count,
                                  std::vector<int64_t> kernel_size,
                                  std::vector<int64_t> stride,
                                  std::vector<int64_t> padding, bool ceil_mode,
                                  bool count_include_pad);

  static XLATensorPtr avg_pool_nd_backward(
      const XLATensorPtr& out_backprop, const XLATensorPtr& input,
      int64_t spatial_dim_count, std::vector<int64_t> kernel_size,
      std::vector<int64_t> stride, std::vector<int64_t> padding, bool ceil_mode,
      bool count_include_pad);

  static XLATensorPtr baddbmm(const XLATensorPtr& input,
                              const XLATensorPtr& batch1,
                              const XLATensorPtr& batch2,
                              const at::Scalar& beta, const at::Scalar& alpha);

  static XLATensorPtr bernoulli(const XLATensorPtr& input, double probability);
  static XLATensorPtr bernoulli(const XLATensorPtr& input);
  static void bernoulli_(XLATensorPtr& input, double probability);
  static void bernoulli_(XLATensorPtr& input, const XLATensorPtr& probability);

  static XLATensorPtr binary_cross_entropy(const XLATensorPtr& input,
                                           const XLATensorPtr& target,
                                           const XLATensorPtr& weight,
                                           int64_t reduction);

  static XLATensorPtr binary_cross_entropy_backward(
      const XLATensorPtr& grad_output, const XLATensorPtr& input,
      const XLATensorPtr& target, const XLATensorPtr& weight,
      int64_t reduction);

  static XLATensorPtr bitwise_and(const XLATensorPtr& input,
                                  const at::Scalar& other);

  static XLATensorPtr bitwise_and(const XLATensorPtr& input,
                                  const XLATensorPtr& other);

  static XLATensorPtr bitwise_not(const XLATensorPtr& input);

  static XLATensorPtr bitwise_or(const XLATensorPtr& input,
                                 const at::Scalar& other);

  static XLATensorPtr bitwise_or(const XLATensorPtr& input,
                                 const XLATensorPtr& other);

  static XLATensorPtr bitwise_xor(const XLATensorPtr& input,
                                  const at::Scalar& other);

  static XLATensorPtr bitwise_xor(const XLATensorPtr& input,
                                  const XLATensorPtr& other);

  // Batch matrix multiplication. Both tensors must be 3D, the batch size must
  // match and the remaining two dimensions must be compatible for matrix
  // multiplication.
  static XLATensorPtr bmm(const XLATensorPtr& batch1,
                          const XLATensorPtr& batch2);

  // Broadcasts the given tensors according to broadcasting semantics.
  static std::vector<XLATensorPtr> broadcast_tensors(
      absl::Span<const XLATensorPtr> tensors);

  static XLATensorPtr cat(absl::Span<const XLATensorPtr> tensors, int64_t dim,
                          at::ScalarType dtype);

  static XLATensorPtr ceil(const XLATensorPtr& input);

  static XLATensorPtr celu(const XLATensorPtr& input, const at::Scalar& alpha);
  static void celu_(XLATensorPtr& input, const at::Scalar& alpha);

  static XLATensorPtr cholesky(const XLATensorPtr& input, bool upper);

  static XLATensorPtr clamp(const XLATensorPtr& input,
                            const c10::optional<at::Scalar>& min,
                            const c10::optional<at::Scalar>& max);
  static XLATensorPtr clamp(const XLATensorPtr& input,
                            const c10::optional<at::Tensor>& min,
                            const c10::optional<at::Tensor>& max);

  static XLATensorPtr clone(const XLATensorPtr& input);

  // Pad with the given value and size specified by the given list of low and
  // high paddings.
  static XLATensorPtr constant_pad_nd(const XLATensorPtr& input,
                                      absl::Span<const int64_t> pad,
                                      const at::Scalar& value);

  static XLATensorPtr convolution_overrideable(
      const XLATensorPtr& input, const XLATensorPtr& weight,
      const XLATensorPtr& bias, std::vector<int64_t> stride,
      std::vector<int64_t> padding, std::vector<int64_t> dilation,
      bool transposed, std::vector<int64_t> output_padding, int64_t groups);

  static std::tuple<XLATensorPtr, XLATensorPtr, XLATensorPtr>
  convolution_backward_overrideable(
      const XLATensorPtr& out_backprop, const XLATensorPtr& input,
      const XLATensorPtr& weight, std::vector<int64_t> stride,
      std::vector<int64_t> padding, std::vector<int64_t> dilation,
      bool transposed, std::vector<int64_t> output_padding, int64_t groups);

  static XLATensorPtr convolution_overrideable(
      const XLATensorPtr& input, const XLATensorPtr& weight,
      std::vector<int64_t> stride, std::vector<int64_t> padding,
      std::vector<int64_t> dilation, bool transposed,
      std::vector<int64_t> output_padding, int64_t groups);

  // Returns the cross product of the two input tensors in the given dimension.
  // If the dimension is not given, it defaults to the first dimension found
  // with the size 3.
  static XLATensorPtr cross(const XLATensorPtr& input,
                            const XLATensorPtr& other,
                            c10::optional<int64_t> dim);

  // Returns the cumulative product of elements of input in the given dimension.
  static XLATensorPtr cumprod(const XLATensorPtr& input, int64_t dim,
                              c10::optional<at::ScalarType> dtype);

  // Returns the cumulative sum of elements of input in the given dimension.
  static XLATensorPtr cumsum(const XLATensorPtr& input, int64_t dim,
                             c10::optional<at::ScalarType> dtype);

  // If the input is a matrix (2-D tensor), returns a 1-D tensor with the
  // diagonal elements of the input. If the input is a vector (1-D tensor),
  // returns a 2-D square tensor with the elements of input as the diagonal.
  static XLATensorPtr diag(const XLATensorPtr& input, int64_t offset);

  // Returns the diagonal of a matrix (2-D tensor) or batch of matrices. The
  // matrix dimensions are specified by dim1 and dim2, the diagonal by offset.
  static XLATensorPtr diagonal(const XLATensorPtr& input, int64_t offset,
                               int64_t dim1, int64_t dim2);

  static XLATensorPtr div(
      const XLATensorPtr& input, const XLATensorPtr& other,
      const c10::optional<c10::string_view>& rounding_mode = c10::nullopt,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static XLATensorPtr div(const XLATensorPtr& input, const at::Scalar& other);

  // A generalized contraction between tensors of arbitrary dimension defined by
  // the given equation and applied to the input tensors.
  static XLATensorPtr einsum(const std::string& equation,
                             absl::Span<const XLATensorPtr> tensors);

  static XLATensorPtr elu(const XLATensorPtr& input, const at::Scalar& alpha,
                          const at::Scalar& scale,
                          const at::Scalar& input_scale);
  static void elu_(XLATensorPtr& input, const at::Scalar& alpha,
                   const at::Scalar& scale, const at::Scalar& input_scale);
  static XLATensorPtr elu_backward(const XLATensorPtr& grad_output,
                                   const at::Scalar& alpha,
                                   const at::Scalar& scale,
                                   const at::Scalar& input_scale,
                                   const XLATensorPtr& output);

  static XLATensorPtr embedding_dense_backward(const XLATensorPtr& grad_output,
                                               const XLATensorPtr& indices,
                                               int64_t num_weights,
                                               int64_t padding_idx,
                                               bool scale_grad_by_freq);

  static XLATensorPtr eq(const XLATensorPtr& input, const at::Scalar& other);

  static XLATensorPtr eq(const XLATensorPtr& input, const XLATensorPtr& other);

  static XLATensorPtr exp(const XLATensorPtr& input);

  static XLATensorPtr expand(const XLATensorPtr& input,
                             std::vector<int64_t> size);

  static XLATensorPtr expand(const XLATensorPtr& input,
                          std::vector<torch::lazy::NodePtr>& size_nodes,
                          const std::vector<int64_t> upper_bounds,
                          const std::vector<bool> dynamic_dims);

  static XLATensor expm1(const XLATensor& input);

  static void exponential_(XLATensorPtr& input, double lambd);

  // Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
  static XLATensorPtr eye(int64_t lines, int64_t cols,
                          const torch::lazy::BackendDevice& device,
                          at::ScalarType element_type);

  static void eye_out(XLATensorPtr& out, int64_t lines, int64_t cols);

  // Fills the input with the given value.
  static void fill_(XLATensorPtr& input, const at::Scalar& value);

  // Flips (reverses) the values in the dimensions of the input tensor.
  static XLATensorPtr flip(const XLATensorPtr& input,
                           absl::Span<const int64_t> dims);

  static XLATensorPtr fmod(
      const XLATensorPtr& input, const XLATensorPtr& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static XLATensorPtr fmod(
      const XLATensorPtr& input, const at::Scalar& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static XLATensorPtr frac(const XLATensorPtr& input);

  static XLATensorPtr full(absl::Span<const int64_t> size,
                           const at::Scalar& fill_value,
                           const torch::lazy::BackendDevice& device,
                           at::ScalarType scalar_type);
  static XLATensorPtr full_like(const XLATensorPtr& input,
                                const at::Scalar& fill_value,
                                const torch::lazy::BackendDevice& device,
                                c10::optional<at::ScalarType> scalar_type);

  static XLATensorPtr gather(const XLATensorPtr& input, int64_t dim,
                             const XLATensorPtr& index);

  static XLATensorPtr ge(const XLATensorPtr& input, const at::Scalar& other);

  static XLATensorPtr ge(const XLATensorPtr& input, const XLATensorPtr& other);

  static XLATensorPtr gelu(const XLATensorPtr& input,
                           const c10::string_view approximate);

  static XLATensorPtr gelu_backward(const XLATensorPtr& grad,
                                    const XLATensorPtr& input,
                                    const c10::string_view approximate);

  static XLATensorPtr ger(const XLATensorPtr& input, const XLATensorPtr& vec2);

  static XLATensorPtr gt(const XLATensorPtr& input, const at::Scalar& other);

  static XLATensorPtr gt(const XLATensorPtr& input, const XLATensorPtr& other);

  // Gather slices from input into a result with shape specified by indices. The
  // shape of the indices are first made consistent using broadcast semantics.
  // For input of shape d1 x d2 x ... x dn and p indices of shape i1 x i2 x ...
  // x ik, the output shape is d1 x ... x d(start_dim) x i1 x ... x ik x
  // d(start_dim+p+1) x ... x dn.
  static XLATensorPtr index(const XLATensorPtr& input,
                            absl::Span<const XLATensorPtr> indices,
                            int64_t start_dim);

  static XLATensorPtr index_add(const XLATensorPtr& input, int64_t dim,
                                const XLATensorPtr& index,
                                const XLATensorPtr& source,
                                const at::Scalar& alpha);

  static XLATensorPtr index_copy(const XLATensorPtr& input, int64_t dim,
                                 const XLATensorPtr& index,
                                 const XLATensorPtr& source);

  // Fills the elements of the base tensor with the given value in the given
  // dimension, at positions given by the index. The index must be a rank-1
  // tensor.
  static XLATensorPtr index_fill(const XLATensorPtr& input, int64_t dim,
                                 const XLATensorPtr& index,
                                 const at::Scalar& value);

  // Same as above, but the value is wrapped as a rank-0 tensor.
  static XLATensorPtr index_fill(const XLATensorPtr& input, int64_t dim,
                                 const XLATensorPtr& index,
                                 const XLATensorPtr& value);

  static void index_fill_(XLATensorPtr& input, int64_t dim,
                          const XLATensorPtr& index, const XLATensorPtr& value);

  static void index_fill_(XLATensorPtr& input, int64_t dim,
                          const XLATensorPtr& index, const at::Scalar& value);

  // Puts values into the input tensor using the given indices (a tuple of
  // tensors) and returns the result.
  static XLATensorPtr index_put(const XLATensorPtr& input,
                                absl::Span<const XLATensorPtr> indices,
                                int64_t start_dim, const XLATensorPtr& values,
                                bool accumulate,
                                absl::Span<const int64_t> result_permutation);

  static void index_put_(XLATensorPtr& input,
                         const XLATensorPtr& canonical_base,
                         absl::Span<const XLATensorPtr> indices,
                         int64_t start_dim, const XLATensorPtr& values,
                         bool accumulate,
                         absl::Span<const int64_t> result_permutation);

  static XLATensorPtr index_select(const XLATensorPtr& input, int64_t dim,
                                   const XLATensorPtr& index);

  static XLATensorPtr isnan(const XLATensorPtr& input);

  static std::tuple<XLATensorPtr, XLATensorPtr> kthvalue(
      const XLATensorPtr& input, int64_t k, int64_t dim, bool keepdim);

  static XLATensorPtr le(const XLATensorPtr& input, const at::Scalar& other);

  static XLATensorPtr le(const XLATensorPtr& input, const XLATensorPtr& other);

  static XLATensorPtr hardshrink(const XLATensorPtr& input,
                                 const at::Scalar& lambda);
  static XLATensorPtr hardshrink_backward(const XLATensorPtr& grad_out,
                                          const XLATensorPtr& input,
                                          const at::Scalar& lambda);

  static XLATensorPtr hardsigmoid(const XLATensorPtr& input);

  static XLATensorPtr hardsigmoid_backward(const XLATensorPtr& grad_output,
                                           const XLATensorPtr& input);

  static XLATensorPtr hardswish(const XLATensorPtr& input);

  static XLATensorPtr hardswish_backward(const XLATensorPtr& grad_output,
                                         const XLATensorPtr& input);

  static XLATensorPtr hardtanh_backward(const XLATensorPtr& grad_output,
                                        const XLATensorPtr& input,
                                        const at::Scalar& min_val,
                                        const at::Scalar& max_val);

  static XLATensorPtr leaky_relu(const XLATensorPtr& input,
                                 double negative_slope);
  static XLATensorPtr leaky_relu_backward(const XLATensorPtr& grad_output,
                                          const XLATensorPtr& input,
                                          double negative_slope);

  static XLATensorPtr lerp(const XLATensorPtr& input, const XLATensorPtr& end,
                           const XLATensorPtr& weight);
  static XLATensorPtr lerp(const XLATensorPtr& input, const XLATensorPtr& end,
                           const at::Scalar& weight);

  static XLATensorPtr linspace(const at::Scalar& start, const at::Scalar& end,
                               const int64_t steps, at::ScalarType element_type,
                               const torch::lazy::BackendDevice& device);

  static XLATensorPtr log(const XLATensorPtr& input);

  static XLATensorPtr log_base(const XLATensorPtr& input,
                               torch::lazy::OpKind op, double base);

  static XLATensorPtr log_sigmoid(const XLATensorPtr& input);
  static std::tuple<XLATensorPtr, XLATensorPtr> log_sigmoid_forward(
      const XLATensorPtr& input);
  static XLATensorPtr log_sigmoid_backward(const XLATensorPtr& grad_output,
                                           const XLATensorPtr& input,
                                           const XLATensorPtr& buffer);

  static XLATensorPtr log_softmax(const XLATensorPtr& input, int64_t dim,
                                  c10::optional<at::ScalarType> dtype);

  static XLATensorPtr log_softmax_backward(const XLATensorPtr& grad_output,
                                           const XLATensorPtr& output,
                                           int64_t dim);

  static XLATensorPtr log1p(const XLATensorPtr& input);
  static void log1p_(XLATensorPtr& input);

  static XLATensorPtr logical_not(const XLATensorPtr& input);

  static XLATensorPtr logical_xor(const XLATensorPtr& input,
                                  const XLATensorPtr& other);

  static XLATensorPtr logical_and(const XLATensorPtr& input,
                                  const XLATensorPtr& other);

  static XLATensorPtr logical_or(const XLATensorPtr& input,
                                 const XLATensorPtr& other);

  static XLATensorPtr logsumexp(const XLATensorPtr& input,
                                std::vector<int64_t> dimensions,
                                bool keep_reduced_dimensions);

  static XLATensorPtr xlogy(const XLATensorPtr& input,
                            const XLATensorPtr& other);

  static XLATensorPtr lt(const XLATensorPtr& input, const at::Scalar& other);

  static XLATensorPtr lt(const XLATensorPtr& input, const XLATensorPtr& other);

  // In-place version of the method above.
  static void masked_fill_(XLATensorPtr& input, const XLATensorPtr& mask,
                           const at::Scalar& value);

  static void masked_scatter_(XLATensorPtr& input, const XLATensorPtr& mask,
                              const XLATensorPtr& source);

  static XLATensorPtr masked_select(const XLATensorPtr& input,
                                    const XLATensorPtr& mask);

  static XLATensorPtr matmul(const XLATensorPtr& input,
                             const XLATensorPtr& other);

  static XLATensorPtr max(const XLATensorPtr& input);

  static std::tuple<XLATensorPtr, XLATensorPtr> max(const XLATensorPtr& input,
                                                    int64_t dim, bool keepdim);

  static void max_out(XLATensorPtr& max, XLATensorPtr& max_values,
                      const XLATensorPtr& input, int64_t dim, bool keepdim);

  static std::tuple<XLATensorPtr, XLATensorPtr> max_pool_nd(
      const XLATensorPtr& input, int64_t spatial_dim_count,
      std::vector<int64_t> kernel_size, std::vector<int64_t> stride,
      std::vector<int64_t> padding, bool ceil_mode);

  static XLATensorPtr max_pool_nd_backward(const XLATensorPtr& out_backprop,
                                           const XLATensorPtr& input,
                                           int64_t spatial_dim_count,
                                           std::vector<int64_t> kernel_size,
                                           std::vector<int64_t> stride,
                                           std::vector<int64_t> padding,
                                           bool ceil_mode);

  static XLATensorPtr max_unpool(const XLATensorPtr& input,
                                 const XLATensorPtr& indices,
                                 std::vector<int64_t> output_size);

  static XLATensorPtr max_unpool_backward(const XLATensorPtr& grad_output,
                                          const XLATensorPtr& input,
                                          const XLATensorPtr& indices,
                                          std::vector<int64_t> output_size);

  static XLATensorPtr mean(const XLATensorPtr& input,
                           std::vector<int64_t> dimensions,
                           bool keep_reduced_dimensions,
                           c10::optional<at::ScalarType> dtype);

  static XLATensorPtr min(
      const XLATensorPtr& input, const XLATensorPtr& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static XLATensorPtr min(const XLATensorPtr& input);

  static std::tuple<XLATensorPtr, XLATensorPtr> min(const XLATensorPtr& input,
                                                    int64_t dim, bool keepdim);

  static void min_out(XLATensorPtr& min, XLATensorPtr& min_indices,
                      const XLATensorPtr& input, int64_t dim, bool keepdim);

  static XLATensorPtr mish(const XLATensorPtr& input);

  static XLATensorPtr mm(const XLATensorPtr& input, const XLATensorPtr& weight);

  static XLATensorPtr mse_loss(const XLATensorPtr& input,
                               const XLATensorPtr& target, int64_t reduction);

  static XLATensorPtr mse_loss_backward(const XLATensorPtr& grad_output,
                                        const XLATensorPtr& input,
                                        const XLATensorPtr& target,
                                        int64_t reduction);

  static XLATensorPtr mul(
      const XLATensorPtr& input, const XLATensorPtr& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static XLATensorPtr mul(
      const XLATensorPtr& input, const at::Scalar& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static XLATensorPtr mv(const XLATensorPtr& input, const XLATensorPtr& vec);
  static void mv_out(XLATensorPtr& out, const XLATensorPtr& input,
                     const XLATensorPtr& vec);

  static XLATensorPtr nan_to_num(const XLATensorPtr& input,
                                 const at::Scalar& nan,
                                 const at::Scalar& posinf,
                                 const at::Scalar& neginf);

  // Returns a new tensor that is a narrowed view of the input in the given
  // dimension.
  static XLATensorPtr narrow(const XLATensorPtr& input, int64_t dim,
                             int64_t start, int64_t length);

  // Like batch_norm, but returns additional save_mean and save_invstd used by
  // the backward pass.
  static std::tuple<XLATensorPtr, XLATensorPtr, XLATensorPtr> native_batch_norm(
      const XLATensorPtr& input, const XLATensorPtr& weight,
      const XLATensorPtr& bias, XLATensorPtr& running_mean,
      XLATensorPtr& running_var, bool training, double momentum, double eps);

  // Returns the input, weight and bias gradients.
  static std::tuple<XLATensorPtr, XLATensorPtr, XLATensorPtr>
  native_batch_norm_backward(const XLATensorPtr& grad_out,
                             const XLATensorPtr& input,
                             const XLATensorPtr& weight,
                             const XLATensorPtr& save_mean,
                             const XLATensorPtr& save_invstd, bool training,
                             double eps);

  static XLATensorPtr ne(const XLATensorPtr& input, const at::Scalar& other);

  static XLATensorPtr ne(const XLATensorPtr& input, const XLATensorPtr& other);

  static XLATensorPtr neg(const XLATensorPtr& input);

  static XLATensorPtr nll_loss(const XLATensorPtr& input,
                               const XLATensorPtr& target,
                               const XLATensorPtr& weight, int64_t reduction,
                               int ignore_index);

  static XLATensorPtr nll_loss2d(const XLATensorPtr& input,
                                 const XLATensorPtr& target,
                                 const XLATensorPtr& weight, int64_t reduction,
                                 int ignore_index);

  static XLATensorPtr nll_loss2d_backward(const XLATensorPtr& grad_output,
                                          const XLATensorPtr& input,
                                          const XLATensorPtr& target,
                                          const XLATensorPtr& weight,
                                          int64_t reduction, int ignore_index,
                                          const XLATensorPtr& total_weight);

  static XLATensorPtr nll_loss_backward(const XLATensorPtr& grad_output,
                                        const XLATensorPtr& input,
                                        const XLATensorPtr& target,
                                        const XLATensorPtr& weight,
                                        int64_t reduction, int ignore_index,
                                        const XLATensorPtr& total_weight);

  static std::pair<XLATensorPtr, XLATensorPtr> nms(
      const XLATensorPtr& boxes, const XLATensorPtr& scores,
      const XLATensorPtr& score_threshold, const XLATensorPtr& iou_threshold,
      int64_t output_size);

  static XLATensorPtr nonzero(const XLATensorPtr& input);

  static XLATensorPtr norm(const XLATensorPtr& input,
                           const c10::optional<at::Scalar>& p,
                           c10::optional<at::ScalarType> dtype,
                           at::IntArrayRef dim, bool keepdim);

  static XLATensorPtr normal(double mean, const XLATensorPtr& std);

  static XLATensorPtr normal(const XLATensorPtr& mean, double std);

  static XLATensorPtr normal(const XLATensorPtr& mean, const XLATensorPtr& std);

  static void normal_(XLATensorPtr& input, double mean, double std);

  static XLATensorPtr not_supported(std::string description, xla::Shape shape,
                                    const torch::lazy::BackendDevice& device);

  static void optimization_barrier_(std::vector<XLATensorPtr>& tensors);

  // Permute the dimensions of this tensor according to the given permutation.
  static XLATensorPtr permute(const XLATensorPtr& input,
                              absl::Span<const int64_t> dims);

  static XLATensorPtr pow(const XLATensorPtr& input,
                          const at::Scalar& exponent);
  static XLATensorPtr pow(const XLATensorPtr& input,
                          const XLATensorPtr& exponent);
  static XLATensorPtr pow(const at::Scalar& input,
                          const XLATensorPtr& exponent);

  static XLATensorPtr prelu(const XLATensorPtr& input,
                            const XLATensorPtr& weight);

  static XLATensorPtr prod(const XLATensorPtr& input,
                           std::vector<int64_t> dimensions,
                           bool keep_reduced_dimensions,
                           c10::optional<at::ScalarType> dtype);

  static void put_(XLATensorPtr& input, const XLATensorPtr& index,
                   const XLATensorPtr& source, bool accumulate);

  static std::tuple<XLATensorPtr, XLATensorPtr> qr(const XLATensorPtr& input,
                                                   bool some);

  static void random_(XLATensorPtr& input, int64_t from, int64_t to);

  static XLATensorPtr randperm(int64_t n,
                               const torch::lazy::BackendDevice& device,
                               at::ScalarType scalar_type);

  static XLATensorPtr reflection_pad2d(const XLATensorPtr& input,
                                       std::vector<int64_t> padding);

  static XLATensorPtr reflection_pad2d_backward(const XLATensorPtr& grad_output,
                                                const XLATensorPtr& input,
                                                std::vector<int64_t> padding);

  static XLATensorPtr relu(const XLATensorPtr& input);
  static void relu_(XLATensorPtr& input);

  static XLATensorPtr remainder(const XLATensorPtr& input,
                                const XLATensorPtr& other);
  static XLATensorPtr remainder(const XLATensorPtr& input,
                                const at::Scalar& other);

  // Repeats the input tensor along each dimension by the given number of
  // repeats.
  static XLATensorPtr repeat(const XLATensorPtr& input,
                             std::vector<int64_t> repeats);

  static XLATensorPtr replication_pad1d(const XLATensorPtr& input,
                                        std::vector<int64_t> padding);
  static XLATensorPtr replication_pad1d_backward(
      const XLATensorPtr& grad_output, const XLATensorPtr& input,
      std::vector<int64_t> padding);

  static XLATensorPtr replication_pad2d(const XLATensorPtr& input,
                                        std::vector<int64_t> padding);
  static XLATensorPtr replication_pad2d_backward(
      const XLATensorPtr& grad_output, const XLATensorPtr& input,
      std::vector<int64_t> padding);

  static void resize_(XLATensorPtr& input, std::vector<int64_t> size);

  static XLATensorPtr roll(const XLATensorPtr& input,
                           absl::Span<const int64_t> shifts,
                           absl::Span<const int64_t> dims);

  static XLATensorPtr round(const XLATensorPtr& input);

  static XLATensorPtr rrelu_with_noise(const XLATensorPtr& input,
                                       XLATensorPtr& noise,
                                       const at::Scalar& lower,
                                       const at::Scalar& upper, bool training);

  static XLATensorPtr rrelu_with_noise_backward(const XLATensorPtr& grad_output,
                                                const XLATensorPtr& input,
                                                const XLATensorPtr& noise,
                                                const at::Scalar& lower,
                                                const at::Scalar& upper,
                                                bool training);

  static XLATensorPtr rsqrt(const XLATensorPtr& input);

  static XLATensorPtr rsub(
      const XLATensorPtr& input, const XLATensorPtr& other,
      const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static XLATensorPtr rsub(
      const XLATensorPtr& input, const at::Scalar& other,
      const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static void copy_(XLATensorPtr& input, XLATensorPtr& src);

  static XLATensorPtr scatter(const XLATensorPtr& input, int64_t dim,
                              const XLATensorPtr& index,
                              const XLATensorPtr& src);
  static XLATensorPtr scatter(const XLATensorPtr& input, int64_t dim,
                              const XLATensorPtr& index,
                              const at::Scalar& value);

  static XLATensorPtr scatter_add(const XLATensorPtr& input, int64_t dim,
                                  const XLATensorPtr& index,
                                  const XLATensorPtr& src);
  static XLATensorPtr scatter_add(const XLATensorPtr& input, int64_t dim,
                                  const XLATensorPtr& index,
                                  const at::Scalar& value);

  static XLATensorPtr select(const XLATensorPtr& input, int64_t dim,
                             int64_t index);

  static XLATensorPtr selu(const XLATensorPtr& input);
  static void selu_(XLATensorPtr& input);

  static XLATensorPtr silu(const XLATensorPtr& input);
  static XLATensorPtr silu_backward(XLATensorPtr& grad_output,
                                    XLATensorPtr& input);
  static XLATensorPtr sigmoid(const XLATensorPtr& input);
  static XLATensorPtr sigmoid_backward(const XLATensorPtr& grad_output,
                                       const XLATensorPtr& output);

  static XLATensorPtr slice(const XLATensorPtr& input, int64_t dim,
                            int64_t start, int64_t end, int64_t step);

  static std::tuple<XLATensorPtr, XLATensorPtr> slogdet(
      const XLATensorPtr& input);

  // Computes a loss that uses a squared term if the absolute element-wise error
  // falls below 1 and an L1 term otherwise.
  static XLATensorPtr smooth_l1_loss(const XLATensorPtr& input,
                                     const XLATensorPtr& target,
                                     int64_t reduction, double beta);

  // Returns the gradient of the input of a smooth_l1_loss operation.
  static XLATensorPtr smooth_l1_loss_backward(const XLATensorPtr& grad_output,
                                              const XLATensorPtr& input,
                                              const XLATensorPtr& target,
                                              int64_t reduction, double beta);

  static XLATensorPtr softmax(const XLATensorPtr& input, int64_t dim,
                              c10::optional<at::ScalarType> dtype);
  static XLATensorPtr softmax_backward(const XLATensorPtr& grad_output,
                                       const XLATensorPtr& output, int64_t dim);

  static XLATensorPtr softplus(const XLATensorPtr& input,
                               const at::Scalar& beta,
                               const at::Scalar& threshold);

  static XLATensorPtr softplus_backward(const XLATensorPtr& grad_output,
                                        const XLATensorPtr& input,
                                        const at::Scalar& beta,
                                        const at::Scalar& threshold);

  static XLATensorPtr softshrink(const XLATensorPtr& input,
                                 const at::Scalar& lambda);
  static XLATensorPtr softshrink_backward(const XLATensorPtr& grad_out,
                                          const XLATensorPtr& input,
                                          const at::Scalar& lambda);

  static std::vector<XLATensorPtr> split(const XLATensorPtr& input,
                                         int64_t split_size, int64_t dim);

  static std::vector<XLATensorPtr> split_with_sizes(
      const XLATensorPtr& input, std::vector<int64_t> split_size, int64_t dim);

  static XLATensorPtr sqrt(const XLATensorPtr& input);

  // Squeeze out all trivial (size 1) dimensions.
  static XLATensorPtr squeeze(const XLATensorPtr& input);

  // Squeeze out the specified dimension index, if trivial (size 1). Returns
  // unchanged input otherwise.
  static XLATensorPtr squeeze(const XLATensorPtr& input, int64_t dim);

  // In-place versions of the methods above.
  static void squeeze_(XLATensorPtr& input);
  static void squeeze_(XLATensorPtr& input, int64_t dim);

  static XLATensorPtr stack(absl::Span<const XLATensorPtr> tensors,
                            int64_t dim);

  static XLATensorPtr std(const XLATensorPtr& input,
                          std::vector<int64_t> dimensions,
                          bool keep_reduced_dimensions, int64_t correction);

  static std::tuple<XLATensorPtr, XLATensorPtr> std_mean(
      const XLATensorPtr& input, std::vector<int64_t> dimensions,
      int64_t correction, bool keep_reduced_dimensions);

  static XLATensorPtr sub(
      const XLATensorPtr& input, const XLATensorPtr& other,
      const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static XLATensorPtr sub(
      const XLATensorPtr& input, const at::Scalar& other,
      const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static XLATensorPtr sum(const XLATensorPtr& input,
                          std::vector<int64_t> dimensions,
                          bool keep_reduced_dimensions,
                          c10::optional<at::ScalarType> dtype);

  static std::tuple<XLATensorPtr, XLATensorPtr, XLATensorPtr> svd(
      const XLATensorPtr& input, bool some, bool compute_uv);

  static std::tuple<XLATensorPtr, XLATensorPtr> symeig(
      const XLATensorPtr& input, bool eigenvectors, bool upper);

  static XLATensorPtr take(const XLATensorPtr& input,
                           const XLATensorPtr& index);

  static XLATensorPtr tanh(const XLATensorPtr& input);

  static XLATensorPtr tanh_backward(const XLATensorPtr& grad_output,
                                    const XLATensorPtr& output);

  static XLATensorPtr threshold(const XLATensorPtr& input, float threshold,
                                float value);

  static XLATensorPtr threshold_backward(const XLATensorPtr& grad_output,
                                         const XLATensorPtr& input,
                                         float threshold);

  static XLATensorPtr to(XLATensorPtr& input,
                         c10::optional<torch::lazy::BackendDevice> device,
                         c10::optional<at::ScalarType> scalar_type);

  static std::tuple<XLATensorPtr, XLATensorPtr> topk(const XLATensorPtr& input,
                                                     int64_t k, int64_t dim,
                                                     bool largest, bool sorted,
                                                     bool stable);

  // Returns the sum of the elements of the diagonal of the input 2-D matrix.
  static XLATensorPtr trace(const XLATensorPtr& input);

  // Swap given dimensions of the input.
  static XLATensorPtr transpose(const XLATensorPtr& input, int64_t dim0,
                                int64_t dim1);

  // In-place version of the method above.
  static void transpose_(XLATensorPtr& input, int64_t dim0, int64_t dim1);

  static std::tuple<XLATensorPtr, XLATensorPtr> triangular_solve(
      const XLATensorPtr& rhs, const XLATensorPtr& lhs, bool left_side,
      bool upper, bool transpose, bool unitriangular);

  // Returns the lower triangular part of a matrix (2-D tensor) or batch of
  // matrices input, the other elements of the result tensor out are set to 0.
  static XLATensorPtr tril(const XLATensorPtr& input, int64_t diagonal);

  // In-place version of the method above.
  static void tril_(XLATensorPtr& input, int64_t diagonal);

  // Returns the upper triangular part of a matrix (2-D tensor) or batch of
  // matrices input, the other elements of the result tensor out are set to 0.
  static XLATensorPtr triu(const XLATensorPtr& input, int64_t diagonal);

  // In-place version of the method above.
  static void triu_(XLATensorPtr& input, int64_t diagonal);

  static XLATensorPtr trunc(const XLATensorPtr& input);

  // Returns a tuple of all slices along a given dimension with that dimension
  // removed.
  static std::vector<XLATensorPtr> unbind(const XLATensorPtr& input,
                                          int64_t dim);

  static void uniform_(XLATensorPtr& input, double from, double to);

  // Insert a dimension of size one at the specified position.
  static XLATensorPtr unsqueeze(const XLATensorPtr& input, int64_t dim);

  // In-place version of the method above.
  static void unsqueeze_(XLATensorPtr& input, int64_t dim);

  static XLATensorPtr upsample_bilinear2d(const XLATensorPtr& input,
                                          std::vector<int64_t> output_size,
                                          bool align_corners);

  static XLATensorPtr upsample_bilinear2d_backward(
      const XLATensorPtr& grad_output, std::vector<int64_t> output_size,
      std::vector<int64_t> input_size, bool align_corners);

  static XLATensorPtr upsample_nearest2d(const XLATensorPtr& input,
                                         std::vector<int64_t> output_size);

  static XLATensorPtr upsample_nearest2d_backward(
      const XLATensorPtr& grad_output, std::vector<int64_t> output_size,
      std::vector<int64_t> input_size);

  static XLATensorPtr var(const XLATensorPtr& input,
                          std::vector<int64_t> dimensions, int64_t correction,
                          bool keep_reduced_dimensions);

  static std::tuple<XLATensorPtr, XLATensorPtr> var_mean(
      const XLATensorPtr& input, std::vector<int64_t> dimensions,
      int64_t correction, bool keep_reduced_dimensions);

  // Like reshape, but it returns a view into the original tensor.
  static XLATensorPtr view(const XLATensorPtr& input,
                           absl::Span<const int64_t> output_size);

  static void zero_(XLATensorPtr& input);

  static XLATensorPtr where(const XLATensorPtr& condition,
                            const XLATensorPtr& input,
                            const XLATensorPtr& other);

  // XLA SPMD sharding spec annoation. The XLA tensor uses this to create
  // HloSharding for replication, manual and tile shardings.
  struct ShardingSpec {
    ShardingSpec(const xla::OpSharding& sharding, bool replicated, bool manual)
        : sharding(sharding), replicated(replicated), manual(manual) {}

    const xla::OpSharding sharding;
    bool replicated;
    bool manual;
  };

  std::shared_ptr<ShardingSpec> sharding_spec() const;
  bool IsShardingAnnotated() const;
  void SetShardingSpec(const xla::OpSharding& sharding, bool replicated,
                       bool manual);
  void ClearShardingSpec();

 private:
  struct SyncTensorsConfig {
    // Whether we want to force XLA data on the target tensors (hence trimming
    // the IR graph above them).
    bool force_xla_data = true;
    // Whether when setting the XLA data, the other properties of the tensor
    // state should be reset.
    bool sync_xla_data = true;
  };

  struct SyncTensorCollection {
    SyncTensorCollection() : hash(0) {}

    SyncTensorsConfig config;
    std::vector<size_t> indices;
    torch::lazy::hash_t hash;
    std::vector<xla::util::ExceptionCleanup> unlocker;
    torch::lazy::BackendDevice device;
  };

  struct PostOrderData {
    std::vector<const torch::lazy::Node*> post_order;
    torch::lazy::Util::EmissionMap emission_map;
    std::vector<torch::lazy::BackendDataPtr> parameters_data;
    std::vector<size_t> parameter_sequence;
  };

  struct CompilationResult {
    torch::lazy::BackendDevice device;
    size_t emitted_nodes = 0;
    std::shared_ptr<xla::ComputationClient::Computation> computation;
    std::vector<torch::lazy::BackendDataPtr> parameters_data;
  };

  struct CachedComputation {
    CachedComputation(
        std::shared_ptr<xla::ComputationClient::Computation> computation)
        : computation(std::move(computation)) {}

    std::shared_ptr<xla::ComputationClient::Computation> computation;
  };

  using ComputationCache =
      xla::util::Cache<torch::lazy::hash_t, CachedComputation,
                       torch::lazy::HashReducer>;

  struct Async {
    Async(SyncTensorCollection* coll,
          std::vector<torch::lazy::BackendDataPtr> parameters_data,
          std::vector<torch::lazy::BackendDataPtr> tensors_data,
          ComputationCache::TypePtr cached_computation);

    void Wait();

    xla::util::MultiWait mwait;
    std::vector<size_t> indices;
    std::vector<xla::util::ExceptionCleanup> unlocker;
    std::vector<torch::lazy::BackendDataPtr> parameters_data;
    std::string device;
    ComputationCache::TypePtr cached_computation;
    std::vector<torch::lazy::BackendDataPtr> tensors_data;
  };

  // This is the core XLA tensor data structure where all the tensor data is
  // held. The XLA tensor is nothing more than a shared pointer to a Data
  // object.
  struct Data {
    Data(torch::lazy::BackendDataPtr xla_data,
         const torch::lazy::BackendDevice& device,
         c10::optional<at::ScalarType> logical_element_type)
        : xla_data(std::move(xla_data)),
          logical_element_type(logical_element_type),
          device(device),
          unique_id(GetNextTensorId()) {}
    Data(torch::lazy::Value ir_value, const torch::lazy::BackendDevice& device,
         c10::optional<at::ScalarType> logical_element_type)
        : ir_value(std::move(ir_value)),
          logical_element_type(logical_element_type),
          device(device),
          unique_id(GetNextTensorId()) {}
    Data(std::shared_ptr<View> view, const torch::lazy::BackendDevice& device,
         c10::optional<at::ScalarType> logical_element_type)
        : view(std::move(view)),
          logical_element_type(logical_element_type),
          device(device),
          unique_id(GetNextTensorId()) {}
    Data(at::Tensor tensor_data, const torch::lazy::BackendDevice& device)
        : logical_element_type(tensor_data.scalar_type()),
          tensor_data(std::move(tensor_data)),
          device(device),
          unique_id(GetNextTensorId()) {}

    ~Data();

    torch::lazy::BackendDataPtr xla_data;
    torch::lazy::Value ir_value;
    std::shared_ptr<View> view;
    // TODO: remove this in favor of torch::lazy::Shape within ir_value.
    c10::optional<at::ScalarType> logical_element_type;
    c10::optional<at::Tensor> tensor_data;
    const torch::lazy::BackendDevice device;
    const int64_t unique_id = 0;
    size_t generation = 1;

    // Sharding annotation for the tensor
    // TODO(yeounoh) detach & clear for the unpartitioned tensor
    std::shared_ptr<ShardingSpec> sharding_spec;
  };

  XLATensor(const at::Tensor& tensor, const torch::lazy::BackendDevice& device);
  XLATensor(torch::lazy::BackendDataPtr xla_data,
            c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  XLATensor(torch::lazy::Value ir_value,
            const torch::lazy::BackendDevice& device,
            c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  XLATensor(std::shared_ptr<View> view,
            const torch::lazy::BackendDevice& device,
            c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  XLATensor(std::shared_ptr<Data> data);

  static XLATensorPtr Create(
      std::shared_ptr<View> view, const torch::lazy::BackendDevice& device,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  Data* data() const;

  std::shared_ptr<Data> data_ptr() const { return data_; }

  void SetXlaData(torch::lazy::BackendDataPtr xla_data, bool sync);

  void SetIrValue(torch::lazy::Value ir_value, bool inplace = true);
  void SetInPlaceIrValue(torch::lazy::Value ir_value);

  void AssignIrValue(torch::lazy::Value ir_value) const;

  void SetTensorData(at::Tensor tensor_data);

  torch::lazy::Value CreateTensorNode(torch::lazy::BackendDataPtr data,
                                      bool read_only) const;

  View::IrNode GetViewUpdate(const std::shared_ptr<View>& view) const;

  std::shared_ptr<View> UpdateView(std::shared_ptr<View> view,
                                   torch::lazy::Value ir_value) const;

  void SetSubView(ViewInfo view_info) const;
  void ModifyCurrentView(ViewInfo view_info) const;
  std::shared_ptr<View> CreateView(ViewInfo view_info) const;
  XLATensorPtr CreateViewTensor(ViewInfo view_info) const;

  XLATensorPtr CopyTensorToDevice(const torch::lazy::BackendDevice& device);

  torch::lazy::Value MaybeCastIrValue(
      torch::lazy::Value ir_value, const torch::lazy::BackendDevice& device,
      c10::optional<at::ScalarType> logical_element_type) const;

  // Create a new XLA tensor with the same metadata of the input tensor (with
  // possible overrides), and the new IR value.
  XLATensorPtr CreateFrom(torch::lazy::Value ir_value) const;
  XLATensorPtr CreateFrom(torch::lazy::Value ir_value,
                          const torch::lazy::BackendDevice& device) const;
  XLATensorPtr CreateFrom(torch::lazy::Value ir_value,
                          at::ScalarType logical_element_type) const;
  XLATensorPtr CreateFrom(
      torch::lazy::Value ir_value,
      c10::optional<at::ScalarType> logical_element_type_opt) const;
  XLATensorPtr CreateFrom(torch::lazy::Value ir_value,
                          const torch::lazy::BackendDevice& device,
                          at::ScalarType logical_element_type) const;

  // We build an XLA graph accumulating XLA operations, but at a given point we
  // need to force a rendering, otherwise the graph can grow without control.
  // Think:
  //   for i in range(0, 100000):
  //     a = a + b
  void TryLimitGraphSize();

  std::vector<XLATensorPtr> MakeOutputTensors(
      torch::lazy::NodePtr node, bool inherit_logical_type = true) const;

  torch::lazy::Value GetIrValueForTensor(
      const at::Tensor& tensor, const torch::lazy::BackendDevice& device) const;

  static ComputationCache* GetComputationCache();

  static SyncTensorCollection CollectSyncTensors(
      const std::vector<XLATensorPtr>& tensors,
      const SyncTensorsConfig& config);

  // Waits for this SyncTensorCollection's device barrier and acuire the lock.
  static void TensorCollectionBarrier(SyncTensorCollection* coll);

  // Implementation of the GetTensors() API using the op-by-op executor.
  static std::vector<at::Tensor> GetTensorsOpByOp(
      std::vector<XLATensorPtr>* tensors);

  static std::vector<at::Tensor> GetTensorsFused(
      std::vector<XLATensorPtr>* tensors);

  // Runs an asynchronous syn operation using the op-by-op executor.
  using OpByOpAsync = xla::util::AsyncTask<int>;
  static OpByOpAsync SyncTensorsGraphOpByOp(
      std::vector<XLATensorPtr>* tensors, absl::Span<const std::string> devices,
      const SyncTensorsConfig& config);

  // Gathers the XLA device data for all the input tensors, after an
  // asynchronous operation.
  static std::vector<torch::lazy::BackendDataPtr> GatherTensorsXlaData(
      const std::vector<XLATensorPtr>& tensors,
      absl::Span<const size_t> indices,
      absl::Span<const torch::lazy::BackendDataPtr> tensors_data);

  static std::vector<torch::lazy::Value> CollectRoots(
      const std::vector<XLATensorPtr>& tensors,
      absl::Span<const size_t> indices);

  static std::vector<torch::lazy::BackendDataPtr> FetchTensorData(
      std::vector<XLATensorPtr>* tensors, const SyncTensorsConfig& config,
      absl::Span<const size_t> indices);

  static std::vector<at::Tensor> FetchTensors(
      std::vector<XLATensorPtr>* tensors,
      absl::Span<const xla::Literal> literals,
      const std::vector<size_t>* indices);

  // Schedules the execution of a sync tensors operation in background. The
  // asynchronous operation will hold the device locks by capturing the ones
  // present within the coll structure.
  static std::shared_ptr<XLATensor::Async> ScheduleSyncTensorsGraph(
      SyncTensorCollection* coll,
      std::vector<torch::lazy::BackendDataPtr> parameters_data,
      std::vector<torch::lazy::BackendDataPtr> tensors_data,
      ComputationCache::TypePtr cached_computation);

  static std::shared_ptr<Async> ScheduleSyncTensorsGraph(
      std::vector<XLATensorPtr>* tensors, SyncTensorCollection* coll,
      std::vector<torch::lazy::BackendDataPtr> parameters_data,
      std::string device, ComputationCache::TypePtr cached_computation);

  static PostOrderData RunPostOrder(const std::vector<XLATensorPtr>& tensors,
                                    SyncTensorCollection* coll);

  static ComputationCache::TypePtr LookupCachedCompile(
      const std::vector<XLATensorPtr>& tensors,
      const torch::lazy::hash_t& hash);

  static std::shared_ptr<Async> TryRunCachedSync(
      std::vector<XLATensorPtr>* tensors, SyncTensorCollection* coll,
      PostOrderData* po_data);

  static void BuildInputOutputAliases(const std::vector<XLATensorPtr>& tensors,
                                      absl::Span<const size_t> indices,
                                      LoweringContext* lowering_ctx);

  static CompilationResult Compile(const std::vector<XLATensorPtr>& tensors,
                                   absl::Span<const std::string> devices,
                                   const SyncTensorCollection& coll,
                                   PostOrderData* po_data);

  static std::shared_ptr<Async> SyncTensorsGraphInternal(
      std::vector<XLATensorPtr>* tensors, absl::Span<const std::string> devices,
      const SyncTensorsConfig& config);

  static int64_t GetNextTensorId();

  static bool UseEagerDebugMode();

  bool ShouldSyncIrNode();

  std::shared_ptr<Data> data_;
};

}  // namespace torch_xla
