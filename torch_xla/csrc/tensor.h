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
#include "torch_xla/csrc/view.h"
#include "torch_xla/csrc/xla_sharding_util.h"

namespace torch_xla {

class XLATensor : public c10::intrusive_ptr_target {
  class DeviceContextArena;
  struct Data;

 public:
  static XLATensor Create(const at::Tensor& tensor,
                          const torch::lazy::BackendDevice& device);
  static XLATensor Create(
      torch::lazy::BackendDataPtr xla_data,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static XLATensor Create(
      torch::lazy::Value ir_value, const torch::lazy::BackendDevice& device,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  // Creates an empty/null tensor.
  XLATensor() = default;

  bool is_null() const { return data_ptr() == nullptr; }

  size_t generation() const { return data()->generation; }

  XLATensor alias() const { return XLATensor(data_ptr()); }

  int64_t size(int64_t dim) const;

  at::Tensor ToTensor(bool detached);

  void ShallowCopyTo(XLATensor* dest) const;

  // Assigns the tensor value to the XLA tensor.
  void SetTensor(at::Tensor tensor);

  void UpdateFromTensor(at::Tensor tensor, bool sync);
  void UpdateFromTensorOut(at::Tensor tensor);
  void UpdateFromTensorOut(const XLATensor& tensor);

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
  static void ApplyEagerSync(std::vector<XLATensor>& tensors);

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
  static XLATensor DispatchComparisonOp(c10::Symbol kind,
                                        const XLATensor& input,
                                        const at::Scalar& other);

  // Same as above, with the second input a tensor as well.
  static XLATensor DispatchComparisonOp(c10::Symbol kind,
                                        const XLATensor& input,
                                        const XLATensor& other);

  // Dumps the XLA HLO text of the computation accumulated in the graph which is
  // attached the tensors.
  static std::string DumpHloComputation(const std::vector<XLATensor>& tensors);

  // Retrieves the set of XLA tensors which are currently live in the system,
  // for the given device. If device is nullptr, the live tensors for all
  // devices will be returned. Returned tensors are sorted by device as primary
  // key, and by unique ID as secondary key.
  static std::vector<XLATensor> GetLiveTensors(
      const torch::lazy::BackendDevice* device);

  // Applies all the pending IR operations queued over the input tensors. All
  // the tensors must be on the same device. If wait is true, the sync operation
  // will be run synchronously. The devices argument, if not empty, tells the
  // devices which should be partecipating into the replicated computation.
  static void SyncTensorsGraph(std::vector<XLATensor>* tensors,
                               absl::Span<const std::string> devices, bool wait,
                               bool sync_xla_data);

  // Makes sure that any outstanding IR operation accumulated over live tensors,
  // gets turned into device data. If wait is true, the sync operation will be
  // run synchronously. The devices argument, if not empty, tells the devices
  // which should be partecipating into the replicated computation.
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
  static std::vector<at::Tensor> GetTensors(std::vector<XLATensor>* tensors);

  // Operation which creates XLA tensors out of PyTorch CPU tensors by batching
  // the requests to the computation servers.
  static std::vector<XLATensor> CreateTensors(
      const std::vector<at::Tensor>& tensors,
      const std::vector<std::string>& devices);

  //////////////////////////////////////////////////////////////////////////////
  // XLA dedicated operators follows here, listed in alphabetical order.
  //////////////////////////////////////////////////////////////////////////////
  static std::pair<XLATensor, torch::lazy::Value> all_reduce(
      const XLATensor& input, const torch::lazy::Value& token,
      AllReduceType reduce_type, double scale,
      std::vector<std::vector<int64_t>> groups, bool pin_layout);

  static torch::lazy::Value all_reduce_(
      XLATensor& input, const torch::lazy::Value& token,
      AllReduceType reduce_type, double scale,
      std::vector<std::vector<int64_t>> groups, bool pin_layout);

  static torch::lazy::Value all_reduce(std::vector<XLATensor>* inputs,
                                       const torch::lazy::Value& token,
                                       AllReduceType reduce_type, double scale,
                                       std::vector<std::vector<int64_t>> groups,
                                       bool pin_layout);

  static std::pair<XLATensor, torch::lazy::Value> reduce_scatter(
      const XLATensor& input, const torch::lazy::Value& token,
      AllReduceType reduce_type, double scale, int64_t scatter_dim,
      int64_t shard_count, std::vector<std::vector<int64_t>> groups,
      bool pin_layout);

  static torch::lazy::Value reduce_scatter_out(
      XLATensor& output, const XLATensor& input,
      const torch::lazy::Value& token, AllReduceType reduce_type, double scale,
      int64_t scatter_dim, int64_t shard_count,
      std::vector<std::vector<int64_t>> groups, bool pin_layout);

  static std::pair<XLATensor, torch::lazy::Value> all_to_all(
      const XLATensor& input, const torch::lazy::Value& token,
      int64_t split_dimension, int64_t concat_dimension, int64_t split_count,
      std::vector<std::vector<int64_t>> groups, bool pin_layout);

  static std::pair<XLATensor, torch::lazy::Value> all_gather(
      const XLATensor& input, const torch::lazy::Value& token, int64_t dim,
      int64_t shard_count, std::vector<std::vector<int64_t>> groups,
      bool pin_layout);

  static torch::lazy::Value all_gather_out(
      XLATensor& output, const XLATensor& input,
      const torch::lazy::Value& token, int64_t dim, int64_t shard_count,
      std::vector<std::vector<int64_t>> groups, bool pin_layout);

  static std::pair<XLATensor, torch::lazy::Value> collective_permute(
      const XLATensor& input, const torch::lazy::Value& token,
      std::vector<std::pair<int64_t, int64_t>> source_target_pairs);

  static XLATensor get_dimensions_size(const XLATensor& input,
                                       std::vector<int64_t> dimensions);

  static std::pair<XLATensor, torch::lazy::Value> recv(
      XLATensor& output, const torch::lazy::Value& token, int64_t channel_id);

  static std::pair<XLATensor, torch::lazy::Value> send(
      const XLATensor& input, const torch::lazy::Value& token,
      int64_t channel_id);

  static void sgd_optimizer_step_(const XLATensor& found_inf, XLATensor& step,
                                  XLATensor& param, XLATensor& buf,
                                  const XLATensor& d_p, double weight_decay,
                                  double momentum, double lr, double dampening,
                                  bool nesterov, bool maximize);

  static void adam_optimizer_step_(const XLATensor& found_inf, XLATensor& step,
                                   XLATensor& param, const XLATensor& grad,
                                   XLATensor& exp_avg, XLATensor& exp_avg_sq,
                                   XLATensor& max_exp_avg_sq, double beta1,
                                   double beta2, double lr, double weight_decay,
                                   double eps, bool amsgrad, bool maximize,
                                   bool use_adamw);

  static std::vector<XLATensor> user_computation(
      const std::string& opname, absl::Span<const XLATensor> inputs,
      ComputationPtr computation);

  //////////////////////////////////////////////////////////////////////////////
  // ATEN operators follows here, listed in alphabetical order.
  //////////////////////////////////////////////////////////////////////////////
  static void __ilshift__(XLATensor& input, const at::Scalar& other);
  static void __ilshift__(XLATensor& input, const XLATensor& other);

  static void __irshift__(XLATensor& input, const at::Scalar& other);
  static void __irshift__(XLATensor& input, const XLATensor& other);

  static XLATensor __lshift__(
      const XLATensor& input, const at::Scalar& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static XLATensor __lshift__(
      const XLATensor& input, const XLATensor& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static XLATensor __rshift__(
      const XLATensor& input, const at::Scalar& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static XLATensor __rshift__(
      const XLATensor& input, const XLATensor& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static std::tuple<XLATensor, XLATensor> adaptive_max_pool2d(
      const XLATensor& input, std::vector<int64_t> output_size);

  static XLATensor adaptive_max_pool2d_backward(const XLATensor& grad_output,
                                                const XLATensor& input);

  static XLATensor adaptive_avg_pool3d(const XLATensor& input,
                                       std::vector<int64_t> output_size);

  static XLATensor adaptive_avg_pool3d_backward(const XLATensor& grad_output,
                                                const XLATensor& input);

  static XLATensor _adaptive_avg_pool2d(const XLATensor& input,
                                        std::vector<int64_t> output_size);

  static XLATensor _adaptive_avg_pool2d_backward(const XLATensor& grad_output,
                                                 const XLATensor& input);

  static void _amp_foreach_non_finite_check_and_unscale_(
      std::vector<XLATensor> self, XLATensor& found_inf,
      const XLATensor& inv_scale);

  static void _amp_update_scale_(XLATensor& current_scale,
                                 XLATensor& growth_tracker,
                                 const XLATensor& found_inf,
                                 double scale_growth_factor,
                                 double scale_backoff_factor,
                                 int growth_interval);

  static XLATensor abs(const XLATensor& input);

  static XLATensor add(
      const XLATensor& input, const XLATensor& other, const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static XLATensor add(
      const XLATensor& input, const at::Scalar& other, const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static XLATensor addcdiv(const XLATensor& input, const at::Scalar& value,
                           const XLATensor& tensor1, const XLATensor& tensor2);
  static void addcdiv_(XLATensor& input, const at::Scalar& value,
                       const XLATensor& tensor1, const XLATensor& tensor2);

  static XLATensor addcmul(const XLATensor& input, const at::Scalar& value,
                           const XLATensor& tensor1, const XLATensor& tensor2);

  static XLATensor addmm(const XLATensor& input, const XLATensor& weight,
                         const XLATensor& bias);

  static XLATensor all(const XLATensor& input, std::vector<int64_t> dimensions,
                       bool keep_reduced_dimensions);

  static XLATensor amax(const XLATensor& input, std::vector<int64_t> dimensions,
                        bool keep_reduced_dimensions);

  static XLATensor amin(const XLATensor& input, std::vector<int64_t> dimensions,
                        bool keep_reduced_dimensions);

  static XLATensor any(const XLATensor& input, std::vector<int64_t> dimensions,
                       bool keep_reduced_dimensions);

  static void arange_out(XLATensor& out, const at::Scalar& start,
                         const at::Scalar& end, const at::Scalar& step,
                         at::ScalarType scalar_type);

  static XLATensor argmax(const XLATensor& input, int64_t dim, bool keepdim);
  static XLATensor argmax(const XLATensor& input);

  static XLATensor argmin(const XLATensor& input, int64_t dim, bool keepdim);
  static XLATensor argmin(const XLATensor& input);

  // Takes a slice from the input as R1 at the specified offset and reshapes it
  // into the provided size.
  static XLATensor as_strided(const XLATensor& input, std::vector<int64_t> size,
                              std::vector<int64_t> stride,
                              c10::optional<int64_t> storage_offset);

  // In-place version of the method above.
  static void as_strided_(XLATensor& input, std::vector<int64_t> size,
                          std::vector<int64_t> stride,
                          c10::optional<int64_t> storage_offset);

  static XLATensor atan2(
      const XLATensor& input, const XLATensor& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static XLATensor avg_pool_nd(const XLATensor& input,
                               int64_t spatial_dim_count,
                               std::vector<int64_t> kernel_size,
                               std::vector<int64_t> stride,
                               std::vector<int64_t> padding, bool ceil_mode,
                               bool count_include_pad);

  static XLATensor avg_pool_nd_backward(const XLATensor& out_backprop,
                                        const XLATensor& input,
                                        int64_t spatial_dim_count,
                                        std::vector<int64_t> kernel_size,
                                        std::vector<int64_t> stride,
                                        std::vector<int64_t> padding,
                                        bool ceil_mode, bool count_include_pad);

  static XLATensor baddbmm(const XLATensor& input, const XLATensor& batch1,
                           const XLATensor& batch2, const at::Scalar& beta,
                           const at::Scalar& alpha);

  static XLATensor bernoulli(const XLATensor& input, double probability);
  static XLATensor bernoulli(const XLATensor& input);
  static void bernoulli_(XLATensor& input, double probability);
  static void bernoulli_(XLATensor& input, const XLATensor& probability);

  static XLATensor binary_cross_entropy(const XLATensor& input,
                                        const XLATensor& target,
                                        const XLATensor& weight,
                                        int64_t reduction);

  static XLATensor binary_cross_entropy_backward(const XLATensor& grad_output,
                                                 const XLATensor& input,
                                                 const XLATensor& target,
                                                 const XLATensor& weight,
                                                 int64_t reduction);

  static XLATensor bitwise_and(const XLATensor& input, const at::Scalar& other);

  static XLATensor bitwise_and(const XLATensor& input, const XLATensor& other);

  static XLATensor bitwise_not(const XLATensor& input);

  static XLATensor bitwise_or(const XLATensor& input, const at::Scalar& other);

  static XLATensor bitwise_or(const XLATensor& input, const XLATensor& other);

  static XLATensor bitwise_xor(const XLATensor& input, const at::Scalar& other);

  static XLATensor bitwise_xor(const XLATensor& input, const XLATensor& other);

  // Batch matrix multiplication. Both tensors must be 3D, the batch size must
  // match and the remaining two dimensions must be compatible for matrix
  // multiplication.
  static XLATensor bmm(const XLATensor& batch1, const XLATensor& batch2);

  // Broadcasts the given tensors according to broadcasting semantics.
  static std::vector<XLATensor> broadcast_tensors(
      absl::Span<const XLATensor> tensors);

  static XLATensor cat(absl::Span<const XLATensor> tensors, int64_t dim,
                       at::ScalarType dtype);

  static XLATensor ceil(const XLATensor& input);

  static XLATensor celu(const XLATensor& input, const at::Scalar& alpha);
  static void celu_(XLATensor& input, const at::Scalar& alpha);

  static XLATensor cholesky(const XLATensor& input, bool upper);

  static XLATensor clamp(const XLATensor& input,
                         const c10::optional<at::Scalar>& min,
                         const c10::optional<at::Scalar>& max);
  static XLATensor clamp(const XLATensor& input,
                         const c10::optional<at::Tensor>& min,
                         const c10::optional<at::Tensor>& max);

  static XLATensor clone(const XLATensor& input);

  // Pad with the given value and size specified by the given list of low and
  // high paddings.
  static XLATensor constant_pad_nd(const XLATensor& input,
                                   absl::Span<const int64_t> pad,
                                   const at::Scalar& value);

  static XLATensor convolution_overrideable(
      const XLATensor& input, const XLATensor& weight, const XLATensor& bias,
      std::vector<int64_t> stride, std::vector<int64_t> padding,
      std::vector<int64_t> dilation, bool transposed,
      std::vector<int64_t> output_padding, int64_t groups);

  static std::tuple<XLATensor, XLATensor, XLATensor>
  convolution_backward_overrideable(
      const XLATensor& out_backprop, const XLATensor& input,
      const XLATensor& weight, std::vector<int64_t> stride,
      std::vector<int64_t> padding, std::vector<int64_t> dilation,
      bool transposed, std::vector<int64_t> output_padding, int64_t groups);

  static XLATensor convolution_overrideable(
      const XLATensor& input, const XLATensor& weight,
      std::vector<int64_t> stride, std::vector<int64_t> padding,
      std::vector<int64_t> dilation, bool transposed,
      std::vector<int64_t> output_padding, int64_t groups);

  // Returns the cross product of the two input tensors in the given dimension.
  // If the dimension is not given, it defaults to the first dimension found
  // with the size 3.
  static XLATensor cross(const XLATensor& input, const XLATensor& other,
                         c10::optional<int64_t> dim);

  // Returns the cumulative product of elements of input in the given dimension.
  static XLATensor cumprod(const XLATensor& input, int64_t dim,
                           c10::optional<at::ScalarType> dtype);

  // Returns the cumulative sum of elements of input in the given dimension.
  static XLATensor cumsum(const XLATensor& input, int64_t dim,
                          c10::optional<at::ScalarType> dtype);

  // If the input is a matrix (2-D tensor), returns a 1-D tensor with the
  // diagonal elements of the input. If the input is a vector (1-D tensor),
  // returns a 2-D square tensor with the elements of input as the diagonal.
  static XLATensor diag(const XLATensor& input, int64_t offset);

  // Returns the diagonal of a matrix (2-D tensor) or batch of matrices. The
  // matrix dimensions are specified by dim1 and dim2, the diagonal by offset.
  static XLATensor diagonal(const XLATensor& input, int64_t offset,
                            int64_t dim1, int64_t dim2);

  static XLATensor div(
      const XLATensor& input, const XLATensor& other,
      const c10::optional<c10::string_view>& rounding_mode = c10::nullopt,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static XLATensor div(const XLATensor& input, const at::Scalar& other);

  // A generalized contraction between tensors of arbitrary dimension defined by
  // the given equation and applied to the input tensors.
  static XLATensor einsum(const std::string& equation,
                          absl::Span<const XLATensor> tensors);

  static XLATensor elu(const XLATensor& input, const at::Scalar& alpha,
                       const at::Scalar& scale, const at::Scalar& input_scale);
  static void elu_(XLATensor& input, const at::Scalar& alpha,
                   const at::Scalar& scale, const at::Scalar& input_scale);
  static XLATensor elu_backward(const XLATensor& grad_output,
                                const at::Scalar& alpha,
                                const at::Scalar& scale,
                                const at::Scalar& input_scale,
                                const XLATensor& output);

  static XLATensor embedding_dense_backward(const XLATensor& grad_output,
                                            const XLATensor& indices,
                                            int64_t num_weights,
                                            int64_t padding_idx,
                                            bool scale_grad_by_freq);

  static XLATensor eq(const XLATensor& input, const at::Scalar& other);

  static XLATensor eq(const XLATensor& input, const XLATensor& other);

  static XLATensor erf(const XLATensor& input);

  static XLATensor erfc(const XLATensor& input);

  static XLATensor erfinv(const XLATensor& input);

  static XLATensor exp(const XLATensor& input);

  static XLATensor expand(const XLATensor& input, std::vector<int64_t> size);

  static XLATensor expm1(const XLATensor& input);

  static void exponential_(XLATensor& input, double lambd);

  // Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
  static XLATensor eye(int64_t lines, int64_t cols,
                       const torch::lazy::BackendDevice& device,
                       at::ScalarType element_type);

  static void eye_out(XLATensor& out, int64_t lines, int64_t cols);

  // Fills the input with the given value.
  static void fill_(XLATensor& input, const at::Scalar& value);

  // Flips (reverses) the values in the dimensions of the input tensor.
  static XLATensor flip(const XLATensor& input, absl::Span<const int64_t> dims);

  static XLATensor fmod(
      const XLATensor& input, const XLATensor& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static XLATensor fmod(
      const XLATensor& input, const at::Scalar& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static XLATensor frac(const XLATensor& input);

  static XLATensor full(absl::Span<const int64_t> size,
                        const at::Scalar& fill_value,
                        const torch::lazy::BackendDevice& device,
                        at::ScalarType scalar_type);
  static XLATensor full_like(const XLATensor& input,
                             const at::Scalar& fill_value,
                             const torch::lazy::BackendDevice& device,
                             c10::optional<at::ScalarType> scalar_type);

  static XLATensor gather(const XLATensor& input, int64_t dim,
                          const XLATensor& index);

  static XLATensor ge(const XLATensor& input, const at::Scalar& other);

  static XLATensor ge(const XLATensor& input, const XLATensor& other);

  static XLATensor gelu(const XLATensor& input,
                        const c10::string_view approximate);

  static XLATensor gelu_backward(const XLATensor& grad, const XLATensor& input,
                                 const c10::string_view approximate);

  static XLATensor ger(const XLATensor& input, const XLATensor& vec2);

  static XLATensor gt(const XLATensor& input, const at::Scalar& other);

  static XLATensor gt(const XLATensor& input, const XLATensor& other);

  // Gather slices from input into a result with shape specified by indices. The
  // shape of the indices are first made consistent using broadcast semantics.
  // For input of shape d1 x d2 x ... x dn and p indices of shape i1 x i2 x ...
  // x ik, the output shape is d1 x ... x d(start_dim) x i1 x ... x ik x
  // d(start_dim+p+1) x ... x dn.
  static XLATensor index(const XLATensor& input,
                         absl::Span<const XLATensor> indices,
                         int64_t start_dim);

  static XLATensor index_add(const XLATensor& input, int64_t dim,
                             const XLATensor& index, const XLATensor& source,
                             const at::Scalar& alpha);

  static XLATensor index_copy(const XLATensor& input, int64_t dim,
                              const XLATensor& index, const XLATensor& source);

  // Fills the elements of the base tensor with the given value in the given
  // dimension, at positions given by the index. The index must be a rank-1
  // tensor.
  static XLATensor index_fill(const XLATensor& input, int64_t dim,
                              const XLATensor& index, const at::Scalar& value);

  // Same as above, but the value is wrapped as a rank-0 tensor.
  static XLATensor index_fill(const XLATensor& input, int64_t dim,
                              const XLATensor& index, const XLATensor& value);

  static void index_fill_(XLATensor& input, int64_t dim, const XLATensor& index,
                          const XLATensor& value);

  static void index_fill_(XLATensor& input, int64_t dim, const XLATensor& index,
                          const at::Scalar& value);

  // Puts values into the input tensor using the given indices (a tuple of
  // tensors) and returns the result.
  static XLATensor index_put(const XLATensor& input,
                             absl::Span<const XLATensor> indices,
                             int64_t start_dim, const XLATensor& values,
                             bool accumulate,
                             absl::Span<const int64_t> result_permutation);

  static void index_put_(XLATensor& input, const XLATensor& canonical_base,
                         absl::Span<const XLATensor> indices, int64_t start_dim,
                         const XLATensor& values, bool accumulate,
                         absl::Span<const int64_t> result_permutation);

  static XLATensor index_select(const XLATensor& input, int64_t dim,
                                const XLATensor& index);

  static XLATensor isnan(const XLATensor& input);

  static XLATensor kl_div_backward(const XLATensor& grad_output,
                                   const XLATensor& input,
                                   const XLATensor& target, int64_t reduction,
                                   bool log_target);

  static std::tuple<XLATensor, XLATensor> kthvalue(const XLATensor& input,
                                                   int64_t k, int64_t dim,
                                                   bool keepdim);

  static XLATensor le(const XLATensor& input, const at::Scalar& other);

  static XLATensor le(const XLATensor& input, const XLATensor& other);

  static XLATensor hardshrink(const XLATensor& input, const at::Scalar& lambda);
  static XLATensor hardshrink_backward(const XLATensor& grad_out,
                                       const XLATensor& input,
                                       const at::Scalar& lambda);

  static XLATensor hardsigmoid(const XLATensor& input);

  static XLATensor hardsigmoid_backward(const XLATensor& grad_output,
                                        const XLATensor& input);

  static XLATensor hardswish(const XLATensor& input);

  static XLATensor hardswish_backward(const XLATensor& grad_output,
                                      const XLATensor& input);

  static XLATensor hardtanh_backward(const XLATensor& grad_output,
                                     const XLATensor& input,
                                     const at::Scalar& min_val,
                                     const at::Scalar& max_val);

  static XLATensor leaky_relu(const XLATensor& input, double negative_slope);
  static XLATensor leaky_relu_backward(const XLATensor& grad_output,
                                       const XLATensor& input,
                                       double negative_slope);

  static XLATensor lerp(const XLATensor& input, const XLATensor& end,
                        const XLATensor& weight);
  static XLATensor lerp(const XLATensor& input, const XLATensor& end,
                        const at::Scalar& weight);

  static XLATensor linspace(const at::Scalar& start, const at::Scalar& end,
                            const int64_t steps, at::ScalarType element_type,
                            const torch::lazy::BackendDevice& device);

  static XLATensor log(const XLATensor& input);

  static XLATensor log_base(const XLATensor& input, torch::lazy::OpKind op,
                            double base);

  static XLATensor log_sigmoid(const XLATensor& input);
  static std::tuple<XLATensor, XLATensor> log_sigmoid_forward(
      const XLATensor& input);
  static XLATensor log_sigmoid_backward(const XLATensor& grad_output,
                                        const XLATensor& input,
                                        const XLATensor& buffer);

  static XLATensor log_softmax(const XLATensor& input, int64_t dim,
                               c10::optional<at::ScalarType> dtype);

  static XLATensor log_softmax_backward(const XLATensor& grad_output,
                                        const XLATensor& output, int64_t dim);

  static XLATensor log1p(const XLATensor& input);
  static void log1p_(XLATensor& input);

  static XLATensor logical_not(const XLATensor& input);

  static XLATensor logical_xor(const XLATensor& input, const XLATensor& other);

  static XLATensor logical_and(const XLATensor& input, const XLATensor& other);

  static XLATensor logical_or(const XLATensor& input, const XLATensor& other);

  static XLATensor logsumexp(const XLATensor& input,
                             std::vector<int64_t> dimensions,
                             bool keep_reduced_dimensions);

  static XLATensor xlogy(const XLATensor& input, const XLATensor& other);

  static XLATensor lt(const XLATensor& input, const at::Scalar& other);

  static XLATensor lt(const XLATensor& input, const XLATensor& other);

  // In-place version of the method above.
  static void masked_fill_(XLATensor& input, const XLATensor& mask,
                           const at::Scalar& value);

  static void masked_scatter_(XLATensor& input, const XLATensor& mask,
                              const XLATensor& source);

  static XLATensor masked_select(const XLATensor& input, const XLATensor& mask);

  static XLATensor matmul(const XLATensor& input, const XLATensor& other);

  static XLATensor max(const XLATensor& input);

  static std::tuple<XLATensor, XLATensor> max(const XLATensor& input,
                                              int64_t dim, bool keepdim);

  static void max_out(XLATensor& max, XLATensor& max_values,
                      const XLATensor& input, int64_t dim, bool keepdim);

  static std::tuple<XLATensor, XLATensor> max_pool_nd(
      const XLATensor& input, int64_t spatial_dim_count,
      std::vector<int64_t> kernel_size, std::vector<int64_t> stride,
      std::vector<int64_t> padding, bool ceil_mode);

  static XLATensor max_pool_nd_backward(const XLATensor& out_backprop,
                                        const XLATensor& input,
                                        int64_t spatial_dim_count,
                                        std::vector<int64_t> kernel_size,
                                        std::vector<int64_t> stride,
                                        std::vector<int64_t> padding,
                                        bool ceil_mode);

  static XLATensor max_unpool(const XLATensor& input, const XLATensor& indices,
                              std::vector<int64_t> output_size);

  static XLATensor max_unpool_backward(const XLATensor& grad_output,
                                       const XLATensor& input,
                                       const XLATensor& indices,
                                       std::vector<int64_t> output_size);

  static XLATensor mean(const XLATensor& input, std::vector<int64_t> dimensions,
                        bool keep_reduced_dimensions,
                        c10::optional<at::ScalarType> dtype);

  static XLATensor min(
      const XLATensor& input, const XLATensor& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static XLATensor min(const XLATensor& input);

  static std::tuple<XLATensor, XLATensor> min(const XLATensor& input,
                                              int64_t dim, bool keepdim);

  static void min_out(XLATensor& min, XLATensor& min_indices,
                      const XLATensor& input, int64_t dim, bool keepdim);

  static XLATensor mish(const XLATensor& input);

  static XLATensor mm(const XLATensor& input, const XLATensor& weight);

  static XLATensor mse_loss(const XLATensor& input, const XLATensor& target,
                            int64_t reduction);

  static XLATensor mse_loss_backward(const XLATensor& grad_output,
                                     const XLATensor& input,
                                     const XLATensor& target,
                                     int64_t reduction);

  static XLATensor mul(
      const XLATensor& input, const XLATensor& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static XLATensor mul(
      const XLATensor& input, const at::Scalar& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static XLATensor mv(const XLATensor& input, const XLATensor& vec);
  static void mv_out(XLATensor& out, const XLATensor& input,
                     const XLATensor& vec);

  static XLATensor nan_to_num(const XLATensor& input, const at::Scalar& nan,
                              const at::Scalar& posinf,
                              const at::Scalar& neginf);

  // Returns a new tensor that is a narrowed view of the input in the given
  // dimension.
  static XLATensor narrow(const XLATensor& input, int64_t dim, int64_t start,
                          int64_t length);

  // Like batch_norm, but returns additional save_mean and save_invstd used by
  // the backward pass.
  static std::tuple<XLATensor, XLATensor, XLATensor> native_batch_norm(
      const XLATensor& input, const XLATensor& weight, const XLATensor& bias,
      XLATensor& running_mean, XLATensor& running_var, bool training,
      double momentum, double eps);

  // Returns the input, weight and bias gradients.
  static std::tuple<XLATensor, XLATensor, XLATensor> native_batch_norm_backward(
      const XLATensor& grad_out, const XLATensor& input,
      const XLATensor& weight, const XLATensor& save_mean,
      const XLATensor& save_invstd, bool training, double eps);

  static XLATensor ne(const XLATensor& input, const at::Scalar& other);

  static XLATensor ne(const XLATensor& input, const XLATensor& other);

  static XLATensor neg(const XLATensor& input);

  static XLATensor nll_loss(const XLATensor& input, const XLATensor& target,
                            const XLATensor& weight, int64_t reduction,
                            int ignore_index);

  static XLATensor nll_loss2d(const XLATensor& input, const XLATensor& target,
                              const XLATensor& weight, int64_t reduction,
                              int ignore_index);

  static XLATensor nll_loss2d_backward(const XLATensor& grad_output,
                                       const XLATensor& input,
                                       const XLATensor& target,
                                       const XLATensor& weight,
                                       int64_t reduction, int ignore_index,
                                       const XLATensor& total_weight);

  static XLATensor nll_loss_backward(const XLATensor& grad_output,
                                     const XLATensor& input,
                                     const XLATensor& target,
                                     const XLATensor& weight, int64_t reduction,
                                     int ignore_index,
                                     const XLATensor& total_weight);

  static std::pair<XLATensor, XLATensor> nms(const XLATensor& boxes,
                                             const XLATensor& scores,
                                             const XLATensor& score_threshold,
                                             const XLATensor& iou_threshold,
                                             int64_t output_size);

  static XLATensor nonzero(const XLATensor& input);

  static XLATensor norm(const XLATensor& input,
                        const c10::optional<at::Scalar>& p,
                        c10::optional<at::ScalarType> dtype,
                        at::IntArrayRef dim, bool keepdim);

  static XLATensor normal(double mean, const XLATensor& std);

  static XLATensor normal(const XLATensor& mean, double std);

  static XLATensor normal(const XLATensor& mean, const XLATensor& std);

  static void normal_(XLATensor& input, double mean, double std);

  static XLATensor not_supported(std::string description, xla::Shape shape,
                                 const torch::lazy::BackendDevice& device);

  static void optimization_barrier_(std::vector<XLATensor>& tensors);

  // Permute the dimensions of this tensor according to the given permutation.
  static XLATensor permute(const XLATensor& input,
                           absl::Span<const int64_t> dims);

  static XLATensor pow(const XLATensor& input, const at::Scalar& exponent);
  static XLATensor pow(const XLATensor& input, const XLATensor& exponent);
  static XLATensor pow(const at::Scalar& input, const XLATensor& exponent);

  static XLATensor prelu(const XLATensor& input, const XLATensor& weight);

  static XLATensor prod(const XLATensor& input, std::vector<int64_t> dimensions,
                        bool keep_reduced_dimensions,
                        c10::optional<at::ScalarType> dtype);

  static void put_(XLATensor& input, const XLATensor& index,
                   const XLATensor& source, bool accumulate);

  static std::tuple<XLATensor, XLATensor> qr(const XLATensor& input, bool some);

  static void random_(XLATensor& input, int64_t from, int64_t to);

  static XLATensor randperm(int64_t n, const torch::lazy::BackendDevice& device,
                            at::ScalarType scalar_type);

  static XLATensor reflection_pad2d(const XLATensor& input,
                                    std::vector<int64_t> padding);

  static XLATensor reflection_pad2d_backward(const XLATensor& grad_output,
                                             const XLATensor& input,
                                             std::vector<int64_t> padding);

  static XLATensor relu(const XLATensor& input);
  static void relu_(XLATensor& input);

  static XLATensor remainder(const XLATensor& input, const XLATensor& other);
  static XLATensor remainder(const XLATensor& input, const at::Scalar& other);

  // Repeats the input tensor along each dimension by the given number of
  // repeats.
  static XLATensor repeat(const XLATensor& input, std::vector<int64_t> repeats);

  static XLATensor replication_pad1d(const XLATensor& input,
                                     std::vector<int64_t> padding);
  static XLATensor replication_pad1d_backward(const XLATensor& grad_output,
                                              const XLATensor& input,
                                              std::vector<int64_t> padding);

  static XLATensor replication_pad2d(const XLATensor& input,
                                     std::vector<int64_t> padding);
  static XLATensor replication_pad2d_backward(const XLATensor& grad_output,
                                              const XLATensor& input,
                                              std::vector<int64_t> padding);

  static void resize_(XLATensor& input, std::vector<int64_t> size);

  static XLATensor roll(const XLATensor& input,
                        absl::Span<const int64_t> shifts,
                        absl::Span<const int64_t> dims);

  static XLATensor round(const XLATensor& input);

  static XLATensor rrelu_with_noise(const XLATensor& input, XLATensor& noise,
                                    const at::Scalar& lower,
                                    const at::Scalar& upper, bool training);

  static XLATensor rrelu_with_noise_backward(const XLATensor& grad_output,
                                             const XLATensor& input,
                                             const XLATensor& noise,
                                             const at::Scalar& lower,
                                             const at::Scalar& upper,
                                             bool training);

  static XLATensor rsqrt(const XLATensor& input);

  static XLATensor rsub(
      const XLATensor& input, const XLATensor& other, const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static XLATensor rsub(
      const XLATensor& input, const at::Scalar& other, const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static void copy_(XLATensor& input, XLATensor& src);

  static XLATensor scatter(const XLATensor& input, int64_t dim,
                           const XLATensor& index, const XLATensor& src);
  static XLATensor scatter(const XLATensor& input, int64_t dim,
                           const XLATensor& index, const at::Scalar& value);

  static XLATensor scatter_add(const XLATensor& input, int64_t dim,
                               const XLATensor& index, const XLATensor& src);
  static XLATensor scatter_add(const XLATensor& input, int64_t dim,
                               const XLATensor& index, const at::Scalar& value);

  static XLATensor select(const XLATensor& input, int64_t dim, int64_t index);

  static XLATensor selu(const XLATensor& input);
  static void selu_(XLATensor& input);

  static XLATensor silu(const XLATensor& input);
  static XLATensor silu_backward(XLATensor& grad_output, XLATensor& input);
  static XLATensor sigmoid(const XLATensor& input);
  static XLATensor sigmoid_backward(const XLATensor& grad_output,
                                    const XLATensor& output);

  static XLATensor slice(const XLATensor& input, int64_t dim, int64_t start,
                         int64_t end, int64_t step);

  static std::tuple<XLATensor, XLATensor> slogdet(const XLATensor& input);

  // Computes a loss that uses a squared term if the absolute element-wise error
  // falls below 1 and an L1 term otherwise.
  static XLATensor smooth_l1_loss(const XLATensor& input,
                                  const XLATensor& target, int64_t reduction,
                                  double beta);

  // Returns the gradient of the input of a smooth_l1_loss operation.
  static XLATensor smooth_l1_loss_backward(const XLATensor& grad_output,
                                           const XLATensor& input,
                                           const XLATensor& target,
                                           int64_t reduction, double beta);

  static XLATensor softmax(const XLATensor& input, int64_t dim,
                           c10::optional<at::ScalarType> dtype);
  static XLATensor softmax_backward(const XLATensor& grad_output,
                                    const XLATensor& output, int64_t dim);

  static XLATensor softplus(const XLATensor& input, const at::Scalar& beta,
                            const at::Scalar& threshold);

  static XLATensor softplus_backward(const XLATensor& grad_output,
                                     const XLATensor& input,
                                     const at::Scalar& beta,
                                     const at::Scalar& threshold);

  static XLATensor softshrink(const XLATensor& input, const at::Scalar& lambda);
  static XLATensor softshrink_backward(const XLATensor& grad_out,
                                       const XLATensor& input,
                                       const at::Scalar& lambda);

  static std::vector<XLATensor> split(const XLATensor& input,
                                      int64_t split_size, int64_t dim);

  static std::vector<XLATensor> split_with_sizes(
      const XLATensor& input, std::vector<int64_t> split_size, int64_t dim);

  static XLATensor sqrt(const XLATensor& input);

  // Squeeze out all trivial (size 1) dimensions.
  static XLATensor squeeze(const XLATensor& input);

  // Squeeze out the specified dimension index, if trivial (size 1). Returns
  // unchanged input otherwise.
  static XLATensor squeeze(const XLATensor& input, int64_t dim);

  // In-place versions of the methods above.
  static void squeeze_(XLATensor& input);
  static void squeeze_(XLATensor& input, int64_t dim);

  static XLATensor stack(absl::Span<const XLATensor> tensors, int64_t dim);

  static XLATensor std(const XLATensor& input, std::vector<int64_t> dimensions,
                       bool keep_reduced_dimensions, int64_t correction);

  static std::tuple<XLATensor, XLATensor> std_mean(
      const XLATensor& input, std::vector<int64_t> dimensions,
      int64_t correction, bool keep_reduced_dimensions);

  static XLATensor sub(
      const XLATensor& input, const XLATensor& other, const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static XLATensor sub(
      const XLATensor& input, const at::Scalar& other, const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static XLATensor sum(const XLATensor& input, std::vector<int64_t> dimensions,
                       bool keep_reduced_dimensions,
                       c10::optional<at::ScalarType> dtype);

  static std::tuple<XLATensor, XLATensor, XLATensor> svd(const XLATensor& input,
                                                         bool some,
                                                         bool compute_uv);

  static std::tuple<XLATensor, XLATensor> symeig(const XLATensor& input,
                                                 bool eigenvectors, bool upper);

  static XLATensor take(const XLATensor& input, const XLATensor& index);

  static XLATensor tanh(const XLATensor& input);

  static XLATensor tanh_backward(const XLATensor& grad_output,
                                 const XLATensor& output);

  static XLATensor threshold(const XLATensor& input, float threshold,
                             float value);

  static XLATensor threshold_backward(const XLATensor& grad_output,
                                      const XLATensor& input, float threshold);

  static XLATensor to(XLATensor& input,
                      c10::optional<torch::lazy::BackendDevice> device,
                      c10::optional<at::ScalarType> scalar_type);

  static std::tuple<XLATensor, XLATensor> topk(const XLATensor& input,
                                               int64_t k, int64_t dim,
                                               bool largest, bool sorted,
                                               bool stable);

  // Returns the sum of the elements of the diagonal of the input 2-D matrix.
  static XLATensor trace(const XLATensor& input);

  // Swap given dimensions of the input.
  static XLATensor transpose(const XLATensor& input, int64_t dim0,
                             int64_t dim1);

  // In-place version of the method above.
  static void transpose_(XLATensor& input, int64_t dim0, int64_t dim1);

  static std::tuple<XLATensor, XLATensor> triangular_solve(
      const XLATensor& rhs, const XLATensor& lhs, bool left_side, bool upper,
      bool transpose, bool unitriangular);

  // Returns the lower triangular part of a matrix (2-D tensor) or batch of
  // matrices input, the other elements of the result tensor out are set to 0.
  static XLATensor tril(const XLATensor& input, int64_t diagonal);

  // In-place version of the method above.
  static void tril_(XLATensor& input, int64_t diagonal);

  // Returns the upper triangular part of a matrix (2-D tensor) or batch of
  // matrices input, the other elements of the result tensor out are set to 0.
  static XLATensor triu(const XLATensor& input, int64_t diagonal);

  // In-place version of the method above.
  static void triu_(XLATensor& input, int64_t diagonal);

  static XLATensor trunc(const XLATensor& input);

  // Returns a tuple of all slices along a given dimension with that dimension
  // removed.
  static std::vector<XLATensor> unbind(const XLATensor& input, int64_t dim);

  static void uniform_(XLATensor& input, double from, double to);

  // Insert a dimension of size one at the specified position.
  static XLATensor unsqueeze(const XLATensor& input, int64_t dim);

  // In-place version of the method above.
  static void unsqueeze_(XLATensor& input, int64_t dim);

  static XLATensor upsample_bilinear2d(const XLATensor& input,
                                       std::vector<int64_t> output_size,
                                       bool align_corners);

  static XLATensor upsample_bilinear2d_backward(
      const XLATensor& grad_output, std::vector<int64_t> output_size,
      std::vector<int64_t> input_size, bool align_corners);

  static XLATensor upsample_nearest2d(const XLATensor& input,
                                      std::vector<int64_t> output_size);

  static XLATensor upsample_nearest2d_backward(const XLATensor& grad_output,
                                               std::vector<int64_t> output_size,
                                               std::vector<int64_t> input_size);

  static XLATensor var(const XLATensor& input, std::vector<int64_t> dimensions,
                       int64_t correction, bool keep_reduced_dimensions);

  static std::tuple<XLATensor, XLATensor> var_mean(
      const XLATensor& input, std::vector<int64_t> dimensions,
      int64_t correction, bool keep_reduced_dimensions);

  // Like reshape, but it returns a view into the original tensor.
  static XLATensor view(const XLATensor& input,
                        absl::Span<const int64_t> output_size);

  static void zero_(XLATensor& input);

  static XLATensor where(const XLATensor& condition, const XLATensor& input,
                         const XLATensor& other);

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

  static XLATensor Create(
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
  XLATensor CreateViewTensor(ViewInfo view_info) const;

  XLATensor CopyTensorToDevice(const torch::lazy::BackendDevice& device);

  torch::lazy::Value MaybeCastIrValue(
      torch::lazy::Value ir_value, const torch::lazy::BackendDevice& device,
      c10::optional<at::ScalarType> logical_element_type) const;

  // Create a new XLA tensor with the same metadata of the input tensor (with
  // possible overrides), and the new IR value.
  XLATensor CreateFrom(torch::lazy::Value ir_value) const;
  XLATensor CreateFrom(torch::lazy::Value ir_value,
                       const torch::lazy::BackendDevice& device) const;
  XLATensor CreateFrom(torch::lazy::Value ir_value,
                       at::ScalarType logical_element_type) const;
  XLATensor CreateFrom(
      torch::lazy::Value ir_value,
      c10::optional<at::ScalarType> logical_element_type_opt) const;
  XLATensor CreateFrom(torch::lazy::Value ir_value,
                       const torch::lazy::BackendDevice& device,
                       at::ScalarType logical_element_type) const;

  // We build an XLA graph accumulating XLA operations, but at a given point we
  // need to force a rendering, otherwise the graph can grow without control.
  // Think:
  //   for i in range(0, 100000):
  //     a = a + b
  void TryLimitGraphSize();

  std::vector<XLATensor> MakeOutputTensors(
      torch::lazy::NodePtr node, bool inherit_logical_type = true) const;

  torch::lazy::Value GetIrValueForTensor(
      const at::Tensor& tensor, const torch::lazy::BackendDevice& device) const;

  static ComputationCache* GetComputationCache();

  static SyncTensorCollection CollectSyncTensors(
      const std::vector<XLATensor>& tensors, const SyncTensorsConfig& config);

  // Waits for this SyncTensorCollection's device barrier and acuire the lock.
  static void TensorCollectionBarrier(SyncTensorCollection* coll);

  // Implementation of the GetTensors() API using the op-by-op executor.
  static std::vector<at::Tensor> GetTensorsOpByOp(
      std::vector<XLATensor>* tensors);

  static std::vector<at::Tensor> GetTensorsFused(
      std::vector<XLATensor>* tensors);

  // Runs an asynchronous syn operation using the op-by-op executor.
  using OpByOpAsync = xla::util::AsyncTask<int>;
  static OpByOpAsync SyncTensorsGraphOpByOp(
      std::vector<XLATensor>* tensors, absl::Span<const std::string> devices,
      const SyncTensorsConfig& config);

  // Gathers the XLA device data for all the input tensors, after an
  // asynchronous operation.
  static std::vector<torch::lazy::BackendDataPtr> GatherTensorsXlaData(
      const std::vector<XLATensor>& tensors, absl::Span<const size_t> indices,
      absl::Span<const torch::lazy::BackendDataPtr> tensors_data);

  static std::vector<torch::lazy::Value> CollectRoots(
      const std::vector<XLATensor>& tensors, absl::Span<const size_t> indices);

  static std::vector<torch::lazy::BackendDataPtr> FetchTensorData(
      std::vector<XLATensor>* tensors, const SyncTensorsConfig& config,
      absl::Span<const size_t> indices);

  static std::vector<at::Tensor> FetchTensors(
      std::vector<XLATensor>* tensors, absl::Span<const xla::Literal> literals,
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
      std::vector<XLATensor>* tensors, SyncTensorCollection* coll,
      std::vector<torch::lazy::BackendDataPtr> parameters_data,
      std::string device, ComputationCache::TypePtr cached_computation);

  static PostOrderData RunPostOrder(const std::vector<XLATensor>& tensors,
                                    SyncTensorCollection* coll);

  static ComputationCache::TypePtr LookupCachedCompile(
      const std::vector<XLATensor>& tensors, const torch::lazy::hash_t& hash);

  static std::shared_ptr<Async> TryRunCachedSync(
      std::vector<XLATensor>* tensors, SyncTensorCollection* coll,
      PostOrderData* po_data);

  static void BuildInputOutputAliases(const std::vector<XLATensor>& tensors,
                                      absl::Span<const size_t> indices,
                                      LoweringContext* lowering_ctx);

  static CompilationResult Compile(const std::vector<XLATensor>& tensors,
                                   absl::Span<const std::string> devices,
                                   const SyncTensorCollection& coll,
                                   PostOrderData* po_data);

  static std::shared_ptr<Async> SyncTensorsGraphInternal(
      std::vector<XLATensor>* tensors, absl::Span<const std::string> devices,
      const SyncTensorsConfig& config);

  static int64_t GetNextTensorId();

  static bool UseEagerDebugMode();

  bool ShouldSyncIrNode();

  std::shared_ptr<Data> data_;
};

using XLATensorPtr = c10::intrusive_ptr<XLATensor>;

}  // namespace torch_xla
