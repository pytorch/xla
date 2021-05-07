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
#include "torch_xla/csrc/computation.h"
#include "torch_xla/csrc/cross_replica_reduces.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ir_util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/view.h"

namespace torch_xla {

class XLATensor {
  class DeviceContextArena;
  struct Data;

 public:
  static XLATensor Create(const at::Tensor& tensor, const Device& device);
  static XLATensor Create(
      xla::ComputationClient::DataPtr xla_data,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static XLATensor Create(
      ir::Value ir_value, const Device& device,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  // Creates an empty/null tensor.
  XLATensor() = default;

  bool is_null() const { return data_ptr() == nullptr; }

  size_t generation() const { return data()->generation; }

  XLATensor alias() const { return XLATensor(data_ptr()); }

  xla::int64 size(xla::int64 dim) const;

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

  const Device& GetDevice() const;
  xla::int64 GetUniqueId() const;

  // Retrieves an opaque ID of the alias object upon which the tensor's view is
  // rooted, or 0 if this tensor is not a view.
  std::ptrdiff_t GetViewAliasId() const;

  // Fetches the XLA data behind the tensor. If the tensor has a graph defining
  // its current value, executes the graph and fetches the XLA data result.
  xla::ComputationClient::DataPtr GetXlaData();

  // Fetches the current value of the XLA data, which can be missing (nullptr)
  // in case the tensor has a graph defining its current value,
  xla::ComputationClient::DataPtr CurrentXlaData() const;

  void SetXlaData(xla::ComputationClient::DataPtr xla_data);

  // Retrieves the current IR Node, or nullptr in case no active IR Node is
  // available.
  ir::Value CurrentIrValue() const;

  // Retrieves the IR Node representing this XLATensor. One will be created if
  // missing. Note that although this is a const API, it actually changes the
  // internal state ofthe object.
  ir::Value GetIrValue() const;

  c10::optional<at::Tensor> CurrentTensorData() const;

  // Applies the queue of operations in preparation for using the data.
  void ApplyPendingGraph();

  static ir::Value GetDeviceDataIrValue(const at::Scalar& value,
                                        xla::PrimitiveType type,
                                        const Device& device);
  static ir::Value GetIrValueForScalar(const at::Scalar& value,
                                       xla::PrimitiveType type,
                                       const Device& device);
  static ir::Value GetIrValueForScalar(const at::Scalar& value,
                                       const Device& device);
  static ir::Value GetIrValueForScalar(const at::Scalar& value,
                                       xla::PrimitiveType type,
                                       absl::Span<const xla::int64> dimensions,
                                       const Device& device);
  static ir::Value GetIrValueForScalar(const at::Scalar& value,
                                       const xla::Shape& shape,
                                       const Device& device);
  static ir::Value GetIrValueForScalar(
      const at::Scalar& value, const xla::Shape& shape,
      c10::optional<at::ScalarType> logical_element_type, const Device& device);

  static ir::Value GetRngSeed(const Device& device);

  static void SetRngSeed(const Device& device, xla::uint64 seed);

  static xla::uint64 GetRunningSeed(const Device& device);

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
  static std::vector<XLATensor> GetLiveTensors(const Device* device);

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
  static void SyncLiveTensorsGraph(const Device* device,
                                   absl::Span<const std::string> devices,
                                   bool wait);

  // Marks an execution step, which allows the tensor framework to understand
  // the computation boundaries.
  static void MarkStep(const Device& device);

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
  static std::pair<XLATensor, ir::Value> all_reduce(
      const XLATensor& input, const ir::Value& token, AllReduceType reduce_type,
      double scale, std::vector<std::vector<xla::int64>> groups);

  static ir::Value all_reduce_(XLATensor& input, const ir::Value& token,
                               AllReduceType reduce_type, double scale,
                               std::vector<std::vector<xla::int64>> groups);

  static ir::Value all_reduce(std::vector<XLATensor>* inputs,
                              const ir::Value& token, AllReduceType reduce_type,
                              double scale,
                              std::vector<std::vector<xla::int64>> groups);

  static std::pair<XLATensor, ir::Value> all_to_all(
      const XLATensor& input, const ir::Value& token,
      xla::int64 split_dimension, xla::int64 concat_dimension,
      xla::int64 split_count, std::vector<std::vector<xla::int64>> groups);

  static std::pair<XLATensor, ir::Value> collective_permute(
      const XLATensor& input, const ir::Value& token,
      std::vector<std::pair<xla::int64, xla::int64>> source_target_pairs);

  static XLATensor get_dimensions_size(const XLATensor& input,
                                       std::vector<xla::int64> dimensions);

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

  static XLATensor adaptive_avg_pool3d(const XLATensor& input,
                                       std::vector<xla::int64> output_size);

  static XLATensor adaptive_avg_pool3d_backward(const XLATensor& grad_output,
                                                const XLATensor& input);

  static XLATensor _adaptive_avg_pool2d(const XLATensor& input,
                                        std::vector<xla::int64> output_size);

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
  static void abs_(XLATensor& input);

  static XLATensor acos(const XLATensor& input);
  static void acos_(XLATensor& input);

  static XLATensor acosh(const XLATensor& input);
  static void acosh_(XLATensor& input);

  static XLATensor add(
      const XLATensor& input, const XLATensor& other, const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static void add_(XLATensor& input, const XLATensor& other,
                   const at::Scalar& alpha);
  static XLATensor add(
      const XLATensor& input, const at::Scalar& other, const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static void add_(XLATensor& input, const at::Scalar& other,
                   const at::Scalar& alpha);

  static XLATensor addcdiv(const XLATensor& input, const at::Scalar& value,
                           const XLATensor& tensor1, const XLATensor& tensor2);
  static void addcdiv_(XLATensor& input, const at::Scalar& value,
                       const XLATensor& tensor1, const XLATensor& tensor2);

  static XLATensor addcmul(const XLATensor& input, const at::Scalar& value,
                           const XLATensor& tensor1, const XLATensor& tensor2);
  static void addcmul_(XLATensor& input, const at::Scalar& value,
                       const XLATensor& tensor1, const XLATensor& tensor2);

  static XLATensor addmm(const XLATensor& input, const XLATensor& weight,
                         const XLATensor& bias);

  static XLATensor all(const XLATensor& input,
                       std::vector<xla::int64> dimensions,
                       bool keep_reduced_dimensions);

  static XLATensor any(const XLATensor& input,
                       std::vector<xla::int64> dimensions,
                       bool keep_reduced_dimensions);

  static void arange_out(XLATensor& out, const at::Scalar& start,
                         const at::Scalar& end, const at::Scalar& step,
                         at::ScalarType scalar_type);

  static XLATensor argmax(const XLATensor& input, xla::int64 dim, bool keepdim);
  static XLATensor argmax(const XLATensor& input);

  static XLATensor argmin(const XLATensor& input, xla::int64 dim, bool keepdim);
  static XLATensor argmin(const XLATensor& input);

  // Takes a slice from the input as R1 at the specified offset and reshapes it
  // into the provided size.
  static XLATensor as_strided(const XLATensor& input,
                              std::vector<xla::int64> size,
                              std::vector<xla::int64> stride,
                              c10::optional<xla::int64> storage_offset);

  // In-place version of the method above.
  static void as_strided_(XLATensor& input, std::vector<xla::int64> size,
                          std::vector<xla::int64> stride,
                          c10::optional<xla::int64> storage_offset);

  static XLATensor asin(const XLATensor& input);
  static void asin_(XLATensor& input);

  static XLATensor asinh(const XLATensor& input);
  static void asinh_(XLATensor& input);

  static XLATensor atan(const XLATensor& input);
  static void atan_(XLATensor& input);

  static XLATensor atanh(const XLATensor& input);
  static void atanh_(XLATensor& input);

  static XLATensor atan2(
      const XLATensor& input, const XLATensor& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static void atan2_(XLATensor& input, const XLATensor& other);

  static XLATensor avg_pool_nd(const XLATensor& input,
                               xla::int64 spatial_dim_count,
                               std::vector<xla::int64> kernel_size,
                               std::vector<xla::int64> stride,
                               std::vector<xla::int64> padding, bool ceil_mode,
                               bool count_include_pad);

  static XLATensor avg_pool_nd_backward(const XLATensor& out_backprop,
                                        const XLATensor& input,
                                        xla::int64 spatial_dim_count,
                                        std::vector<xla::int64> kernel_size,
                                        std::vector<xla::int64> stride,
                                        std::vector<xla::int64> padding,
                                        bool ceil_mode, bool count_include_pad);

  static XLATensor baddbmm(const XLATensor& input, const XLATensor& batch1,
                           const XLATensor& batch2, const at::Scalar& beta,
                           const at::Scalar& alpha);
  static void baddbmm_(XLATensor& input, const XLATensor& batch1,
                       const XLATensor& batch2, const at::Scalar& beta,
                       const at::Scalar& alpha);

  static XLATensor bernoulli(const XLATensor& input, double probability);
  static XLATensor bernoulli(const XLATensor& input);
  static void bernoulli_(XLATensor& input, double probability);
  static void bernoulli_(XLATensor& input, const XLATensor& probability);

  static XLATensor binary_cross_entropy(const XLATensor& input,
                                        const XLATensor& target,
                                        const XLATensor& weight,
                                        xla::int64 reduction);

  static XLATensor binary_cross_entropy_backward(const XLATensor& grad_output,
                                                 const XLATensor& input,
                                                 const XLATensor& target,
                                                 const XLATensor& weight,
                                                 xla::int64 reduction);

  static void bitwise_and_out(XLATensor& out, const XLATensor& input,
                              const at::Scalar& other);

  static void bitwise_and_out(XLATensor& out, const XLATensor& input,
                              const XLATensor& other);

  static void bitwise_not_out(XLATensor& out, const XLATensor& input);

  static void bitwise_or_out(XLATensor& out, const XLATensor& input,
                             const at::Scalar& other);

  static void bitwise_or_out(XLATensor& out, const XLATensor& input,
                             const XLATensor& other);

  static void bitwise_xor_out(XLATensor& out, const XLATensor& input,
                              const at::Scalar& other);

  static void bitwise_xor_out(XLATensor& out, const XLATensor& input,
                              const XLATensor& other);

  // Batch matrix multiplication. Both tensors must be 3D, the batch size must
  // match and the remaining two dimensions must be compatible for matrix
  // multiplication.
  static XLATensor bmm(const XLATensor& batch1, const XLATensor& batch2);

  // Broadcasts the given tensors according to broadcasting semantics.
  static std::vector<XLATensor> broadcast_tensors(
      absl::Span<const XLATensor> tensors);

  static XLATensor cat(absl::Span<const XLATensor> tensors, xla::int64 dim);

  static XLATensor ceil(const XLATensor& input);
  static void ceil_(XLATensor& input);

  static XLATensor cholesky(const XLATensor& input, bool upper);

  static XLATensor clamp(const XLATensor& input,
                         const c10::optional<at::Scalar>& min,
                         const c10::optional<at::Scalar>& max);
  static XLATensor clamp(const XLATensor& input,
                         const c10::optional<at::Tensor>& min,
                         const c10::optional<at::Tensor>& max);
  static void clamp_(XLATensor& input, const c10::optional<at::Scalar>& min,
                     const c10::optional<at::Scalar>& max);
  static void clamp_out(XLATensor& out, const XLATensor& input,
                        const c10::optional<at::Tensor>& min,
                        const c10::optional<at::Tensor>& max);

  static XLATensor clone(const XLATensor& input);

  // Pad with the given value and size specified by the given list of low and
  // high paddings.
  static XLATensor constant_pad_nd(const XLATensor& input,
                                   absl::Span<const xla::int64> pad,
                                   const at::Scalar& value);

  static XLATensor convolution_overrideable(
      const XLATensor& input, const XLATensor& weight, const XLATensor& bias,
      std::vector<xla::int64> stride, std::vector<xla::int64> padding,
      std::vector<xla::int64> dilation, bool transposed,
      std::vector<xla::int64> output_padding, xla::int64 groups);

  static std::tuple<XLATensor, XLATensor, XLATensor>
  convolution_backward_overrideable(
      const XLATensor& out_backprop, const XLATensor& input,
      const XLATensor& weight, std::vector<xla::int64> stride,
      std::vector<xla::int64> padding, std::vector<xla::int64> dilation,
      bool transposed, std::vector<xla::int64> output_padding,
      xla::int64 groups);

  static XLATensor convolution_overrideable(
      const XLATensor& input, const XLATensor& weight,
      std::vector<xla::int64> stride, std::vector<xla::int64> padding,
      std::vector<xla::int64> dilation, bool transposed,
      std::vector<xla::int64> output_padding, xla::int64 groups);

  static XLATensor cos(const XLATensor& input);
  static void cos_(XLATensor& input);

  static XLATensor cosh(const XLATensor& input);
  static void cosh_(XLATensor& input);

  // Returns the cross product of the two input tensors in the given dimension.
  // If the dimension is not given, it defaults to the first dimension found
  // with the size 3.
  static XLATensor cross(const XLATensor& input, const XLATensor& other,
                         c10::optional<xla::int64> dim);

  // Returns the cumulative product of elements of input in the given dimension.
  static XLATensor cumprod(const XLATensor& input, xla::int64 dim,
                           c10::optional<at::ScalarType> dtype);

  // Returns the cumulative sum of elements of input in the given dimension.
  static XLATensor cumsum(const XLATensor& input, xla::int64 dim,
                          c10::optional<at::ScalarType> dtype);

  // If the input is a matrix (2-D tensor), returns a 1-D tensor with the
  // diagonal elements of the input. If the input is a vector (1-D tensor),
  // returns a 2-D square tensor with the elements of input as the diagonal.
  static XLATensor diag(const XLATensor& input, xla::int64 offset);

  // Returns the diagonal of a matrix (2-D tensor) or batch of matrices. The
  // matrix dimensions are specified by dim1 and dim2, the diagonal by offset.
  static XLATensor diagonal(const XLATensor& input, xla::int64 offset,
                            xla::int64 dim1, xla::int64 dim2);

  static XLATensor div(
      const XLATensor& input, const XLATensor& other,
      const c10::optional<std::string>& rounding_mode = c10::nullopt,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static XLATensor div(const XLATensor& input, const at::Scalar& other);
  static void div_(
      XLATensor& input, const XLATensor& other,
      const c10::optional<std::string>& rounding_mode = c10::nullopt);
  static void div_(XLATensor& input, const at::Scalar& other);

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
                                            xla::int64 num_weights,
                                            xla::int64 padding_idx,
                                            bool scale_grad_by_freq);

  static XLATensor eq(const XLATensor& input, const at::Scalar& other);
  static void eq_(XLATensor& input, const at::Scalar& other);

  static XLATensor eq(const XLATensor& input, const XLATensor& other);
  static void eq_(XLATensor& input, const XLATensor& other);

  static XLATensor erf(const XLATensor& input);
  static void erf_(XLATensor& input);

  static XLATensor erfc(const XLATensor& input);
  static void erfc_(XLATensor& input);

  static XLATensor erfinv(const XLATensor& input);
  static void erfinv_(XLATensor& input);

  static XLATensor exp(const XLATensor& input);
  static void exp_(XLATensor& input);

  static XLATensor expand(const XLATensor& input, std::vector<xla::int64> size);

  static XLATensor expm1(const XLATensor& input);
  static void expm1_(XLATensor& input);

  static void exponential_(XLATensor& input, double lambd);

  // Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
  static XLATensor eye(xla::int64 lines, xla::int64 cols, const Device& device,
                       at::ScalarType element_type);

  static void eye_out(XLATensor& out, xla::int64 lines, xla::int64 cols);

  // Fills the input with the given value.
  static void fill_(XLATensor& input, const at::Scalar& value);

  // Flips (reverses) the values in the dimensions of the input tensor.
  static XLATensor flip(const XLATensor& input,
                        absl::Span<const xla::int64> dims);

  static XLATensor floor(const XLATensor& input);
  static void floor_(XLATensor& input);

  static XLATensor fmod(
      const XLATensor& input, const XLATensor& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static XLATensor fmod(
      const XLATensor& input, const at::Scalar& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static void fmod_(XLATensor& input, const XLATensor& other);
  static void fmod_(XLATensor& input, const at::Scalar& other);

  static XLATensor frac(const XLATensor& input);
  static void frac_(XLATensor& input);

  static XLATensor full(absl::Span<const xla::int64> size,
                        const at::Scalar& fill_value, const Device& device,
                        at::ScalarType scalar_type);
  static XLATensor full_like(const XLATensor& input,
                             const at::Scalar& fill_value, const Device& device,
                             c10::optional<at::ScalarType> scalar_type);

  static XLATensor gather(const XLATensor& input, xla::int64 dim,
                          const XLATensor& index);

  static XLATensor ge(const XLATensor& input, const at::Scalar& other);
  static void ge_(XLATensor& input, const at::Scalar& other);

  static XLATensor ge(const XLATensor& input, const XLATensor& other);
  static void ge_(XLATensor& input, const XLATensor& other);

  static XLATensor gelu(const XLATensor& input);
  static XLATensor gelu_backward(const XLATensor& grad, const XLATensor& input);

  static XLATensor ger(const XLATensor& input, const XLATensor& vec2);

  static XLATensor gt(const XLATensor& input, const at::Scalar& other);
  static void gt_(XLATensor& input, const at::Scalar& other);

  static XLATensor gt(const XLATensor& input, const XLATensor& other);
  static void gt_(XLATensor& input, const XLATensor& other);

  // Gather slices from input into a result with shape specified by indices. The
  // shape of the indices are first made consistent using broadcast semantics.
  // For input of shape d1 x d2 x ... x dn and p indices of shape i1 x i2 x ...
  // x ik, the output shape is d1 x ... x d(start_dim) x i1 x ... x ik x
  // d(start_dim+p+1) x ... x dn.
  static XLATensor index(const XLATensor& input,
                         absl::Span<const XLATensor> indices,
                         xla::int64 start_dim);

  static XLATensor index_add(const XLATensor& input, xla::int64 dim,
                             const XLATensor& index, const XLATensor& source);

  static void index_add_(XLATensor& input, xla::int64 dim,
                         const XLATensor& index, const XLATensor& source);

  static XLATensor index_copy(const XLATensor& input, xla::int64 dim,
                              const XLATensor& index, const XLATensor& source);

  static void index_copy_(XLATensor& input, xla::int64 dim,
                          const XLATensor& index, const XLATensor& source);

  // Fills the elements of the base tensor with the given value in the given
  // dimension, at positions given by the index. The index must be a rank-1
  // tensor.
  static XLATensor index_fill(const XLATensor& input, xla::int64 dim,
                              const XLATensor& index, const at::Scalar& value);

  // Same as above, but the value is wrapped as a rank-0 tensor.
  static XLATensor index_fill(const XLATensor& input, xla::int64 dim,
                              const XLATensor& index, const XLATensor& value);

  static void index_fill_(XLATensor& input, xla::int64 dim,
                          const XLATensor& index, const XLATensor& value);

  static void index_fill_(XLATensor& input, xla::int64 dim,
                          const XLATensor& index, const at::Scalar& value);

  // Puts values into the input tensor using the given indices (a tuple of
  // tensors) and returns the result.
  static XLATensor index_put(const XLATensor& input,
                             absl::Span<const XLATensor> indices,
                             xla::int64 start_dim, const XLATensor& values,
                             bool accumulate,
                             absl::Span<const xla::int64> result_permutation);

  static void index_put_(XLATensor& input, const XLATensor& canonical_base,
                         absl::Span<const XLATensor> indices,
                         xla::int64 start_dim, const XLATensor& values,
                         bool accumulate,
                         absl::Span<const xla::int64> result_permutation);

  static XLATensor index_select(const XLATensor& input, xla::int64 dim,
                                const XLATensor& index);

  static XLATensor inverse(const XLATensor& input);

  static XLATensor kl_div_backward(const XLATensor& grad_output,
                                   const XLATensor& input,
                                   const XLATensor& target,
                                   xla::int64 reduction, bool log_target);

  static std::tuple<XLATensor, XLATensor> kthvalue(const XLATensor& input,
                                                   xla::int64 k, xla::int64 dim,
                                                   bool keepdim);

  static XLATensor l1_loss(const XLATensor& input, const XLATensor& target,
                           xla::int64 reduction);

  static XLATensor l1_loss_backward(const XLATensor& grad_output,
                                    const XLATensor& input,
                                    const XLATensor& target,
                                    xla::int64 reduction);

  static XLATensor le(const XLATensor& input, const at::Scalar& other);
  static void le_(XLATensor& input, const at::Scalar& other);

  static XLATensor le(const XLATensor& input, const XLATensor& other);
  static void le_(XLATensor& input, const XLATensor& other);

  static XLATensor hardshrink(const XLATensor& input, const at::Scalar& lambda);
  static XLATensor hardshrink_backward(const XLATensor& grad_out,
                                       const XLATensor& input,
                                       const at::Scalar& lambda);

  static XLATensor hardsigmoid(const XLATensor& input);

  static void hardsigmoid_(XLATensor& input);

  static XLATensor hardsigmoid_backward(const XLATensor& grad_output,
                                        const XLATensor& input);

  static XLATensor hardtanh_backward(const XLATensor& grad_output,
                                     const XLATensor& input,
                                     const at::Scalar& min_val,
                                     const at::Scalar& max_val);

  static XLATensor leaky_relu(const XLATensor& input, double negative_slope);
  static XLATensor leaky_relu_backward(const XLATensor& grad_output,
                                       const XLATensor& input,
                                       double negative_slope);
  static void leaky_relu_(XLATensor& input, double negative_slope);

  static XLATensor log(const XLATensor& input);
  static void log_(XLATensor& input);

  static XLATensor log_base(const XLATensor& input, ir::OpKind op, double base);
  static void log_base_(XLATensor& input, ir::OpKind op, double base);

  static XLATensor log_sigmoid(const XLATensor& input);
  static std::tuple<XLATensor, XLATensor> log_sigmoid_forward(
      const XLATensor& input);
  static XLATensor log_sigmoid_backward(const XLATensor& grad_output,
                                        const XLATensor& input,
                                        const XLATensor& buffer);

  static XLATensor log_softmax(const XLATensor& input, xla::int64 dim,
                               c10::optional<at::ScalarType> dtype);

  static XLATensor log_softmax_backward(const XLATensor& grad_output,
                                        const XLATensor& output,
                                        xla::int64 dim);

  static XLATensor log1p(const XLATensor& input);
  static void log1p_(XLATensor& input);

  static XLATensor logdet(const XLATensor& input);

  static XLATensor logsumexp(const XLATensor& input,
                             std::vector<xla::int64> dimensions,
                             bool keep_reduced_dimensions);

  static XLATensor lt(const XLATensor& input, const at::Scalar& other);
  static void lt_(XLATensor& input, const at::Scalar& other);

  static XLATensor lt(const XLATensor& input, const XLATensor& other);
  static void lt_(XLATensor& input, const XLATensor& other);

  // In-place version of the method above.
  static void masked_fill_(XLATensor& input, const XLATensor& mask,
                           const at::Scalar& value);

  static void masked_scatter_(XLATensor& input, const XLATensor& mask,
                              const XLATensor& source);

  static XLATensor masked_select(const XLATensor& input, const XLATensor& mask);

  static XLATensor matmul(const XLATensor& input, const XLATensor& other);

  static XLATensor max(
      const XLATensor& input, const XLATensor& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static XLATensor max(const XLATensor& input);

  static std::tuple<XLATensor, XLATensor> max(const XLATensor& input,
                                              xla::int64 dim, bool keepdim);

  static void max_out(XLATensor& max, XLATensor& max_values,
                      const XLATensor& input, xla::int64 dim, bool keepdim);

  static std::tuple<XLATensor, XLATensor> max_pool_nd(
      const XLATensor& input, xla::int64 spatial_dim_count,
      std::vector<xla::int64> kernel_size, std::vector<xla::int64> stride,
      std::vector<xla::int64> padding, bool ceil_mode);

  static XLATensor max_pool_nd_backward(const XLATensor& out_backprop,
                                        const XLATensor& input,
                                        xla::int64 spatial_dim_count,
                                        std::vector<xla::int64> kernel_size,
                                        std::vector<xla::int64> stride,
                                        std::vector<xla::int64> padding,
                                        bool ceil_mode);

  static XLATensor max_unpool(const XLATensor& input, const XLATensor& indices,
                              std::vector<xla::int64> output_size);

  static XLATensor max_unpool_backward(const XLATensor& grad_output,
                                       const XLATensor& input,
                                       const XLATensor& indices,
                                       std::vector<xla::int64> output_size);

  static XLATensor mean(const XLATensor& input,
                        std::vector<xla::int64> dimensions,
                        bool keep_reduced_dimensions,
                        c10::optional<at::ScalarType> dtype);

  static XLATensor min(
      const XLATensor& input, const XLATensor& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static XLATensor min(const XLATensor& input);

  static std::tuple<XLATensor, XLATensor> min(const XLATensor& input,
                                              xla::int64 dim, bool keepdim);

  static void min_out(XLATensor& min, XLATensor& min_indices,
                      const XLATensor& input, xla::int64 dim, bool keepdim);

  static XLATensor mm(const XLATensor& input, const XLATensor& weight);

  static XLATensor mse_loss(const XLATensor& input, const XLATensor& target,
                            xla::int64 reduction);

  static XLATensor mse_loss_backward(const XLATensor& grad_output,
                                     const XLATensor& input,
                                     const XLATensor& target,
                                     xla::int64 reduction);

  static XLATensor mul(
      const XLATensor& input, const XLATensor& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static XLATensor mul(
      const XLATensor& input, const at::Scalar& other,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static void mul_(XLATensor& input, const XLATensor& other);
  static void mul_(XLATensor& input, const at::Scalar& other);

  static XLATensor mv(const XLATensor& input, const XLATensor& vec);
  static void mv_out(XLATensor& out, const XLATensor& input,
                     const XLATensor& vec);

  // Returns a new tensor that is a narrowed view of the input in the given
  // dimension.
  static XLATensor narrow(const XLATensor& input, xla::int64 dim,
                          xla::int64 start, xla::int64 length);

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
  static void ne_(XLATensor& input, const at::Scalar& other);

  static XLATensor ne(const XLATensor& input, const XLATensor& other);
  static void ne_(XLATensor& input, const XLATensor& other);

  static XLATensor neg(const XLATensor& input);
  static void neg_(XLATensor& input);

  static XLATensor nll_loss(const XLATensor& input, const XLATensor& target,
                            const XLATensor& weight, xla::int64 reduction,
                            int ignore_index);

  static XLATensor nll_loss2d(const XLATensor& input, const XLATensor& target,
                              const XLATensor& weight, xla::int64 reduction,
                              int ignore_index);

  static XLATensor nll_loss2d_backward(const XLATensor& grad_output,
                                       const XLATensor& input,
                                       const XLATensor& target,
                                       const XLATensor& weight,
                                       xla::int64 reduction, int ignore_index,
                                       const XLATensor& total_weight);

  static XLATensor nll_loss_backward(const XLATensor& grad_output,
                                     const XLATensor& input,
                                     const XLATensor& target,
                                     const XLATensor& weight,
                                     xla::int64 reduction, int ignore_index,
                                     const XLATensor& total_weight);

  static std::pair<XLATensor, XLATensor> nms(const XLATensor& boxes,
                                             const XLATensor& scores,
                                             const XLATensor& score_threshold,
                                             const XLATensor& iou_threshold,
                                             xla::int64 output_size);

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
                                 const Device& device);

  // Permute the dimensions of this tensor according to the given permutation.
  static XLATensor permute(const XLATensor& input,
                           absl::Span<const xla::int64> dims);

  static XLATensor pow(const XLATensor& input, const at::Scalar& exponent);
  static XLATensor pow(const XLATensor& input, const XLATensor& exponent);
  static XLATensor pow(const at::Scalar& input, const XLATensor& exponent);
  static void pow_(XLATensor& input, const at::Scalar& exponent);
  static void pow_(XLATensor& input, const XLATensor& exponent);

  static XLATensor prod(const XLATensor& input,
                        std::vector<xla::int64> dimensions,
                        bool keep_reduced_dimensions,
                        c10::optional<at::ScalarType> dtype);

  static void put_(XLATensor& input, const XLATensor& index,
                   const XLATensor& source, bool accumulate);

  static std::tuple<XLATensor, XLATensor> qr(const XLATensor& input, bool some);

  static void random_(XLATensor& input, int64_t from, int64_t to);

  static XLATensor randperm(xla::int64 n, const Device& device,
                            at::ScalarType scalar_type);

  static XLATensor reciprocal(const XLATensor& input);
  static void reciprocal_(XLATensor& input);

  static XLATensor reflection_pad2d(const XLATensor& input,
                                    std::vector<xla::int64> padding);

  static XLATensor reflection_pad2d_backward(const XLATensor& grad_output,
                                             const XLATensor& input,
                                             std::vector<xla::int64> padding);

  static XLATensor relu(const XLATensor& input);
  static void relu_(XLATensor& input);

  static XLATensor remainder(const XLATensor& input, const XLATensor& other);
  static XLATensor remainder(const XLATensor& input, const at::Scalar& other);
  static void remainder_(XLATensor& input, const XLATensor& other);
  static void remainder_(XLATensor& input, const at::Scalar& other);

  // Repeats the input tensor along each dimension by the given number of
  // repeats.
  static XLATensor repeat(const XLATensor& input,
                          std::vector<xla::int64> repeats);

  static XLATensor replication_pad1d(const XLATensor& input,
                                     std::vector<xla::int64> padding);
  static XLATensor replication_pad1d_backward(const XLATensor& grad_output,
                                              const XLATensor& input,
                                              std::vector<xla::int64> padding);

  static XLATensor replication_pad2d(const XLATensor& input,
                                     std::vector<xla::int64> padding);
  static XLATensor replication_pad2d_backward(const XLATensor& grad_output,
                                              const XLATensor& input,
                                              std::vector<xla::int64> padding);

  static void resize_(XLATensor& input, std::vector<xla::int64> size);

  static XLATensor round(const XLATensor& input);
  static void round_(XLATensor& input);

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
  static void rsqrt_(XLATensor& input);

  static XLATensor rsub(
      const XLATensor& input, const XLATensor& other, const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static XLATensor rsub(
      const XLATensor& input, const at::Scalar& other, const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static void copy_(XLATensor& input, XLATensor& src);

  static void scatter_(XLATensor& input, xla::int64 dim, const XLATensor& index,
                       const XLATensor& src);
  static void scatter_(XLATensor& input, xla::int64 dim, const XLATensor& index,
                       const at::Scalar& value);

  static void scatter_add_(XLATensor& input, xla::int64 dim,
                           const XLATensor& index, const XLATensor& src);

  static XLATensor select(const XLATensor& input, xla::int64 dim,
                          xla::int64 index);

  static void silu_out(XLATensor& input, XLATensor& out);
  static XLATensor sigmoid(const XLATensor& input);
  static void sigmoid_(XLATensor& input);
  static XLATensor sigmoid_backward(const XLATensor& grad_output,
                                    const XLATensor& output);

  static XLATensor sign(const XLATensor& input);
  static void sign_(XLATensor& input);

  static XLATensor sin(const XLATensor& input);
  static void sin_(XLATensor& input);

  static XLATensor sinh(const XLATensor& input);
  static void sinh_(XLATensor& input);

  static XLATensor slice(const XLATensor& input, xla::int64 dim,
                         xla::int64 start, xla::int64 end, xla::int64 step);

  // Computes a loss that uses a squared term if the absolute element-wise error
  // falls below 1 and an L1 term otherwise.
  static XLATensor smooth_l1_loss(const XLATensor& input,
                                  const XLATensor& target, xla::int64 reduction,
                                  double beta);

  // Returns the gradient of the input of a smooth_l1_loss operation.
  static XLATensor smooth_l1_loss_backward(const XLATensor& grad_output,
                                           const XLATensor& input,
                                           const XLATensor& target,
                                           xla::int64 reduction, double beta);

  static XLATensor softmax(const XLATensor& input, xla::int64 dim,
                           c10::optional<at::ScalarType> dtype);
  static XLATensor softmax_backward(const XLATensor& grad_output,
                                    const XLATensor& output, xla::int64 dim);

  static XLATensor softplus(const XLATensor& input, const at::Scalar& beta,
                            const at::Scalar& threshold);
  static XLATensor softplus_backward(const XLATensor& grad_output,
                                     const XLATensor& input,
                                     const at::Scalar& beta,
                                     const at::Scalar& threshold,
                                     const XLATensor& output);

  static XLATensor softshrink(const XLATensor& input, const at::Scalar& lambda);
  static XLATensor softshrink_backward(const XLATensor& grad_out,
                                       const XLATensor& input,
                                       const at::Scalar& lambda);

  static std::vector<XLATensor> split(const XLATensor& input,
                                      xla::int64 split_size, xla::int64 dim);

  static std::vector<XLATensor> split_with_sizes(
      const XLATensor& input, std::vector<xla::int64> split_size,
      xla::int64 dim);

  static XLATensor sqrt(const XLATensor& input);
  static void sqrt_(XLATensor& input);

  // Squeeze out all trivial (size 1) dimensions.
  static XLATensor squeeze(const XLATensor& input);

  // Squeeze out the specified dimension index, if trivial (size 1). Returns
  // unchanged input otherwise.
  static XLATensor squeeze(const XLATensor& input, xla::int64 dim);

  // In-place versions of the methods above.
  static void squeeze_(XLATensor& input);
  static void squeeze_(XLATensor& input, xla::int64 dim);

  static XLATensor stack(absl::Span<const XLATensor> tensors, xla::int64 dim);

  static XLATensor std(const XLATensor& input,
                       std::vector<xla::int64> dimensions,
                       bool keep_reduced_dimensions, xla::int64 correction);

  static XLATensor sub(
      const XLATensor& input, const XLATensor& other, const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static void sub_(XLATensor& input, const XLATensor& other,
                   const at::Scalar& alpha);
  static XLATensor sub(
      const XLATensor& input, const at::Scalar& other, const at::Scalar& alpha,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  static void sub_(XLATensor& input, const at::Scalar& other,
                   const at::Scalar& alpha);

  static XLATensor sum(const XLATensor& input,
                       std::vector<xla::int64> dimensions,
                       bool keep_reduced_dimensions,
                       c10::optional<at::ScalarType> dtype);

  static std::tuple<XLATensor, XLATensor, XLATensor> svd(const XLATensor& input,
                                                         bool some,
                                                         bool compute_uv);

  static std::tuple<XLATensor, XLATensor> symeig(const XLATensor& input,
                                                 bool eigenvectors, bool upper);

  static XLATensor take(const XLATensor& input, const XLATensor& index);

  static XLATensor tan(const XLATensor& input);
  static void tan_(XLATensor& input);

  static XLATensor tanh(const XLATensor& input);
  static void tanh_(XLATensor& input);
  static XLATensor tanh_backward(const XLATensor& grad_output,
                                 const XLATensor& output);

  static XLATensor threshold(const XLATensor& input, float threshold,
                             float value);
  static void threshold_(XLATensor& input, float threshold, float value);

  static XLATensor threshold_backward(const XLATensor& grad_output,
                                      const XLATensor& input, float threshold);

  static XLATensor to(XLATensor& input, c10::optional<Device> device,
                      c10::optional<at::ScalarType> scalar_type);

  static std::tuple<XLATensor, XLATensor> topk(const XLATensor& input,
                                               xla::int64 k, xla::int64 dim,
                                               bool largest, bool sorted);

  // Returns the sum of the elements of the diagonal of the input 2-D matrix.
  static XLATensor trace(const XLATensor& input);

  // Swap given dimensions of the input.
  static XLATensor transpose(const XLATensor& input, xla::int64 dim0,
                             xla::int64 dim1);

  // In-place version of the method above.
  static void transpose_(XLATensor& input, xla::int64 dim0, xla::int64 dim1);

  static std::tuple<XLATensor, XLATensor> triangular_solve(
      const XLATensor& rhs, const XLATensor& lhs, bool left_side, bool upper,
      bool transpose, bool unitriangular);

  // Returns the lower triangular part of a matrix (2-D tensor) or batch of
  // matrices input, the other elements of the result tensor out are set to 0.
  static XLATensor tril(const XLATensor& input, xla::int64 diagonal);

  // In-place version of the method above.
  static void tril_(XLATensor& input, xla::int64 diagonal);

  // Returns the upper triangular part of a matrix (2-D tensor) or batch of
  // matrices input, the other elements of the result tensor out are set to 0.
  static XLATensor triu(const XLATensor& input, xla::int64 diagonal);

  // In-place version of the method above.
  static void triu_(XLATensor& input, xla::int64 diagonal);

  static XLATensor trunc(const XLATensor& input);
  static void trunc_(XLATensor& input);

  // Returns a tuple of all slices along a given dimension with that dimension
  // removed.
  static std::vector<XLATensor> unbind(const XLATensor& input, xla::int64 dim);

  static void uniform_(XLATensor& input, double from, double to);

  // Insert a dimension of size one at the specified position.
  static XLATensor unsqueeze(const XLATensor& input, xla::int64 dim);

  // In-place version of the method above.
  static void unsqueeze_(XLATensor& input, xla::int64 dim);

  static XLATensor upsample_bilinear2d(const XLATensor& input,
                                       std::vector<xla::int64> output_size,
                                       bool align_corners);

  static XLATensor upsample_bilinear2d_backward(
      const XLATensor& grad_output, std::vector<xla::int64> output_size,
      std::vector<xla::int64> input_size, bool align_corners);

  static XLATensor upsample_nearest2d(const XLATensor& input,
                                      std::vector<xla::int64> output_size);

  static XLATensor upsample_nearest2d_backward(
      const XLATensor& grad_output, std::vector<xla::int64> output_size,
      std::vector<xla::int64> input_size);

  static XLATensor var(const XLATensor& input,
                       std::vector<xla::int64> dimensions,
                       xla::int64 correction, bool keep_reduced_dimensions);

  // Like reshape, but it returns a view into the original tensor.
  static XLATensor view(const XLATensor& input,
                        absl::Span<const xla::int64> output_size);

  static void zero_(XLATensor& input);

  static XLATensor where(const XLATensor& condition, const XLATensor& input,
                         const XLATensor& other);

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
    xla::hash_t hash;
    std::vector<xla::util::ExceptionCleanup> unlocker;
    Device device;
  };

  struct PostOrderData {
    std::vector<const ir::Node*> post_order;
    ir::Util::EmissionMap emission_map;
    std::vector<xla::ComputationClient::DataPtr> parameters_data;
    std::vector<size_t> parameter_sequence;
  };

  struct CompilationResult {
    Device device;
    size_t emitted_nodes = 0;
    std::shared_ptr<xla::ComputationClient::Computation> computation;
    std::vector<xla::ComputationClient::DataPtr> parameters_data;
  };

  struct CachedComputation {
    CachedComputation(
        std::shared_ptr<xla::ComputationClient::Computation> computation)
        : computation(std::move(computation)) {}

    std::shared_ptr<xla::ComputationClient::Computation> computation;
  };

  using ComputationCache =
      xla::util::Cache<xla::hash_t, CachedComputation, xla::util::HashReducer>;

  struct Async {
    Async(SyncTensorCollection* coll,
          std::vector<xla::ComputationClient::DataPtr> parameters_data,
          std::vector<xla::ComputationClient::DataPtr> tensors_data,
          ComputationCache::TypePtr cached_computation);

    void Wait();

    xla::util::MultiWait mwait;
    std::vector<size_t> indices;
    std::vector<xla::util::ExceptionCleanup> unlocker;
    std::vector<xla::ComputationClient::DataPtr> parameters_data;
    std::string device;
    ComputationCache::TypePtr cached_computation;
    std::vector<xla::ComputationClient::DataPtr> tensors_data;
  };

  // This is the core XLA tensor data structure where all the tensor data is
  // held. The XLA tensor is nothing more than a shared pointer to a Data
  // object.
  struct Data {
    Data(xla::ComputationClient::DataPtr xla_data, const Device& device,
         c10::optional<at::ScalarType> logical_element_type)
        : xla_data(std::move(xla_data)),
          logical_element_type(logical_element_type),
          device(device),
          unique_id(GetNextTensorId()) {}
    Data(ir::Value ir_value, const Device& device,
         c10::optional<at::ScalarType> logical_element_type)
        : ir_value(std::move(ir_value)),
          logical_element_type(logical_element_type),
          device(device),
          unique_id(GetNextTensorId()) {}
    Data(std::shared_ptr<View> view, const Device& device,
         c10::optional<at::ScalarType> logical_element_type)
        : view(std::move(view)),
          logical_element_type(logical_element_type),
          device(device),
          unique_id(GetNextTensorId()) {}
    Data(at::Tensor tensor_data, const Device& device)
        : logical_element_type(tensor_data.scalar_type()),
          tensor_data(std::move(tensor_data)),
          device(device),
          unique_id(GetNextTensorId()) {}

    ~Data();

    xla::ComputationClient::DataPtr xla_data;
    ir::Value ir_value;
    std::shared_ptr<View> view;
    c10::optional<at::ScalarType> logical_element_type;
    c10::optional<at::Tensor> tensor_data;
    const Device device;
    const xla::int64 unique_id = 0;
    size_t generation = 1;
  };

  XLATensor(const at::Tensor& tensor, const Device& device);
  XLATensor(xla::ComputationClient::DataPtr xla_data,
            c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  XLATensor(ir::Value ir_value, const Device& device,
            c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  XLATensor(std::shared_ptr<View> view, const Device& device,
            c10::optional<at::ScalarType> logical_element_type = c10::nullopt);
  XLATensor(std::shared_ptr<Data> data);

  static XLATensor Create(
      std::shared_ptr<View> view, const Device& device,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  Data* data() const;

  std::shared_ptr<Data> data_ptr() const { return data_; }

  void SetXlaData(xla::ComputationClient::DataPtr xla_data, bool sync);

  void SetIrValue(ir::Value ir_value);
  void SetInPlaceIrValue(ir::Value ir_value);

  void AssignIrValue(ir::Value ir_value) const;

  void SetTensorData(at::Tensor tensor_data);

  ir::Value CreateTensorNode(xla::ComputationClient::DataPtr data,
                             bool read_only) const;

  View::IrNode GetViewUpdate(const std::shared_ptr<View>& view) const;

  std::shared_ptr<View> UpdateView(std::shared_ptr<View> view,
                                   ir::Value ir_value) const;

  void SetSubView(ViewInfo view_info) const;
  std::shared_ptr<View> CreateView(ViewInfo view_info) const;
  XLATensor CreateViewTensor(ViewInfo view_info) const;

  XLATensor CopyTensorToDevice(const Device& device);

  ir::Value MaybeCastIrValue(
      ir::Value ir_value, const Device& device,
      c10::optional<at::ScalarType> logical_element_type) const;

  // Create a new XLA tensor with the same metadata of the input tensor (with
  // possible overrides), and the new IR value.
  XLATensor CreateFrom(ir::Value ir_value) const;
  XLATensor CreateFrom(ir::Value ir_value, const Device& device) const;
  XLATensor CreateFrom(ir::Value ir_value,
                       at::ScalarType logical_element_type) const;
  XLATensor CreateFrom(
      ir::Value ir_value,
      c10::optional<at::ScalarType> logical_element_type_opt) const;
  XLATensor CreateFrom(ir::Value ir_value, const Device& device,
                       at::ScalarType logical_element_type) const;

  // We build an XLA graph accumulating XLA operations, but at a given point we
  // need to force a rendering, otherwise the graph can grow without control.
  // Think:
  //   for i in range(0, 100000):
  //     a = a + b
  void TryLimitGraphSize();

  std::vector<XLATensor> MakeOutputTensors(ir::NodePtr node) const;

  ir::Value GetIrValueForTensor(const at::Tensor& tensor,
                                const Device& device) const;

  static ComputationCache* GetComputationCache();

  static SyncTensorCollection CollectSyncTensors(
      const std::vector<XLATensor>& tensors, const SyncTensorsConfig& config);

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
  static std::vector<xla::ComputationClient::DataPtr> GatherTensorsXlaData(
      const std::vector<XLATensor>& tensors, absl::Span<const size_t> indices,
      absl::Span<const xla::ComputationClient::DataPtr> tensors_data);

  static std::vector<ir::Value> CollectRoots(
      const std::vector<XLATensor>& tensors, absl::Span<const size_t> indices);

  static std::vector<xla::ComputationClient::DataPtr> FetchTensorData(
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
      std::vector<xla::ComputationClient::DataPtr> parameters_data,
      std::vector<xla::ComputationClient::DataPtr> tensors_data,
      ComputationCache::TypePtr cached_computation);

  static std::shared_ptr<Async> ScheduleSyncTensorsGraph(
      std::vector<XLATensor>* tensors, SyncTensorCollection* coll,
      std::vector<xla::ComputationClient::DataPtr> parameters_data,
      std::string device, ComputationCache::TypePtr cached_computation);

  static PostOrderData RunPostOrder(const std::vector<XLATensor>& tensors,
                                    absl::Span<const size_t> indices);

  static ComputationCache::TypePtr LookupCachedCompile(
      const std::vector<XLATensor>& tensors, const xla::hash_t& hash);

  static std::shared_ptr<Async> TryRunCachedSync(
      std::vector<XLATensor>* tensors, SyncTensorCollection* coll,
      PostOrderData* po_data);

  static void BuildInputOutputAliases(const std::vector<XLATensor>& tensors,
                                      absl::Span<const size_t> indices,
                                      ir::LoweringContext* lowering_ctx);

  static CompilationResult Compile(const std::vector<XLATensor>& tensors,
                                   absl::Span<const std::string> devices,
                                   const SyncTensorCollection& coll,
                                   PostOrderData* po_data);

  static std::shared_ptr<Async> SyncTensorsGraphInternal(
      std::vector<XLATensor>* tensors, absl::Span<const std::string> devices,
      const SyncTensorsConfig& config);

  static xla::int64 GetNextTensorId();

  std::shared_ptr<Data> data_;
};

}  // namespace torch_xla
