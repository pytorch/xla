#pragma once

#include <iostream>
#include <string>
#include <unordered_map>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/cache.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch/csrc/autograd/variable.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/view.h"

namespace torch_xla {

class XLATensor {
  class TensorsArena;
  struct Data;

 public:
  static XLATensor Create(const at::Tensor& tensor, const Device& device,
                          bool requires_grad);
  static XLATensor Create(
      xla::ComputationClient::DataPtr xla_data, bool requires_grad,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  static XLATensor Create(
      ir::Value ir_value, const Device& device,
      c10::optional<at::ScalarType> logical_element_type = c10::nullopt);

  // Creates an empty/null tensor.
  XLATensor() = default;

  bool RequiresGrad() const { return data()->requires_grad; }

  void detach_() { data()->requires_grad = false; }

  XLATensor detach() const;

  bool is_null() const { return data_ptr() == nullptr; }

  XLATensor alias() const { return XLATensor(data_ptr()); }

  xla::int64 size(xla::int64 dim) const;

  at::Tensor ToTensor();

  // This API should be called instead of ToTensor() when the tensor is passed
  // to other ATEN APIs which will modify its value.
  at::Tensor ToMutableTensor();

  // Assigns the tensor value to the XLA tensor.
  void SetTensor(at::Tensor tensor);

  c10::optional<XLATensor> grad() const;
  void SetGradient(const XLATensor& grad);

  at::ScalarType dtype() const;
  xla::util::MaybeRef<xla::Shape> shape() const;

  const Device& GetDevice() const;
  xla::int64 GetUniqueId() const;

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

  // Makes the data references from the current tensor, point to the ones from
  // the source tensor.
  void ReferenceDataFrom(const XLATensor& source);

  // Dispatches a comparison operator, setting the logical type of the result
  // appropriately.
  static XLATensor DispatchComparisonOp(c10::Symbol kind,
                                        const XLATensor& input,
                                        at::Scalar other);

  // Same as above, with the second input a tensor as well.
  static XLATensor DispatchComparisonOp(c10::Symbol kind,
                                        const XLATensor& input,
                                        const XLATensor& other);

  // Applies the queue of operations in preparation for using the data.
  void ApplyPendingGraph();

  // Dumps the XLA HLO text of the computation accumulated in the graph which is
  // attached the tensors.
  static std::string DumpHloComputation(const std::vector<XLATensor>& tensors);

  // Returns the common device for "tensors". Throws if not all tensors have the
  // same device.
  static Device CommonDeviceForTensors(const std::vector<XLATensor>& tensors);

  // Retrieves the set of XLA tensors which are currently live in the system.
  static std::vector<XLATensor> GetLiveTensors();

  // Applies all the pending IR operations queued over the input tensors. All
  // the tensors must be on the same device.
  static void SyncTensorsGraph(std::vector<XLATensor>* tensors);

  // Applies the queue of operations for a list of tensors.
  static void ApplyPendingGraph(std::vector<XLATensor>* tensors);

  // Retrieves the PyTorch tensors behind the XLA tensors. If the writeable
  // vector is not nullptr, it must be the same size as tensors, and the
  // corresponding bool tells whether the ATEN tensor to be retrieved should the
  // a writeable copy. All the tensors must be on the same device.
  static std::vector<at::Tensor> GetTensors(std::vector<XLATensor>* tensors,
                                            const std::vector<bool>* writeable);

  // Operation which creates XLA tensors out of autograd variable by batching
  // the requests to the computation servers.
  static std::vector<XLATensor> CreateTensors(
      const std::vector<at::Tensor>& tensors,
      const std::vector<std::string>& devices);

  //////////////////////////////////////////////////////////////////////////////
  // ATEN operators follows here, listed in alphabetical order.
  //////////////////////////////////////////////////////////////////////////////
  static XLATensor __and__(const XLATensor& input, at::Scalar other);
  static XLATensor __and__(const XLATensor& input, const XLATensor& other);

  static void __iand__(XLATensor& input, const XLATensor& other);
  static void __iand__(XLATensor& input, at::Scalar other);

  static void __ilshift__(XLATensor& input, at::Scalar other);
  static void __ilshift__(XLATensor& input, const XLATensor& other);

  static void __ior__(XLATensor& input, const XLATensor& other);
  static void __ior__(XLATensor& input, at::Scalar other);

  static void __irshift__(XLATensor& input, at::Scalar other);
  static void __irshift__(XLATensor& input, const XLATensor& other);

  static void __ixor__(XLATensor& input, const XLATensor& other);
  static void __ixor__(XLATensor& input, at::Scalar other);

  static XLATensor __lshift__(const XLATensor& input, at::Scalar other);
  static XLATensor __lshift__(const XLATensor& input, const XLATensor& other);

  static XLATensor __or__(const XLATensor& input, at::Scalar other);
  static XLATensor __or__(const XLATensor& input, const XLATensor& other);

  static XLATensor __rshift__(const XLATensor& input, at::Scalar other);
  static XLATensor __rshift__(const XLATensor& input, const XLATensor& other);

  static XLATensor __xor__(const XLATensor& input, at::Scalar other);
  static XLATensor __xor__(const XLATensor& input, const XLATensor& other);

  static XLATensor _adaptive_avg_pool2d(const XLATensor& input,
                                        std::vector<xla::int64> output_size);

  static XLATensor _adaptive_avg_pool2d_backward(const XLATensor& grad_output,
                                                 const XLATensor& input);

  static XLATensor abs(const XLATensor& input);
  static void abs_(XLATensor& input);

  static XLATensor acos(const XLATensor& input);
  static void acos_(XLATensor& input);

  static XLATensor add(const XLATensor& input, const XLATensor& other,
                       at::Scalar alpha);
  static void add_(XLATensor& input, const XLATensor& other, at::Scalar alpha);
  static XLATensor add(const XLATensor& input, at::Scalar other,
                       at::Scalar alpha);
  static void add_(XLATensor& input, at::Scalar other, at::Scalar alpha);

  static XLATensor addcdiv(const XLATensor& input, at::Scalar value,
                           const XLATensor& tensor1, const XLATensor& tensor2);
  static void addcdiv_(XLATensor& input, at::Scalar value,
                       const XLATensor& tensor1, const XLATensor& tensor2);

  static XLATensor addcmul(const XLATensor& input, at::Scalar value,
                           const XLATensor& tensor1, const XLATensor& tensor2);
  static void addcmul_(XLATensor& input, at::Scalar value,
                       const XLATensor& tensor1, const XLATensor& tensor2);

  static XLATensor addmm(const XLATensor& input, const XLATensor& weight,
                         const XLATensor& bias);

  static XLATensor all(const XLATensor& input,
                       std::vector<xla::int64> dimensions,
                       bool keep_reduced_dimensions);

  static XLATensor any(const XLATensor& input,
                       std::vector<xla::int64> dimensions,
                       bool keep_reduced_dimensions);

  static XLATensor arange(at::Scalar start, at::Scalar end, at::Scalar step,
                          const Device& device, at::ScalarType scalar_type);

  static XLATensor argmax(const XLATensor& input, xla::int64 dim, bool keepdim);
  static XLATensor argmax(const XLATensor& input);

  static XLATensor argmin(const XLATensor& input, xla::int64 dim, bool keepdim);
  static XLATensor argmin(const XLATensor& input);

  // Takes a slice from the input as R1 at the specified offset and reshapes it
  // into the provided size.
  static XLATensor as_strided(const XLATensor& input,
                              std::vector<xla::int64> size,
                              c10::optional<xla::int64> storage_offset);

  // In-place version of the method above.
  static void as_strided_(XLATensor& input, std::vector<xla::int64> size,
                          c10::optional<xla::int64> storage_offset);

  static XLATensor asin(const XLATensor& input);
  static void asin_(XLATensor& input);

  static XLATensor atan(const XLATensor& input);
  static void atan_(XLATensor& input);

  static XLATensor atan2(const XLATensor& input, const XLATensor& other);
  static void atan2_(XLATensor& input, const XLATensor& other);

  static XLATensor avg_pool_nd(const XLATensor& input,
                               xla::int64 spatial_dim_count,
                               std::vector<xla::int64> kernel_size,
                               std::vector<xla::int64> stride,
                               std::vector<xla::int64> padding,
                               bool count_include_pad);

  static XLATensor avg_pool_nd_backward(const XLATensor& out_backprop,
                                        const XLATensor& input,
                                        xla::int64 spatial_dim_count,
                                        std::vector<xla::int64> kernel_size,
                                        std::vector<xla::int64> stride,
                                        std::vector<xla::int64> padding,
                                        bool count_include_pad);

  static XLATensor batch_norm(const XLATensor& input, const XLATensor& weight,
                              const XLATensor& bias, double momentum,
                              double eps);

  static XLATensor bernoulli(const XLATensor& input, double probability);
  static XLATensor bernoulli(const XLATensor& input);
  static void bernoulli_(XLATensor& input, double probability);
  static void bernoulli_(XLATensor& input, const XLATensor& probability);

  // Batch matrix multiplication. Both tensors must be 3D, the batch size must
  // match and the remaining two dimensions must be compatible for matrix
  // multiplication.
  static XLATensor bmm(const XLATensor& batch1, const XLATensor& batch2);

  // Broadcasts the given tensors according to broadcasting semantics.
  static std::vector<XLATensor> broadcast_tensors(
      tensorflow::gtl::ArraySlice<const XLATensor> tensors);

  static XLATensor cast(const XLATensor& input, at::ScalarType dtype);

  static XLATensor cat(tensorflow::gtl::ArraySlice<const XLATensor> tensors,
                       xla::int64 dim);

  static XLATensor ceil(const XLATensor& input);
  static void ceil_(XLATensor& input);

  static XLATensor cholesky(const XLATensor& input, bool upper);

  static XLATensor clamp(const XLATensor& input, c10::optional<at::Scalar> min,
                         c10::optional<at::Scalar> max);
  static void clamp_(XLATensor& input, c10::optional<at::Scalar> min,
                     c10::optional<at::Scalar> max);

  static XLATensor clone(const XLATensor& input);

  // Pad with the given value and size specified by the given list of low and
  // high paddings.
  static XLATensor constant_pad_nd(
      const XLATensor& input, tensorflow::gtl::ArraySlice<const xla::int64> pad,
      at::Scalar value);

  static XLATensor conv2d(const XLATensor& input, const XLATensor& weight,
                          const XLATensor& bias, std::vector<xla::int64> stride,
                          std::vector<xla::int64> padding);

  static XLATensor conv2d(const XLATensor& input, const XLATensor& weight,
                          std::vector<xla::int64> stride,
                          std::vector<xla::int64> padding);

  static std::tuple<XLATensor, XLATensor, XLATensor> conv2d_backward(
      const XLATensor& out_backprop, const XLATensor& input,
      const XLATensor& weight, std::vector<xla::int64> stride,
      std::vector<xla::int64> padding);

  static XLATensor conv_transpose2d(const XLATensor& input,
                                    const XLATensor& weight,
                                    const XLATensor& bias,
                                    std::vector<xla::int64> stride,
                                    std::vector<xla::int64> padding);

  static XLATensor conv_transpose2d(const XLATensor& input,
                                    const XLATensor& weight,
                                    std::vector<xla::int64> stride,
                                    std::vector<xla::int64> padding);

  static std::tuple<XLATensor, XLATensor, XLATensor> conv_transpose2d_backward(
      const XLATensor& out_backprop, const XLATensor& input,
      const XLATensor& weight, std::vector<xla::int64> stride,
      std::vector<xla::int64> padding);

  static XLATensor cos(const XLATensor& input);
  static void cos_(XLATensor& input);

  static XLATensor cosh(const XLATensor& input);
  static void cosh_(XLATensor& input);

  // Returns the cross product of the two input tensors in the given dimension.
  // If the dimension is not given, it defaults to the first dimension found
  // with the size 3.
  static XLATensor cross(const XLATensor& input, const XLATensor& other,
                         c10::optional<xla::int64> dim);

  static XLATensor cross_replica_sum(
      const XLATensor& input, double scale,
      const std::vector<std::vector<xla::int64>>& groups);

  static void cross_replica_sum_(
      XLATensor& input, double scale,
      const std::vector<std::vector<xla::int64>>& groups);

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

  static XLATensor div(const XLATensor& input, const XLATensor& other);
  static XLATensor div(const XLATensor& input, at::Scalar other);
  static void div_(XLATensor& input, const XLATensor& other);
  static void div_(XLATensor& input, at::Scalar other);

  static XLATensor dropout(const XLATensor& input, double p);
  static void dropout_(XLATensor& input, double p);

  // A generalized contraction between tensors of arbitrary dimension defined by
  // the given equation and applied to the input tensors.
  static XLATensor einsum(const std::string& equation,
                          tensorflow::gtl::ArraySlice<const XLATensor> tensors);

  static XLATensor elu(const XLATensor& input, at::Scalar alpha,
                       at::Scalar scale, at::Scalar input_scale);
  static void elu_(XLATensor& input, at::Scalar alpha, at::Scalar scale,
                   at::Scalar input_scale);
  static XLATensor elu_backward(const XLATensor& grad_output, at::Scalar alpha,
                                at::Scalar scale, at::Scalar input_scale,
                                const XLATensor& output);

  static XLATensor embedding_dense_backward(const XLATensor& grad_output,
                                            const XLATensor& indices,
                                            xla::int64 num_weights,
                                            xla::int64 padding_idx,
                                            bool scale_grad_by_freq);

  static XLATensor eq(const XLATensor& input, at::Scalar other);
  static void eq_(XLATensor& input, at::Scalar other);

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

  // Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
  static XLATensor eye(xla::int64 lines, xla::int64 cols, const Device& device,
                       at::ScalarType element_type);

  // Fills the input with the given value.
  static void fill_(XLATensor& input, at::Scalar value);

  // Flips (reverses) the values in the dimensions of the input tensor.
  static XLATensor flip(const XLATensor& input,
                        tensorflow::gtl::ArraySlice<const xla::int64> dims);

  static XLATensor floor(const XLATensor& input);
  static void floor_(XLATensor& input);

  static XLATensor fmod(const XLATensor& input, const XLATensor& other);
  static XLATensor fmod(const XLATensor& input, at::Scalar other);
  static void fmod_(XLATensor& input, const XLATensor& other);
  static void fmod_(XLATensor& input, at::Scalar other);

  static XLATensor frac(const XLATensor& input);
  static void frac_(XLATensor& input);

  static XLATensor full(tensorflow::gtl::ArraySlice<const xla::int64> size,
                        at::Scalar fill_value, const Device& device,
                        at::ScalarType scalar_type);
  static XLATensor full_like(const XLATensor& input, at::Scalar fill_value,
                             const Device& device,
                             c10::optional<at::ScalarType> scalar_type);

  static XLATensor gather(const XLATensor& input, xla::int64 dim,
                          const XLATensor& index);

  static XLATensor ge(const XLATensor& input, at::Scalar other);
  static void ge_(XLATensor& input, at::Scalar other);

  static XLATensor ge(const XLATensor& input, const XLATensor& other);
  static void ge_(XLATensor& input, const XLATensor& other);

  static XLATensor gt(const XLATensor& input, at::Scalar other);
  static void gt_(XLATensor& input, at::Scalar other);

  static XLATensor gt(const XLATensor& input, const XLATensor& other);
  static void gt_(XLATensor& input, const XLATensor& other);

  static std::tuple<XLATensor, XLATensor> kthvalue(const XLATensor& input,
                                                   xla::int64 k, xla::int64 dim,
                                                   bool keepdim);

  // Gather slices from input into a result with shape specified by indices. The
  // shape of the indices are first made consistent using broadcast semantics.
  // For input of shape d1 x d2 x ... x dn and p indices of shape i1 x i2 x ...
  // x ik, the output shape is i1 x i2 x ... x ik x d(p+1) x ... x dn.
  static XLATensor index(const XLATensor& input,
                         tensorflow::gtl::ArraySlice<const XLATensor> indices);

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
                              const XLATensor& index, at::Scalar value);

  // Same as above, but the value is wrapped as a rank-0 tensor.
  static XLATensor index_fill(const XLATensor& input, xla::int64 dim,
                              const XLATensor& index, const XLATensor& value);

  static void index_fill_(XLATensor& input, xla::int64 dim,
                          const XLATensor& index, const XLATensor& value);

  static void index_fill_(XLATensor& input, xla::int64 dim,
                          const XLATensor& index, at::Scalar value);

  // Puts values into the input tensor using the given indices (a tuple of
  // tensors) and returns the result.
  static XLATensor index_put(
      const XLATensor& input,
      tensorflow::gtl::ArraySlice<const XLATensor> indices,
      const XLATensor& values, bool accumulate,
      tensorflow::gtl::ArraySlice<const xla::int64> result_permutation);

  static void index_put_(
      XLATensor& input, const XLATensor& canonical_base,
      tensorflow::gtl::ArraySlice<const XLATensor> indices,
      const XLATensor& values, bool accumulate,
      tensorflow::gtl::ArraySlice<const xla::int64> result_permutation);

  static XLATensor index_select(const XLATensor& input, xla::int64 dim,
                                const XLATensor& index);

  static XLATensor kl_div_backward(const XLATensor& grad_output,
                                   const XLATensor& input,
                                   const XLATensor& target,
                                   xla::int64 reduction);

  static XLATensor le(const XLATensor& input, at::Scalar other);
  static void le_(XLATensor& input, at::Scalar other);

  static XLATensor le(const XLATensor& input, const XLATensor& other);
  static void le_(XLATensor& input, const XLATensor& other);

  static XLATensor hardshrink(const XLATensor& input, at::Scalar lambda);
  static XLATensor hardshrink_backward(const XLATensor& grad_out,
                                       const XLATensor& input,
                                       at::Scalar lambda);

  static XLATensor hardtanh_backward(const XLATensor& grad_output,
                                     const XLATensor& input, at::Scalar min_val,
                                     at::Scalar max_val);

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

  static XLATensor log_softmax(const XLATensor& input, xla::int64 dim);

  static XLATensor log_softmax_backward(const XLATensor& grad_output,
                                        const XLATensor& output,
                                        xla::int64 dim);

  static XLATensor log1p(const XLATensor& input);
  static void log1p_(XLATensor& input);

  static XLATensor lt(const XLATensor& input, at::Scalar other);
  static void lt_(XLATensor& input, at::Scalar other);

  static XLATensor lt(const XLATensor& input, const XLATensor& other);
  static void lt_(XLATensor& input, const XLATensor& other);

  // Fills elements of the input tensor with the provided value where mask is
  // one. The shape of mask must be broadcastable with the shape of the
  // underlying tensor.
  static XLATensor masked_fill(const XLATensor& input, const XLATensor& mask,
                               at::Scalar value);

  // In-place version of the method above.
  static void masked_fill_(XLATensor& input, const XLATensor& mask,
                           at::Scalar value);

  static XLATensor matmul(const XLATensor& input, const XLATensor& other);

  static XLATensor max(const XLATensor& input, const XLATensor& other);

  static XLATensor max(const XLATensor& input);

  static XLATensor max_pool_nd(const XLATensor& input,
                               xla::int64 spatial_dim_count,
                               std::vector<xla::int64> kernel_size,
                               std::vector<xla::int64> stride,
                               std::vector<xla::int64> padding);

  static XLATensor max_pool_nd_backward(const XLATensor& out_backprop,
                                        const XLATensor& input,
                                        xla::int64 spatial_dim_count,
                                        std::vector<xla::int64> kernel_size,
                                        std::vector<xla::int64> stride,
                                        std::vector<xla::int64> padding);

  static XLATensor mean(const XLATensor& input,
                        std::vector<xla::int64> dimensions,
                        bool keep_reduced_dimensions,
                        c10::optional<at::ScalarType> dtype);

  static XLATensor min(const XLATensor& input, const XLATensor& other);

  static XLATensor min(const XLATensor& input);

  static XLATensor mm(const XLATensor& input, const XLATensor& weight);

  static XLATensor mul(const XLATensor& input, const XLATensor& other);
  static XLATensor mul(const XLATensor& input, at::Scalar other);
  static void mul_(XLATensor& input, const XLATensor& other);
  static void mul_(XLATensor& input, at::Scalar other);

  // Returns a new tensor that is a narrowed view of the input in the given
  // dimension.
  static XLATensor narrow(const XLATensor& input, xla::int64 dim,
                          xla::int64 start, xla::int64 length);

  // Like batch_norm, but returns additional save_mean and save_invstd used by
  // the backward pass.
  static std::tuple<XLATensor, XLATensor, XLATensor> native_batch_norm(
      const XLATensor& input, const XLATensor& weight, const XLATensor& bias,
      double momentum, double eps);

  // Returns the input, weight and bias gradients.
  static std::tuple<XLATensor, XLATensor, XLATensor> native_batch_norm_backward(
      const XLATensor& grad_out, const XLATensor& input,
      const XLATensor& weight, const XLATensor& save_mean,
      const XLATensor& save_invstd, double eps);

  static XLATensor ne(const XLATensor& input, at::Scalar other);
  static void ne_(XLATensor& input, at::Scalar other);

  static XLATensor ne(const XLATensor& input, const XLATensor& other);
  static void ne_(XLATensor& input, const XLATensor& other);

  static XLATensor neg(const XLATensor& input);
  static void neg_(XLATensor& input);

  static XLATensor nll_loss(const XLATensor& input, const XLATensor& target);

  static XLATensor nll_loss_backward(const XLATensor& input,
                                     const XLATensor& target);

  static XLATensor norm(const XLATensor& input, c10::optional<at::Scalar> p,
                        c10::optional<at::ScalarType> dtype,
                        at::IntArrayRef dim, bool keepdim);

  static XLATensor not_supported(std::string description, xla::Shape shape,
                                 const Device& device);

  // Permute the dimensions of this tensor according to the given permutation.
  static XLATensor permute(const XLATensor& input,
                           tensorflow::gtl::ArraySlice<const xla::int64> dims);

  static XLATensor pow(const XLATensor& input, at::Scalar exponent);
  static XLATensor pow(const XLATensor& input, const XLATensor& exponent);
  static XLATensor pow(at::Scalar input, const XLATensor& exponent);
  static void pow_(XLATensor& input, at::Scalar exponent);
  static void pow_(XLATensor& input, const XLATensor& exponent);

  static XLATensor prod(const XLATensor& input,
                        std::vector<xla::int64> dimensions,
                        bool keep_reduced_dimensions,
                        c10::optional<at::ScalarType> dtype);

  static std::tuple<XLATensor, XLATensor> qr(const XLATensor& input,
                                             bool full_matrices);

  static XLATensor reciprocal(const XLATensor& input);
  static void reciprocal_(XLATensor& input);

  static XLATensor relu(const XLATensor& input);
  static void relu_(XLATensor& input);

  static XLATensor remainder(const XLATensor& input, const XLATensor& other);
  static XLATensor remainder(const XLATensor& input, at::Scalar other);
  static void remainder_(XLATensor& input, const XLATensor& other);
  static void remainder_(XLATensor& input, at::Scalar other);

  // Repeats the input tensor along each dimension by the given number of
  // repeats.
  static XLATensor repeat(const XLATensor& input,
                          std::vector<xla::int64> repeats);

  // Returns a tensor with the same data and number of elements as input, but
  // with the specified shape.
  static XLATensor reshape(const XLATensor& input,
                           std::vector<xla::int64> output_size);

  static XLATensor rsqrt(const XLATensor& input);
  static void rsqrt_(XLATensor& input);

  static XLATensor rsub(const XLATensor& input, const XLATensor& other,
                        at::Scalar alpha);

  static XLATensor rsub(const XLATensor& input, at::Scalar other,
                        at::Scalar alpha);

  static void s_copy_(XLATensor& input, XLATensor& src);

  static void scatter_(XLATensor& input, xla::int64 dim, const XLATensor& index,
                       const XLATensor& src);
  static XLATensor scatter(const XLATensor& input, xla::int64 dim,
                           const XLATensor& index, const XLATensor& src);
  static void scatter_(XLATensor& input, xla::int64 dim, const XLATensor& index,
                       at::Scalar value);
  static XLATensor scatter(const XLATensor& input, xla::int64 dim,
                           const XLATensor& index, at::Scalar value);

  static XLATensor select(const XLATensor& input, xla::int64 dim,
                          xla::int64 index);

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
                                  const XLATensor& target,
                                  xla::int64 reduction);

  // Returns the gradient of the input of a smooth_l1_loss operation.
  static XLATensor smooth_l1_loss_backward(const XLATensor& grad_output,
                                           const XLATensor& input,
                                           const XLATensor& target,
                                           xla::int64 reduction);

  static XLATensor softmax(const XLATensor& input, xla::int64 dim);
  static XLATensor softmax_backward(const XLATensor& grad_output,
                                    const XLATensor& output, xla::int64 dim);

  static XLATensor softplus(const XLATensor& input, at::Scalar beta,
                            at::Scalar threshold);
  static XLATensor softplus_backward(const XLATensor& grad_output,
                                     const XLATensor& input, at::Scalar beta,
                                     at::Scalar threshold,
                                     const XLATensor& output);

  static XLATensor softshrink(const XLATensor& input, at::Scalar lambda);
  static XLATensor softshrink_backward(const XLATensor& grad_out,
                                       const XLATensor& input,
                                       at::Scalar lambda);

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

  static XLATensor stack(tensorflow::gtl::ArraySlice<const XLATensor> tensors,
                         xla::int64 dim);

  static XLATensor sub(const XLATensor& input, const XLATensor& other,
                       at::Scalar alpha);
  static void sub_(XLATensor& input, const XLATensor& other, at::Scalar alpha);
  static XLATensor sub(const XLATensor& input, at::Scalar other,
                       at::Scalar alpha);
  static void sub_(XLATensor& input, at::Scalar other, at::Scalar alpha);

  static XLATensor sum(const XLATensor& input,
                       std::vector<xla::int64> dimensions,
                       bool keep_reduced_dimensions,
                       c10::optional<at::ScalarType> dtype);

  static std::tuple<XLATensor, XLATensor, XLATensor> svd(const XLATensor& input,
                                                         bool some,
                                                         bool compute_uv);

  static std::tuple<XLATensor, XLATensor> symeig(const XLATensor& input,
                                                 bool eigenvectors, bool upper);

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

  // Insert a dimension of size one at the specified position.
  static XLATensor unsqueeze(const XLATensor& input, xla::int64 dim);

  // In-place version of the method above.
  static void unsqueeze_(XLATensor& input, xla::int64 dim);

  // Like reshape, but it returns a view into the original tensor.
  static XLATensor view(
      const XLATensor& input,
      tensorflow::gtl::ArraySlice<const xla::int64> output_size);

  static void zero_(XLATensor& input);

  static XLATensor where(const XLATensor& condition, const XLATensor& input,
                         const XLATensor& other);

 private:
  // Maps from ComputationClient Data unique ID to XLA tensor unique ID.
  using DataUidMap = std::unordered_map<xla::int64, xla::int64>;

  struct SyncTensorCollection {
    std::vector<size_t> indices;
    size_t hash = 0;
    std::vector<xla::util::Cleanup> unlocker;
  };

  struct CachedComputation {
    CachedComputation(
        std::shared_ptr<xla::ComputationClient::Computation> computation,
        size_t num_parameters)
        : computation(std::move(computation)), num_parameters(num_parameters) {}

    std::shared_ptr<xla::ComputationClient::Computation> computation;
    size_t num_parameters;
  };

  using ComputationCache = xla::util::Cache<size_t, CachedComputation>;

  struct Async {
    Async(SyncTensorCollection* coll,
          std::vector<xla::ComputationClient::DataPtr> parameters_data,
          std::string device, ComputationCache::TypePtr cached_computation)
        : mwait(1),
          indices(std::move(coll->indices)),
          unlocker(std::move(coll->unlocker)),
          parameters_data(std::move(parameters_data)),
          device(std::move(device)),
          cached_computation(std::move(cached_computation)) {
      tensors_data.reserve(indices.size());
    }

    xla::xla_util::MultiWait mwait;
    std::vector<size_t> indices;
    std::vector<xla::util::Cleanup> unlocker;
    std::vector<xla::ComputationClient::DataPtr> parameters_data;
    std::string device;
    ComputationCache::TypePtr cached_computation;
    std::vector<xla::ComputationClient::DataPtr> tensors_data;
  };

  struct SyncTensorsConfig {
    // Whether we want to force XLA data on the target tensors (hence trimming
    // the IR graph above them).
    bool force_xla_data = true;
  };

  // The context used by the ApplyPendingGraph() API, in order to allow it speed
  // up operations in case the new tensors graph apply matches the one stored
  // within the apply context.
  struct ApplyContext {
    ApplyContext() = default;
    ApplyContext(
        std::vector<std::shared_ptr<xla::ComputationClient::Computation>>
            computations,
        std::vector<xla::int64> uid_order,
        std::vector<std::vector<xla::int64>> input_mapping,
        std::vector<std::vector<xla::int64>> index_mapping,
        std::vector<std::string> devices)
        : computations(std::move(computations)),
          uid_order(std::move(uid_order)),
          input_mapping(std::move(input_mapping)),
          index_mapping(std::move(index_mapping)),
          devices(std::move(devices)) {}

    std::vector<std::shared_ptr<xla::ComputationClient::Computation>>
        computations;
    std::vector<xla::int64> uid_order;
    std::vector<std::vector<xla::int64>> input_mapping;
    std::vector<std::vector<xla::int64>> index_mapping;
    std::vector<std::string> devices;
  };

  using ApplyContextCache = xla::util::Cache<size_t, ApplyContext>;

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
    Device device;
    xla::int64 unique_id = 0;
    std::shared_ptr<XLATensor> grad;
    bool requires_grad = false;
  };

  XLATensor(const at::Tensor& tensor, const Device& device, bool requires_grad);
  XLATensor(xla::ComputationClient::DataPtr xla_data, bool requires_grad,
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

  void SetIrValue(ir::Value ir_value);

  void SetTensorData(at::Tensor tensor_data);

  View::IrNode GetViewUpdate(const std::shared_ptr<View>& view) const;

  std::shared_ptr<View> UpdateView(std::shared_ptr<View> view,
                                   ir::Value ir_value) const;

  XLATensor CreateView(ViewInfo view_info) const;

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

  void SetScalarType(c10::optional<at::ScalarType> logical_element_type);

  // Discards all the XLA and IR data, by making the ATEN tensor one the only
  // source for this XLA tensor. An error is generated if the XLA tensor does
  // not have ATEN tensors data.
  void MakeWriteableTensorDataSource();

  // We build an XLA graph accumulating XLA operations, but at a given point we
  // need to force a rendering, otherwise the graph can grow without control.
  // Think:
  //   for i in range(0, 100000):
  //     a = a + b
  void TryLimitGraphSize();

  std::vector<XLATensor> MakeOutputTensors(ir::NodePtr node) const;

  static ir::Value GetIrValueForTensor(const at::Tensor& tensor,
                                       const Device& device);

  static ir::Value GetIrValueForScalar(at::Scalar value,
                                       xla::PrimitiveType type,
                                       const Device& device);
  static ir::Value GetIrValueForScalar(at::Scalar value,
                                       const xla::Shape& shape,
                                       const Device& device);

  // Create the mapping from computation client Data pointers to the XLA tensors
  // unique ID which are holding it.
  static DataUidMap CreateDataUidMap(const std::vector<XLATensor>& tensors);

  // Tries to run a cached ApplyPendingGraph() with the information in
  // apply_context. Returns whether the cached run could be completed
  // successfully.
  static bool RunCachedApply(std::vector<XLATensor>* tensors,
                             const ApplyContext& apply_context);

  static ComputationCache* GetComputationCache();

  static ApplyContextCache* GetApplyContextCache();

  static SyncTensorCollection CollectSyncTensors(
      const std::vector<XLATensor>& tensors, const SyncTensorsConfig& config);

  // Gathers the XLA device data for all the input tensors, after an
  // asynchronous operation.
  static std::vector<xla::ComputationClient::DataPtr> GatherTensorsXlaData(
      const std::vector<XLATensor>& tensors, std::shared_ptr<Async> async);

  // Schedules the execution of a sync tensors operation in background. The
  // asynchronous operation will hold the device locks by capturing the ones
  // present within the coll structure.
  static std::shared_ptr<Async> ScheduleSyncTensorsGraph(
      std::vector<XLATensor>* tensors, const SyncTensorsConfig& config,
      SyncTensorCollection* coll,
      std::vector<xla::ComputationClient::DataPtr> parameters_data,
      std::string device, ComputationCache::TypePtr cached_computation);

  static std::shared_ptr<Async> TryRunCachedSync(
      std::vector<XLATensor>* tensors, const SyncTensorsConfig& config,
      SyncTensorCollection* coll);

  static std::shared_ptr<Async> SyncTensorsGraphInternal(
      std::vector<XLATensor>* tensors, const SyncTensorsConfig& config);

  // Returns a permutation which represents an ordering by tensor device and
  // unique ID, of all the tensors which needs sync (the ones which have a graph
  // backing their value). The tensors which are already sync, will not be
  // returned within the permutation. If a tensor has at::Tensor data only, the
  // at::Tensor data will be uploaded to the device and the tensor will receive
  // new XLA data. This API will perform a barrier on device locks, and will not
  // hold locks of partecipating devices (as ApplyPendingGraph() is never
  // asynchronous).
  static SyncTensorCollection CollectApplyGraphTensors(
      const std::vector<XLATensor>& tensors);

  static ir::Value CreateTensorNode(xla::ComputationClient::DataPtr data);

  static xla::int64 GetNextTensorId();

  std::shared_ptr<Data> data_;
};

}  // namespace torch_xla
