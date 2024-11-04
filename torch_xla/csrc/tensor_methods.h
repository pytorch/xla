#ifndef XLA_TORCH_XLA_CSRC_TENSOR_METHODS_H_
#define XLA_TORCH_XLA_CSRC_TENSOR_METHODS_H_

#include "torch_xla/csrc/cross_replica_reduces.h"
#include "torch_xla/csrc/ops/custom_sharding.h"
#include "torch_xla/csrc/runtime/computation_client.h"
#include "torch_xla/csrc/tensor.h"

namespace torch_xla {
namespace tensor_methods {

//////////////////////////////////////////////////////////////////////////////
// XLA dedicated operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////
XLATensorPtr all_reduce(const XLATensorPtr& input, AllReduceType reduce_type,
                        double scale, std::vector<std::vector<int64_t>> groups,
                        bool pin_layout);

void all_reduce(const std::vector<XLATensorPtr>& inputs,
                AllReduceType reduce_type, double scale,
                std::vector<std::vector<int64_t>> groups, bool pin_layout);

XLATensorPtr all_reduce(const XLATensorPtr& input, AllReduceType reduce_type,
                        double scale, std::vector<std::vector<int64_t>> groups);

std::pair<XLATensorPtr, torch::lazy::Value> reduce_scatter(
    const XLATensorPtr& input, const torch::lazy::Value& token,
    AllReduceType reduce_type, double scale, int64_t scatter_dim,
    int64_t shard_count, std::vector<std::vector<int64_t>> groups,
    bool pin_layout);

XLATensorPtr reduce_scatter(const XLATensorPtr& input,
                            AllReduceType reduce_type, double scale,
                            int64_t scatter_dim, int64_t shard_count,
                            std::vector<std::vector<int64_t>> groups);

torch::lazy::Value reduce_scatter_out(XLATensorPtr& output,
                                      const XLATensorPtr& input,
                                      const torch::lazy::Value& token,
                                      AllReduceType reduce_type, double scale,
                                      int64_t scatter_dim, int64_t shard_count,
                                      std::vector<std::vector<int64_t>> groups,
                                      bool pin_layout);

std::pair<std::vector<XLATensorPtr>, torch::lazy::Value>
reduce_scatter_coalesced(const std::vector<XLATensorPtr>& inputs,
                         const torch::lazy::Value& token,
                         AllReduceType reduce_type, double scale,
                         int64_t scatter_dim, int64_t shard_count,
                         std::vector<std::vector<int64_t>> groups,
                         bool pin_layout);

torch::lazy::Value reduce_scatter_coalesced_out(
    const std::vector<XLATensorPtr>& outputs,
    const std::vector<XLATensorPtr>& inputs, const torch::lazy::Value& token,
    AllReduceType reduce_type, double scale, int64_t scatter_dim,
    int64_t shard_count, std::vector<std::vector<int64_t>> groups,
    bool pin_layout);

std::pair<XLATensorPtr, torch::lazy::Value> all_to_all(
    const XLATensorPtr& input, const torch::lazy::Value& token,
    int64_t split_dimension, int64_t concat_dimension, int64_t split_count,
    std::vector<std::vector<int64_t>> groups, bool pin_layout);

XLATensorPtr all_gather(const XLATensorPtr& input, int64_t dim,
                        int64_t shard_count,
                        std::vector<std::vector<int64_t>> groups,
                        bool pin_layout);

torch::lazy::Value all_gather_out(XLATensorPtr& output,
                                  const XLATensorPtr& input,
                                  const torch::lazy::Value& token, int64_t dim,
                                  int64_t shard_count,
                                  std::vector<std::vector<int64_t>> groups,
                                  bool pin_layout);

std::pair<std::vector<XLATensorPtr>, torch::lazy::Value> all_gather_coalesced(
    const std::vector<XLATensorPtr>& inputs, const torch::lazy::Value& token,
    int64_t dim, int64_t shard_count, std::vector<std::vector<int64_t>> groups,
    bool pin_layout);

torch::lazy::Value all_gather_coalesced_out(
    std::vector<XLATensorPtr>& outputs, const std::vector<XLATensorPtr>& inputs,
    const torch::lazy::Value& token, int64_t dim, int64_t shard_count,
    std::vector<std::vector<int64_t>> groups, bool pin_layout);

std::pair<XLATensorPtr, torch::lazy::Value> collective_permute(
    const XLATensorPtr& input, const torch::lazy::Value& token,
    std::vector<std::pair<int64_t, int64_t>> source_target_pairs);

std::vector<XLATensorPtr> custom_call(
    const std::vector<XLATensorPtr>& inputs, const std::string& target,
    const std::vector<std::vector<int64_t>>& output_shapes,
    const std::vector<at::ScalarType>& output_dtypes, bool has_side_effect,
    const std::string& backend_config, const int api_version,
    const std::unordered_map<std::string, std::string>& frontend_attributes);

void custom_sharding_(
    const XLATensorPtr& input,
    const std::shared_ptr<XLATensor::ShardingSpec>& spec,
    const CustomSharding::Type& type = CustomSharding::Type::kSharding);

std::vector<XLATensorPtr> gpu_custom_call(
    const std::vector<XLATensorPtr>& inputs, const std::string& payload,
    const std::vector<std::vector<int64_t>>& output_shapes,
    const std::vector<at::ScalarType>& output_dtypes);

std::vector<XLATensorPtr> tpu_custom_call(
    const std::vector<XLATensorPtr>& inputs, const std::string& payload,
    const std::vector<std::vector<int64_t>>& output_shapes,
    const std::vector<at::ScalarType>& output_dtypes);

XLATensorPtr get_dimensions_size(const XLATensorPtr& input,
                                 std::vector<int64_t> dimensions);

std::pair<XLATensorPtr, torch::lazy::Value> recv(
    XLATensorPtr& output, const torch::lazy::Value& token, int64_t channel_id);

std::pair<XLATensorPtr, torch::lazy::Value> send(
    const XLATensorPtr& input, const torch::lazy::Value& token,
    int64_t channel_id);

void sgd_optimizer_step_(const XLATensorPtr& found_inf, XLATensorPtr& step,
                         XLATensorPtr& param, XLATensorPtr& buf,
                         const XLATensorPtr& d_p, double weight_decay,
                         double momentum, double lr, double dampening,
                         bool nesterov, bool maximize);

void adam_optimizer_step_(const XLATensorPtr& found_inf, XLATensorPtr& step,
                          XLATensorPtr& param, const XLATensorPtr& grad,
                          XLATensorPtr& exp_avg, XLATensorPtr& exp_avg_sq,
                          XLATensorPtr& max_exp_avg_sq, double beta1,
                          double beta2, double lr, double weight_decay,
                          double eps, bool amsgrad, bool maximize,
                          bool use_adamw);

std::vector<XLATensorPtr> user_computation(
    const std::string& opname, absl::Span<const XLATensorPtr> inputs,
    runtime::ComputationClient::ComputationPtr computation);

//////////////////////////////////////////////////////////////////////////////
// Quantization related ops here.
//////////////////////////////////////////////////////////////////////////////
XLATensorPtr quantize_tensor(const XLATensorPtr& input,
                             const std::vector<float>& scale_list,
                             const std::vector<int>& zero_point_list,
                             int quant_min, int quant_max,
                             const std::string& dtype, int axis);

XLATensorPtr dequantize_tensor(const XLATensorPtr& input,
                               const std::vector<float>& scale_list,
                               const std::vector<int>& zero_point_list,
                               int quant_min, int quant_max,
                               const std::string& dtype, int axis);

XLATensorPtr cast_int4(const XLATensorPtr& weight,
                       const std::vector<int>& int4_vals);

//////////////////////////////////////////////////////////////////////////////
// Dynamic Reshape ops here.
//////////////////////////////////////////////////////////////////////////////

XLATensorPtr dynamic_expand(const XLATensorPtr& input,
                            const std::vector<int64_t>& size,
                            const XLATensorPtr& src_tensor, int src_dim,
                            int target_dim);

XLATensorPtr dynamic_view(const XLATensorPtr& input,
                          const std::vector<int64_t>& size,
                          const XLATensorPtr& src_tensor, int src_dim,
                          int target_dim, float mul_scaler);

//////////////////////////////////////////////////////////////////////////////
// ATEN operators follows here, listed in alphabetical order.
//////////////////////////////////////////////////////////////////////////////
void __ilshift__(XLATensorPtr& input, const at::Scalar& other);
void __ilshift__(XLATensorPtr& input, const XLATensorPtr& other);

void __irshift__(XLATensorPtr& input, const at::Scalar& other);
void __irshift__(XLATensorPtr& input, const XLATensorPtr& other);

XLATensorPtr __lshift__(
    const XLATensorPtr& input, const at::Scalar& other,
    std::optional<at::ScalarType> logical_element_type = std::nullopt);
XLATensorPtr __lshift__(
    const XLATensorPtr& input, const XLATensorPtr& other,
    std::optional<at::ScalarType> logical_element_type = std::nullopt);

XLATensorPtr __rshift__(
    const XLATensorPtr& input, const at::Scalar& other,
    std::optional<at::ScalarType> logical_element_type = std::nullopt);
XLATensorPtr __rshift__(
    const XLATensorPtr& input, const XLATensorPtr& other,
    std::optional<at::ScalarType> logical_element_type = std::nullopt);

std::tuple<XLATensorPtr, XLATensorPtr> adaptive_max_pool2d(
    const XLATensorPtr& input, std::vector<int64_t> output_size);

XLATensorPtr adaptive_max_pool2d_backward(const XLATensorPtr& grad_output,
                                          const XLATensorPtr& input);

XLATensorPtr _adaptive_avg_pool2d(const XLATensorPtr& input,
                                  std::vector<int64_t> output_size);

XLATensorPtr _adaptive_avg_pool2d_backward(const XLATensorPtr& grad_output,
                                           const XLATensorPtr& input);

void _amp_foreach_non_finite_check_and_unscale_(std::vector<XLATensorPtr> self,
                                                XLATensorPtr& found_inf,
                                                const XLATensorPtr& inv_scale);

void _amp_update_scale_(XLATensorPtr& current_scale,
                        XLATensorPtr& growth_tracker,
                        const XLATensorPtr& found_inf,
                        double scale_growth_factor, double scale_backoff_factor,
                        int growth_interval);

XLATensorPtr abs(const XLATensorPtr& input);

XLATensorPtr add(
    const XLATensorPtr& input, const XLATensorPtr& other,
    const at::Scalar& alpha,
    std::optional<at::ScalarType> logical_element_type = std::nullopt);
XLATensorPtr add(
    const XLATensorPtr& input, const at::Scalar& other, const at::Scalar& alpha,
    std::optional<at::ScalarType> logical_element_type = std::nullopt);

XLATensorPtr addcdiv(const XLATensorPtr& input, const at::Scalar& value,
                     const XLATensorPtr& tensor1, const XLATensorPtr& tensor2);
void addcdiv_(XLATensorPtr& input, const at::Scalar& value,
              const XLATensorPtr& tensor1, const XLATensorPtr& tensor2);

XLATensorPtr addcmul(const XLATensorPtr& input, const at::Scalar& value,
                     const XLATensorPtr& tensor1, const XLATensorPtr& tensor2);

XLATensorPtr addmm(const XLATensorPtr& input, const XLATensorPtr& weight,
                   const XLATensorPtr& bias);

XLATensorPtr alias(const XLATensorPtr& input);

XLATensorPtr amax(const XLATensorPtr& input, std::vector<int64_t> dimensions,
                  bool keep_reduced_dimensions);

XLATensorPtr amin(const XLATensorPtr& input, std::vector<int64_t> dimensions,
                  bool keep_reduced_dimensions);

void arange_out(XLATensorPtr& out, const at::Scalar& start,
                const at::Scalar& end, const at::Scalar& step,
                at::ScalarType scalar_type);

// Takes a slice from the input as R1 at the specified offset and reshapes it
// into the provided size.
XLATensorPtr as_strided(const XLATensorPtr& input, std::vector<int64_t> size,
                        std::vector<int64_t> stride,
                        std::optional<int64_t> storage_offset);

// In-place version of the method above.
void as_strided_(XLATensorPtr& input, std::vector<int64_t> size,
                 std::vector<int64_t> stride,
                 std::optional<int64_t> storage_offset);

XLATensorPtr avg_pool_nd(const XLATensorPtr& input, int64_t spatial_dim_count,
                         std::vector<int64_t> kernel_size,
                         std::vector<int64_t> stride,
                         std::vector<int64_t> padding, bool ceil_mode,
                         bool count_include_pad,
                         std::optional<int> divisor_override);

XLATensorPtr avg_pool_nd_backward(const XLATensorPtr& out_backprop,
                                  const XLATensorPtr& input,
                                  int64_t spatial_dim_count,
                                  std::vector<int64_t> kernel_size,
                                  std::vector<int64_t> stride,
                                  std::vector<int64_t> padding, bool ceil_mode,
                                  bool count_include_pad);

XLATensorPtr baddbmm(const XLATensorPtr& input, const XLATensorPtr& batch1,
                     const XLATensorPtr& batch2, const at::Scalar& beta,
                     const at::Scalar& alpha);

XLATensorPtr bernoulli(const XLATensorPtr& input, double probability);
XLATensorPtr bernoulli(const XLATensorPtr& input);
void bernoulli_(XLATensorPtr& input, const XLATensorPtr& probability);

XLATensorPtr bitwise_and(const XLATensorPtr& input, const at::Scalar& other);

XLATensorPtr bitwise_and(const XLATensorPtr& input, const XLATensorPtr& other);

XLATensorPtr bitwise_not(const XLATensorPtr& input);

XLATensorPtr bitwise_or(const XLATensorPtr& input, const at::Scalar& other);

XLATensorPtr bitwise_or(const XLATensorPtr& input, const XLATensorPtr& other);

XLATensorPtr bitwise_xor(const XLATensorPtr& input, const at::Scalar& other);

XLATensorPtr bitwise_xor(const XLATensorPtr& input, const XLATensorPtr& other);

// Batch matrix multiplication. Both tensors must be 3D, the batch size must
// match and the remaining two dimensions must be compatible for matrix
// multiplication.
XLATensorPtr bmm(const XLATensorPtr& batch1, const XLATensorPtr& batch2);

// Broadcasts the given tensors according to broadcasting semantics.
std::vector<XLATensorPtr> broadcast_tensors(
    absl::Span<const XLATensorPtr> tensors);

XLATensorPtr cat(absl::Span<const XLATensorPtr> tensors, int64_t dim,
                 at::ScalarType dtype);

XLATensorPtr cdist_forward(const XLATensorPtr& x1, const XLATensorPtr& x2,
                           double p);

XLATensorPtr pdist_forward(const XLATensorPtr& input, double p);

XLATensorPtr pixel_shuffle(const XLATensorPtr& self, int64_t upscale_factor);

XLATensorPtr celu(const XLATensorPtr& input, const at::Scalar& alpha);
void celu_(XLATensorPtr& input, const at::Scalar& alpha);

XLATensorPtr clamp(const XLATensorPtr& input,
                   const std::optional<at::Scalar>& min,
                   const std::optional<at::Scalar>& max);
XLATensorPtr clamp(const XLATensorPtr& input,
                   const std::optional<at::Tensor>& min,
                   const std::optional<at::Tensor>& max);

XLATensorPtr clone(const XLATensorPtr& input);

// Pad with the given value and size specified by the given list of low and
// high paddings.
XLATensorPtr constant_pad_nd(const XLATensorPtr& input,
                             absl::Span<const int64_t> pad,
                             const at::Scalar& value);

XLATensorPtr convolution_overrideable(
    const XLATensorPtr& input, const XLATensorPtr& weight,
    const XLATensorPtr& bias, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    bool transposed, std::vector<int64_t> output_padding, int64_t groups);

std::tuple<XLATensorPtr, XLATensorPtr, XLATensorPtr>
convolution_backward_overrideable(
    const XLATensorPtr& out_backprop, const XLATensorPtr& input,
    const XLATensorPtr& weight, std::vector<int64_t> stride,
    std::vector<int64_t> padding, std::vector<int64_t> dilation,
    bool transposed, std::vector<int64_t> output_padding, int64_t groups);

XLATensorPtr convolution_overrideable(
    const XLATensorPtr& input, const XLATensorPtr& weight,
    std::vector<int64_t> stride, std::vector<int64_t> padding,
    std::vector<int64_t> dilation, bool transposed,
    std::vector<int64_t> output_padding, int64_t groups);

XLATensorPtr count_nonzero(const XLATensorPtr& input,
                           std::vector<int64_t> dims);

// Returns the cross product of the two input tensors in the given dimension.
// If the dimension is not given, it defaults to the first dimension found
// with the size 3.
XLATensorPtr cross(const XLATensorPtr& input, const XLATensorPtr& other,
                   std::optional<int64_t> dim);

// Returns the cumulative product of elements of input in the given dimension.
XLATensorPtr cumprod(const XLATensorPtr& input, int64_t dim,
                     std::optional<at::ScalarType> dtype);

// Returns the cumulative sum of elements of input in the given dimension.
XLATensorPtr cumsum(const XLATensorPtr& input, int64_t dim,
                    std::optional<at::ScalarType> dtype);

// If the input is a matrix (2-D tensor), returns a 1-D tensor with the
// diagonal elements of the input. If the input is a vector (1-D tensor),
// returns a 2-D square tensor with the elements of input as the diagonal.
XLATensorPtr diag(const XLATensorPtr& input, int64_t offset);

// Returns the diagonal of a matrix (2-D tensor) or batch of matrices. The
// matrix dimensions are specified by dim1 and dim2, the diagonal by offset.
XLATensorPtr diagonal(const XLATensorPtr& input, int64_t offset, int64_t dim1,
                      int64_t dim2);

XLATensorPtr div(
    const XLATensorPtr& input, const XLATensorPtr& other,
    const std::optional<c10::string_view>& rounding_mode = std::nullopt,
    std::optional<at::ScalarType> logical_element_type = std::nullopt);
XLATensorPtr div(const XLATensorPtr& input, const at::Scalar& other);

XLATensorPtr xla_dot_general(
    const XLATensorPtr& lhs, const XLATensorPtr& rhs,
    const std::vector<std::vector<int>>& dim_vectors,
    std::optional<at::ScalarType> preferred_element_type);

// A generalized contraction between tensors of arbitrary dimension defined by
// the given equation and applied to the input tensors.
XLATensorPtr einsum(const std::string& equation,
                    absl::Span<const XLATensorPtr> tensors);

std::tuple<XLATensorPtr, XLATensorPtr> einsum_backward(
    const XLATensorPtr& grad_output,
    const absl::Span<const XLATensorPtr> tensors, const std::string& equation);

XLATensorPtr elu_backward(const XLATensorPtr& grad_output,
                          const at::Scalar& alpha, const at::Scalar& scale,
                          const at::Scalar& input_scale,
                          const XLATensorPtr& output);

XLATensorPtr embedding_dense_backward(const XLATensorPtr& grad_output,
                                      const XLATensorPtr& indices,
                                      int64_t num_weights, int64_t padding_idx,
                                      bool scale_grad_by_freq);

std::tuple<XLATensorPtr, XLATensorPtr, XLATensorPtr, XLATensorPtr>
embedding_bag(const XLATensorPtr& weight, const XLATensorPtr& indices,
              const XLATensorPtr& offsets, int64_t mode,
              const XLATensorPtr& per_sample_weights, bool include_last_offset);

XLATensorPtr embedding(const XLATensorPtr& weight, const XLATensorPtr& indices);

XLATensorPtr eq(const XLATensorPtr& input, const at::Scalar& other);

XLATensorPtr eq(const XLATensorPtr& input, const XLATensorPtr& other);

XLATensorPtr exp(const XLATensorPtr& input);

XLATensorPtr expand(const XLATensorPtr& input, std::vector<int64_t> size);

XLATensorPtr expand_symint(const XLATensorPtr& input,
                           c10::SymIntArrayRef sym_size);

void exponential_(XLATensorPtr& input, double lambd);

// Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
XLATensorPtr eye(int64_t lines, int64_t cols,
                 const torch::lazy::BackendDevice& device,
                 at::ScalarType element_type);

void eye_out(XLATensorPtr& out, int64_t lines, int64_t cols);

// Fills the input with the given value.
void fill_(XLATensorPtr& input, const at::Scalar& value);

// Flips (reverses) the values in the dimensions of the input tensor.
XLATensorPtr flip(const XLATensorPtr& input, absl::Span<const int64_t> dims);

XLATensorPtr fmod(
    const XLATensorPtr& input, const XLATensorPtr& other,
    std::optional<at::ScalarType> logical_element_type = std::nullopt);
XLATensorPtr fmod(
    const XLATensorPtr& input, const at::Scalar& other,
    std::optional<at::ScalarType> logical_element_type = std::nullopt);

XLATensorPtr full(absl::Span<const int64_t> size, const at::Scalar& fill_value,
                  const torch::lazy::BackendDevice& device,
                  at::ScalarType scalar_type);
XLATensorPtr full_like(const XLATensorPtr& input, const at::Scalar& fill_value,
                       const torch::lazy::BackendDevice& device,
                       std::optional<at::ScalarType> scalar_type);
XLATensorPtr full_symint(at::SymIntArrayRef sym_size,
                         const at::Scalar& fill_value,
                         const torch::lazy::BackendDevice& device,
                         at::ScalarType scalar_type);

XLATensorPtr gather(const XLATensorPtr& input, int64_t dim,
                    const XLATensorPtr& index);

XLATensorPtr ge(const XLATensorPtr& input, const at::Scalar& other);

XLATensorPtr ge(const XLATensorPtr& input, const XLATensorPtr& other);

XLATensorPtr gelu(const XLATensorPtr& input,
                  const c10::string_view approximate);

XLATensorPtr gelu_backward(const XLATensorPtr& grad, const XLATensorPtr& input,
                           const c10::string_view approximate);

XLATensorPtr gt(const XLATensorPtr& input, const at::Scalar& other);

XLATensorPtr gt(const XLATensorPtr& input, const XLATensorPtr& other);

// Gather slices from input into a result with shape specified by indices. The
// shape of the indices are first made consistent using broadcast semantics.
// For input of shape d1 x d2 x ... x dn and p indices of shape i1 x i2 x ...
// x ik, the output shape is d1 x ... x d(start_dim) x i1 x ... x ik x
// d(start_dim+p+1) x ... x dn.
XLATensorPtr index(const XLATensorPtr& input,
                   absl::Span<const XLATensorPtr> indices, int64_t start_dim);

XLATensorPtr index_add(const XLATensorPtr& input, int64_t dim,
                       const XLATensorPtr& index, const XLATensorPtr& source,
                       const at::Scalar& alpha);

XLATensorPtr index_copy(const XLATensorPtr& input, int64_t dim,
                        const XLATensorPtr& index, const XLATensorPtr& source);

// Fills the elements of the base tensor with the given value in the given
// dimension, at positions given by the index. The index must be a rank-1
// tensor.
XLATensorPtr index_fill(const XLATensorPtr& input, int64_t dim,
                        const XLATensorPtr& index, const at::Scalar& value);

// Same as above, but the value is wrapped as a rank-0 tensor.
XLATensorPtr index_fill(const XLATensorPtr& input, int64_t dim,
                        const XLATensorPtr& index, const XLATensorPtr& value);

void index_fill_(XLATensorPtr& input, int64_t dim, const XLATensorPtr& index,
                 const XLATensorPtr& value);

void index_fill_(XLATensorPtr& input, int64_t dim, const XLATensorPtr& index,
                 const at::Scalar& value);

// Puts values into the input tensor using the given indices (a tuple of
// tensors) and returns the result.
XLATensorPtr index_put(const XLATensorPtr& input,
                       absl::Span<const XLATensorPtr> indices,
                       int64_t start_dim, const XLATensorPtr& values,
                       bool accumulate,
                       absl::Span<const int64_t> result_permutation);

void index_put_(XLATensorPtr& input, const XLATensorPtr& canonical_base,
                absl::Span<const XLATensorPtr> indices, int64_t start_dim,
                const XLATensorPtr& values, bool accumulate,
                absl::Span<const int64_t> result_permutation);

XLATensorPtr index_select(const XLATensorPtr& input, int64_t dim,
                          const XLATensorPtr& index);

XLATensorPtr isnan(const XLATensorPtr& input);

std::tuple<XLATensorPtr, XLATensorPtr> kthvalue(const XLATensorPtr& input,
                                                int64_t k, int64_t dim,
                                                bool keepdim);

XLATensorPtr le(const XLATensorPtr& input, const at::Scalar& other);

XLATensorPtr le(const XLATensorPtr& input, const XLATensorPtr& other);

XLATensorPtr hardshrink(const XLATensorPtr& input, const at::Scalar& lambda);
XLATensorPtr hardshrink_backward(const XLATensorPtr& grad_out,
                                 const XLATensorPtr& input,
                                 const at::Scalar& lambda);

XLATensorPtr hardtanh_backward(const XLATensorPtr& grad_output,
                               const XLATensorPtr& input,
                               const at::Scalar& min_val,
                               const at::Scalar& max_val);

XLATensorPtr lerp(const XLATensorPtr& input, const XLATensorPtr& end,
                  const XLATensorPtr& weight);
XLATensorPtr lerp(const XLATensorPtr& input, const XLATensorPtr& end,
                  const at::Scalar& weight);

XLATensorPtr linalg_vector_norm(const XLATensorPtr& input,
                                const at::Scalar& ord,
                                std::vector<int64_t> dimensions, bool keep_dim,
                                std::optional<at::ScalarType> dtype);

XLATensorPtr linspace(const at::Scalar& start, const at::Scalar& end,
                      const int64_t steps, at::ScalarType element_type,
                      const torch::lazy::BackendDevice& device);

XLATensorPtr log(const XLATensorPtr& input);

XLATensorPtr logit(const XLATensorPtr& input, std::optional<double> eps);

XLATensorPtr log_base(const XLATensorPtr& input, torch::lazy::OpKind op,
                      double base);

XLATensorPtr log_sigmoid(const XLATensorPtr& input);

XLATensorPtr log_softmax(const XLATensorPtr& input, int64_t dim,
                         std::optional<at::ScalarType> dtype,
                         std::vector<torch::lazy::Shape>&& shapes);

XLATensorPtr log_softmax_backward(const XLATensorPtr& grad_output,
                                  const XLATensorPtr& output, int64_t dim);

XLATensorPtr log1p(const XLATensorPtr& input);
void log1p_(XLATensorPtr& input);

XLATensorPtr logsumexp(const XLATensorPtr& input,
                       std::vector<int64_t> dimensions,
                       bool keep_reduced_dimensions);

XLATensorPtr xlogy(const XLATensorPtr& input, const XLATensorPtr& other);

XLATensorPtr lt(const XLATensorPtr& input, const at::Scalar& other);

XLATensorPtr lt(const XLATensorPtr& input, const XLATensorPtr& other);

XLATensorPtr mark_tensor(const XLATensorPtr& input, const std::string& info);

XLATensorPtr masked_scatter(XLATensorPtr& input, const XLATensorPtr& mask,
                            const XLATensorPtr& source);

XLATensorPtr masked_select(const XLATensorPtr& input, const XLATensorPtr& mask);

XLATensorPtr matmul(const XLATensorPtr& input, const XLATensorPtr& other);

XLATensorPtr max(const XLATensorPtr& input);

std::tuple<XLATensorPtr, XLATensorPtr> max(const XLATensorPtr& input,
                                           int64_t dim, bool keepdim);

void max_out(XLATensorPtr& max, XLATensorPtr& max_values,
             const XLATensorPtr& input, int64_t dim, bool keepdim);

std::tuple<XLATensorPtr, XLATensorPtr> max_pool_nd(
    const XLATensorPtr& input, int64_t spatial_dim_count,
    std::vector<int64_t> kernel_size, std::vector<int64_t> stride,
    std::vector<int64_t> padding, bool ceil_mode);

XLATensorPtr max_pool_nd_backward(const XLATensorPtr& out_backprop,
                                  const XLATensorPtr& input,
                                  int64_t spatial_dim_count,
                                  std::vector<int64_t> kernel_size,
                                  std::vector<int64_t> stride,
                                  std::vector<int64_t> padding, bool ceil_mode);

XLATensorPtr max_unpool(const XLATensorPtr& input, const XLATensorPtr& indices,
                        std::vector<int64_t> output_size);

XLATensorPtr max_unpool_backward(const XLATensorPtr& grad_output,
                                 const XLATensorPtr& input,
                                 const XLATensorPtr& indices,
                                 std::vector<int64_t> output_size);

XLATensorPtr mean(const XLATensorPtr& input, std::vector<int64_t> dimensions,
                  bool keep_reduced_dimensions,
                  std::optional<at::ScalarType> dtype);

XLATensorPtr min(
    const XLATensorPtr& input, const XLATensorPtr& other,
    std::optional<at::ScalarType> logical_element_type = std::nullopt);

XLATensorPtr min(const XLATensorPtr& input);

std::tuple<XLATensorPtr, XLATensorPtr> min(const XLATensorPtr& input,
                                           int64_t dim, bool keepdim);

void min_out(XLATensorPtr& min, XLATensorPtr& min_indices,
             const XLATensorPtr& input, int64_t dim, bool keepdim);

XLATensorPtr mish(const XLATensorPtr& input);

XLATensorPtr mm(const XLATensorPtr& input, const XLATensorPtr& weight);

XLATensorPtr mse_loss(const XLATensorPtr& input, const XLATensorPtr& target,
                      int64_t reduction);

XLATensorPtr mse_loss_backward(const XLATensorPtr& grad_output,
                               const XLATensorPtr& input,
                               const XLATensorPtr& target, int64_t reduction);

XLATensorPtr mul(
    const XLATensorPtr& input, const XLATensorPtr& other,
    std::optional<at::ScalarType> logical_element_type = std::nullopt);
XLATensorPtr mul(
    const XLATensorPtr& input, const at::Scalar& other,
    std::optional<at::ScalarType> logical_element_type = std::nullopt);

XLATensorPtr multinomial(const XLATensorPtr& input, int64_t num_samples,
                         bool replacement);

XLATensorPtr mv(const XLATensorPtr& input, const XLATensorPtr& vec);
void mv_out(XLATensorPtr& out, const XLATensorPtr& input,
            const XLATensorPtr& vec);

XLATensorPtr nan_to_num(const XLATensorPtr& input, const at::Scalar& nan,
                        const at::Scalar& posinf, const at::Scalar& neginf);

// Returns a new tensor that is a narrowed view of the input in the given
// dimension.
XLATensorPtr narrow(const XLATensorPtr& input, int64_t dim, int64_t start,
                    int64_t length);

// Like batch_norm, but returns additional save_mean and save_invstd used by
// the backward pass.
std::tuple<XLATensorPtr, XLATensorPtr, XLATensorPtr> native_batch_norm(
    const XLATensorPtr& input, const XLATensorPtr& weight,
    const XLATensorPtr& bias, XLATensorPtr& running_mean,
    XLATensorPtr& running_var, bool training, double momentum, double eps);

// Returns the input, weight and bias gradients.
std::tuple<XLATensorPtr, XLATensorPtr, XLATensorPtr> native_batch_norm_backward(
    const XLATensorPtr& grad_out, const XLATensorPtr& input,
    const XLATensorPtr& weight, const XLATensorPtr& save_mean,
    const XLATensorPtr& save_invstd, bool training, double eps);

std::tuple<XLATensorPtr, XLATensorPtr> native_dropout(
    const XLATensorPtr& input, double p, std::optional<bool> train);

XLATensorPtr ne(const XLATensorPtr& input, const at::Scalar& other);

XLATensorPtr ne(const XLATensorPtr& input, const XLATensorPtr& other);

XLATensorPtr neg(const XLATensorPtr& input);

XLATensorPtr nll_loss(const XLATensorPtr& input, const XLATensorPtr& target,
                      const XLATensorPtr& weight, int64_t reduction,
                      int ignore_index);

XLATensorPtr nll_loss2d(const XLATensorPtr& input, const XLATensorPtr& target,
                        const XLATensorPtr& weight, int64_t reduction,
                        int ignore_index);

XLATensorPtr nll_loss2d_backward(const XLATensorPtr& grad_output,
                                 const XLATensorPtr& input,
                                 const XLATensorPtr& target,
                                 const XLATensorPtr& weight, int64_t reduction,
                                 int ignore_index,
                                 const XLATensorPtr& total_weight);

XLATensorPtr nll_loss_backward(const XLATensorPtr& grad_output,
                               const XLATensorPtr& input,
                               const XLATensorPtr& target,
                               const XLATensorPtr& weight, int64_t reduction,
                               int ignore_index,
                               const XLATensorPtr& total_weight);

XLATensorPtr nms(const XLATensorPtr& boxes, const XLATensorPtr& scores,
                 double iou_threshold);

XLATensorPtr nonzero(const XLATensorPtr& input);

XLATensorPtr norm(const XLATensorPtr& input, const std::optional<at::Scalar>& p,
                  std::optional<at::ScalarType> dtype, at::IntArrayRef dim,
                  bool keepdim);

XLATensorPtr normal(double mean, const XLATensorPtr& std);

XLATensorPtr normal(const XLATensorPtr& mean, double std);

XLATensorPtr normal(const XLATensorPtr& mean, const XLATensorPtr& std);

void normal_(XLATensorPtr& input, double mean, double std);

XLATensorPtr not_supported(std::string description, xla::Shape shape,
                           const torch::lazy::BackendDevice& device);

void optimization_barrier_(std::vector<XLATensorPtr>& tensors);

// Permute the dimensions of this tensor according to the given permutation.
XLATensorPtr permute(const XLATensorPtr& input, absl::Span<const int64_t> dims);

XLATensorPtr pow(
    const XLATensorPtr& input, const at::Scalar& exponent,
    std::optional<at::ScalarType> logical_element_type = std::nullopt);
XLATensorPtr pow(
    const XLATensorPtr& input, const XLATensorPtr& exponent,
    std::optional<at::ScalarType> logical_element_type = std::nullopt);
XLATensorPtr pow(
    const at::Scalar& input, const XLATensorPtr& exponent,
    std::optional<at::ScalarType> logical_element_type = std::nullopt);

XLATensorPtr prelu(const XLATensorPtr& input, const XLATensorPtr& weight);

std::tuple<XLATensorPtr, XLATensorPtr> prelu_backward(
    const XLATensorPtr& grad_out, const XLATensorPtr& input,
    const XLATensorPtr& weight);

XLATensorPtr prod(const XLATensorPtr& input, std::vector<int64_t> dimensions,
                  bool keep_reduced_dimensions,
                  std::optional<at::ScalarType> dtype);

void put_(XLATensorPtr& input, const XLATensorPtr& index,
          const XLATensorPtr& source, bool accumulate);

std::tuple<XLATensorPtr, XLATensorPtr> qr(const XLATensorPtr& input, bool some);

void random_(XLATensorPtr& input, int64_t from, int64_t to);

XLATensorPtr randperm(int64_t n, const torch::lazy::BackendDevice& device,
                      at::ScalarType scalar_type);

XLATensorPtr reflection_pad1d(const XLATensorPtr& input,
                              std::vector<int64_t> padding);

XLATensorPtr reflection_pad1d_backward(const XLATensorPtr& grad_output,
                                       const XLATensorPtr& input,
                                       std::vector<int64_t> padding);

XLATensorPtr reflection_pad2d(const XLATensorPtr& input,
                              std::vector<int64_t> padding);

XLATensorPtr reflection_pad2d_backward(const XLATensorPtr& grad_output,
                                       const XLATensorPtr& input,
                                       std::vector<int64_t> padding);

XLATensorPtr reflection_pad3d(const XLATensorPtr& input,
                              std::vector<int64_t> padding);

XLATensorPtr reflection_pad3d_backward(const XLATensorPtr& grad_output,
                                       const XLATensorPtr& input,
                                       std::vector<int64_t> padding);

XLATensorPtr remainder(const XLATensorPtr& input, const XLATensorPtr& other);
XLATensorPtr remainder(const XLATensorPtr& input, const at::Scalar& other);

XLATensorPtr replication_pad1d(const XLATensorPtr& input,
                               std::vector<int64_t> padding);
XLATensorPtr replication_pad1d_backward(const XLATensorPtr& grad_output,
                                        const XLATensorPtr& input,
                                        std::vector<int64_t> padding);

XLATensorPtr replication_pad2d(const XLATensorPtr& input,
                               std::vector<int64_t> padding);
XLATensorPtr replication_pad2d_backward(const XLATensorPtr& grad_output,
                                        const XLATensorPtr& input,
                                        std::vector<int64_t> padding);

XLATensorPtr replication_pad3d(const XLATensorPtr& input,
                               std::vector<int64_t> padding);
XLATensorPtr replication_pad3d_backward(const XLATensorPtr& grad_output,
                                        const XLATensorPtr& input,
                                        std::vector<int64_t> padding);

void resize_(XLATensorPtr& input, std::vector<int64_t> size);

XLATensorPtr roll(const XLATensorPtr& input, absl::Span<const int64_t> shifts,
                  absl::Span<const int64_t> dims);

XLATensorPtr rrelu_with_noise(const XLATensorPtr& input, XLATensorPtr& noise,
                              const at::Scalar& lower, const at::Scalar& upper,
                              bool training);

XLATensorPtr rrelu_with_noise_backward(const XLATensorPtr& grad_output,
                                       const XLATensorPtr& input,
                                       const XLATensorPtr& noise,
                                       const at::Scalar& lower,
                                       const at::Scalar& upper, bool training);

XLATensorPtr rsub(
    const XLATensorPtr& input, const XLATensorPtr& other,
    const at::Scalar& alpha,
    std::optional<at::ScalarType> logical_element_type = std::nullopt);
XLATensorPtr rsub(
    const XLATensorPtr& input, const at::Scalar& other, const at::Scalar& alpha,
    std::optional<at::ScalarType> logical_element_type = std::nullopt);

void copy_(XLATensorPtr& input, XLATensorPtr& src);

XLATensorPtr scatter(const XLATensorPtr& input, int64_t dim,
                     const XLATensorPtr& index, const XLATensorPtr& src);
XLATensorPtr scatter(const XLATensorPtr& input, int64_t dim,
                     const XLATensorPtr& index, const at::Scalar& value);

XLATensorPtr scatter_add(const XLATensorPtr& input, int64_t dim,
                         const XLATensorPtr& index, const XLATensorPtr& src);
XLATensorPtr scatter_add(const XLATensorPtr& input, int64_t dim,
                         const XLATensorPtr& index, const at::Scalar& value);

XLATensorPtr scatter_reduce(const XLATensorPtr& input, int64_t dim,
                            const XLATensorPtr& index, const XLATensorPtr& src,
                            c10::string_view reduce, bool include_self);

XLATensorPtr select(const XLATensorPtr& input, int64_t dim, int64_t index);

void selu_(XLATensorPtr& input);

XLATensorPtr silu(const XLATensorPtr& input);
XLATensorPtr silu_backward(XLATensorPtr& grad_output, XLATensorPtr& input);
XLATensorPtr sigmoid(const XLATensorPtr& input);
XLATensorPtr sigmoid_backward(const XLATensorPtr& grad_output,
                              const XLATensorPtr& output);

XLATensorPtr slice(const XLATensorPtr& input, int64_t dim, int64_t start,
                   int64_t end, int64_t step);

std::tuple<XLATensorPtr, XLATensorPtr> eigh(const XLATensorPtr& input,
                                            c10::string_view uplo);

std::tuple<XLATensorPtr, XLATensorPtr> slogdet(const XLATensorPtr& input);

// Computes a loss that uses a squared term if the absolute element-wise error
// falls below 1 and an L1 term otherwise.
XLATensorPtr smooth_l1_loss(const XLATensorPtr& input,
                            const XLATensorPtr& target, int64_t reduction,
                            double beta);

// Returns the gradient of the input of a smooth_l1_loss operation.
XLATensorPtr smooth_l1_loss_backward(const XLATensorPtr& grad_output,
                                     const XLATensorPtr& input,
                                     const XLATensorPtr& target,
                                     int64_t reduction, double beta);

XLATensorPtr softmax(const XLATensorPtr& input, int64_t dim,
                     std::optional<at::ScalarType> dtype);
XLATensorPtr softmax_backward(const XLATensorPtr& grad_output,
                              const XLATensorPtr& output, int64_t dim);

XLATensorPtr softplus(const XLATensorPtr& input, const at::Scalar& beta,
                      const at::Scalar& threshold);

XLATensorPtr softplus_backward(const XLATensorPtr& grad_output,
                               const XLATensorPtr& input,
                               const at::Scalar& beta,
                               const at::Scalar& threshold);

XLATensorPtr softshrink(const XLATensorPtr& input, const at::Scalar& lambda);
XLATensorPtr softshrink_backward(const XLATensorPtr& grad_out,
                                 const XLATensorPtr& input,
                                 const at::Scalar& lambda);

std::vector<XLATensorPtr> split(const XLATensorPtr& input, int64_t split_size,
                                int64_t dim);

std::vector<XLATensorPtr> split_with_sizes(const XLATensorPtr& input,
                                           std::vector<int64_t> split_size,
                                           int64_t dim);

// Squeeze out all trivial (size 1) dimensions.
XLATensorPtr squeeze(const XLATensorPtr& input);

// Squeeze out the specified dimension index, if trivial (size 1). Returns
// unchanged input otherwise.
XLATensorPtr squeeze(const XLATensorPtr& input, int64_t dim);

// Same as above, but with a tuple of dims.
XLATensorPtr squeeze(const XLATensorPtr& input, std::vector<int64_t> dims);

// In-place versions of the methods above.
void squeeze_(XLATensorPtr& input);
void squeeze_(XLATensorPtr& input, int64_t dim);

XLATensorPtr stack(absl::Span<const XLATensorPtr> tensors, int64_t dim);

XLATensorPtr std(const XLATensorPtr& input, std::vector<int64_t> dimensions,
                 bool keep_reduced_dimensions, double correction);

std::tuple<XLATensorPtr, XLATensorPtr> std_mean(const XLATensorPtr& input,
                                                std::vector<int64_t> dimensions,
                                                double correction,
                                                bool keep_reduced_dimensions);

XLATensorPtr sub(
    const XLATensorPtr& input, const XLATensorPtr& other,
    const at::Scalar& alpha,
    std::optional<at::ScalarType> logical_element_type = std::nullopt);
XLATensorPtr sub(
    const XLATensorPtr& input, const at::Scalar& other, const at::Scalar& alpha,
    std::optional<at::ScalarType> logical_element_type = std::nullopt);

XLATensorPtr sum(const XLATensorPtr& input, std::vector<int64_t> dimensions,
                 bool keep_reduced_dimensions,
                 std::optional<at::ScalarType> dtype);

std::tuple<XLATensorPtr, XLATensorPtr, XLATensorPtr> svd(
    const XLATensorPtr& input, bool some, bool compute_uv);

XLATensorPtr take(const XLATensorPtr& input, const XLATensorPtr& index);

XLATensorPtr tanh_backward(const XLATensorPtr& grad_output,
                           const XLATensorPtr& output);

XLATensorPtr threshold(const XLATensorPtr& input, float threshold, float value);

XLATensorPtr threshold_backward(const XLATensorPtr& grad_output,
                                const XLATensorPtr& input, float threshold);

XLATensorPtr to(XLATensorPtr& input,
                std::optional<torch::lazy::BackendDevice> device,
                std::optional<at::ScalarType> scalar_type);

std::tuple<XLATensorPtr, XLATensorPtr> topk(const XLATensorPtr& input,
                                            int64_t k, int64_t dim,
                                            bool largest, bool sorted,
                                            bool stable);

// Returns the sum of the elements of the diagonal of the input 2-D matrix.
XLATensorPtr trace(const XLATensorPtr& input);

// Swap given dimensions of the input.
XLATensorPtr transpose(const XLATensorPtr& input, int64_t dim0, int64_t dim1);

// In-place version of the method above.
void transpose_(XLATensorPtr& input, int64_t dim0, int64_t dim1);

std::tuple<XLATensorPtr, XLATensorPtr> triangular_solve(
    const XLATensorPtr& rhs, const XLATensorPtr& lhs, bool left_side,
    bool upper, bool transpose, bool unitriangular);

// Returns a tuple of all slices along a given dimension with that dimension
// removed.
std::vector<XLATensorPtr> unbind(const XLATensorPtr& input, int64_t dim);

void uniform_(XLATensorPtr& input, double from, double to);

// Insert a dimension of size one at the specified position.
XLATensorPtr unsqueeze(const XLATensorPtr& input, int64_t dim);

// In-place version of the method above.
void unsqueeze_(XLATensorPtr& input, int64_t dim);

XLATensorPtr upsample_bilinear2d(const XLATensorPtr& input,
                                 std::vector<int64_t> output_size,
                                 bool align_corners);

XLATensorPtr upsample_bilinear2d_backward(const XLATensorPtr& grad_output,
                                          std::vector<int64_t> output_size,
                                          std::vector<int64_t> input_size,
                                          bool align_corners);

XLATensorPtr upsample_nearest2d(const XLATensorPtr& input,
                                std::vector<int64_t> output_size);

XLATensorPtr upsample_nearest2d_backward(const XLATensorPtr& grad_output,
                                         std::vector<int64_t> output_size,
                                         std::vector<int64_t> input_size);

XLATensorPtr var(const XLATensorPtr& input, std::vector<int64_t> dimensions,
                 double correction, bool keep_reduced_dimensions);

std::tuple<XLATensorPtr, XLATensorPtr> var_mean(const XLATensorPtr& input,
                                                std::vector<int64_t> dimensions,
                                                double correction,
                                                bool keep_reduced_dimensions);

// Like reshape, but it returns a view into the original tensor.
XLATensorPtr view(const XLATensorPtr& input,
                  absl::Span<const int64_t> output_size);
XLATensorPtr view_symint(const XLATensorPtr& input,
                         at::SymIntArrayRef sym_size);

XLATensorPtr view_as_complex_copy(const XLATensorPtr& input);

XLATensorPtr view_as_real_copy(const XLATensorPtr& input);

void zero_(XLATensorPtr& input);

XLATensorPtr where(const XLATensorPtr& condition, const XLATensorPtr& input,
                   const XLATensorPtr& other);

}  // namespace tensor_methods
}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_TENSOR_METHODS_H_
