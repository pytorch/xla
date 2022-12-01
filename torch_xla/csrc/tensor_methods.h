#pragma once

#include "torch_xla/csrc/tensor.h"

namespace torch_xla {

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

static XLATensorPtr alias(const XLATensorPtr& input);

static XLATensorPtr amax(const XLATensorPtr& input,
                        std::vector<int64_t> dimensions,
                        bool keep_reduced_dimensions);

static XLATensorPtr amin(const XLATensorPtr& input,
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
static void bernoulli_(XLATensorPtr& input, const XLATensorPtr& probability);

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

static XLATensorPtr celu(const XLATensorPtr& input, const at::Scalar& alpha);
static void celu_(XLATensorPtr& input, const at::Scalar& alpha);

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

static std::tuple<XLATensorPtr, XLATensorPtr> einsum_backward(
    const XLATensorPtr& grad_output,
    const absl::Span<const XLATensorPtr> tensors,
    const std::string& equation);

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

static XLATensorPtr expand_symint(const XLATensorPtr& input,
                                c10::SymIntArrayRef sym_size);

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

static XLATensorPtr log_softmax(const XLATensorPtr& input, int64_t dim,
                                c10::optional<at::ScalarType> dtype,
                                std::vector<torch::lazy::Shape>&& shapes);

static XLATensorPtr log_softmax_backward(const XLATensorPtr& grad_output,
                                        const XLATensorPtr& output,
                                        int64_t dim);

static XLATensorPtr log1p(const XLATensorPtr& input);
static void log1p_(XLATensorPtr& input);

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

}  // namespace torch_xla
