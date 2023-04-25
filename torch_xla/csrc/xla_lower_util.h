#ifndef XLA_TORCH_XLA_CSRC_XLA_LOWER_UTIL_H_
#define XLA_TORCH_XLA_CSRC_XLA_LOWER_UTIL_H_

#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "torch_xla/csrc/device.h"

namespace torch_xla {

xla::XlaOp PadToSize(xla::XlaOp input, absl::Span<const int64_t> size,
                     absl::optional<xla::XlaOp> pad_value = absl::nullopt);

std::vector<xla::XlaOp> CreateKthValue(xla::XlaOp input, int64_t k, int64_t dim,
                                       bool keepdim);

std::vector<xla::XlaOp> CreateTopK(xla::XlaOp input, int64_t k, int64_t dim,
                                   bool largest, bool stable);

xla::XlaOp CreateMatMul(xla::XlaOp lhs, xla::XlaOp rhs);

xla::XlaOp BuildMatMul(xla::XlaOp lhs, xla::XlaOp rhs, xla::XlaOp bias);

xla::XlaOp BuildMatMulWithMultiplier(xla::XlaOp lhs, xla::XlaOp rhs,
                                     xla::XlaOp bias,
                                     xla::XlaOp product_multiplier,
                                     xla::XlaOp bias_multiplier);

xla::XlaOp BuildDot(xla::XlaOp lhs, xla::XlaOp rhs);

xla::XlaOp BuildBernoulli(xla::XlaOp probability, xla::XlaOp seed,
                          xla::PrimitiveType type);

xla::XlaOp BuildMultinomial(xla::XlaOp input, int64_t num_samples,
                            bool replacement, xla::XlaOp seed);

xla::XlaOp BuildExponential(xla::XlaOp lambda, xla::XlaOp seed,
                            xla::PrimitiveType type);

xla::XlaOp BuildDropout(xla::XlaOp input, float probability, xla::XlaOp seed);

xla::XlaOp BuildSigmoidBackward(xla::XlaOp grad_output, xla::XlaOp output,
                                xla::XlaOp scalar_1);

std::vector<xla::XlaOp> CreateBroadcastTensors(
    absl::Span<const xla::XlaOp> operands);

// Similar to tf.gather_nd, used to implement advanced indexing.
xla::XlaOp CreateIndex(xla::XlaOp input, xla::XlaOp indices, int64_t start_dim);

// Similar to tf.scatter_nd, used to implement advanced indexing updates.
xla::XlaOp CreateIndexUpdate(
    xla::XlaOp buffer, xla::XlaOp indices, int64_t start_dim,
    xla::XlaOp updates,
    const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>& combiner);

xla::XlaOp CreateIndexAdd(xla::XlaOp buffer, int64_t dim, xla::XlaOp index,
                          xla::XlaOp value);

xla::XlaOp CreateIndexCopy(xla::XlaOp buffer, int64_t dim, xla::XlaOp index,
                           xla::XlaOp value);

xla::XlaOp CreateIndexFill(xla::XlaOp buffer, int64_t dim, xla::XlaOp index,
                           xla::XlaOp values);

using XlaOpCombiner = std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>;

XlaOpCombiner NumericAddCombiner();

XlaOpCombiner NumericMulCombiner();

XlaOpCombiner NumericMinCombiner();

XlaOpCombiner NumericMaxCombiner();

struct ScatterOptions {
  explicit ScatterOptions(XlaOpCombiner combiner)
      : combiner(std::move(combiner)) {}

  XlaOpCombiner combiner;
  absl::optional<xla::XlaOp> init_value;
  bool indices_are_unique = true;
};

xla::XlaOp CreateScatter(const torch::lazy::BackendDevice& device,
                         xla::XlaOp input, xla::XlaOp index, xla::XlaOp source,
                         int64_t dim, const ScatterOptions& options);

xla::XlaOp CreatePut(const torch::lazy::BackendDevice& device, xla::XlaOp input,
                     xla::XlaOp index, xla::XlaOp source, bool accumulate);

xla::XlaOp BuildLinspace(const torch::lazy::BackendDevice& device,
                         xla::XlaOp start, xla::XlaOp end, int64_t steps);

std::vector<xla::XlaOp> BuildNonZero(xla::XlaOp input);

std::vector<xla::XlaOp> BuildMaskedSelect(xla::XlaOp input, xla::XlaOp mask);

xla::XlaOp BuildMaskedScatter(xla::XlaOp input, xla::XlaOp mask,
                              xla::XlaOp source);

std::vector<xla::XlaOp> BuildAmpForeachNonFiniteCheckAndUnscale(
    const std::vector<xla::XlaOp>& inputs, const xla::XlaOp& found_inf_float,
    const xla::XlaOp& inv_scale);

std::vector<xla::XlaOp> BuildAmpUpdateScale(const xla::XlaOp& current_scale,
                                            const xla::XlaOp& growth_tracker,
                                            const xla::XlaOp& found_inf,
                                            double scale_growth_factor,
                                            double scale_backoff_factor,
                                            int scale_growth_interval);

std::vector<xla::XlaOp> BuildSgdOptimizerStep(
    const xla::XlaOp& found_inf, const xla::XlaOp& step,
    const xla::XlaOp& param, const xla::XlaOp& buf, const xla::XlaOp& d_p,
    const xla::XlaOp& weight_decay, const xla::XlaOp& momentum,
    const xla::XlaOp& lr, const xla::XlaOp& dampening, bool use_weight_decay,
    bool use_momentum, bool use_nesterov);

std::vector<xla::XlaOp> BuildAdamOptimizerStep(
    const xla::XlaOp& found_inf, const xla::XlaOp& step,
    const xla::XlaOp& param, const xla::XlaOp& grad, const xla::XlaOp& exp_avg,
    const xla::XlaOp& exp_avg_sq, const xla::XlaOp& max_exp_avg_sq,
    const xla::XlaOp& beta1, const xla::XlaOp& beta2, const xla::XlaOp& lr,
    const xla::XlaOp& weight_decay, const xla::XlaOp& eps,
    bool use_weight_decay, bool use_amsgrad, bool use_adamw);

xla::XlaOp BuildXLogY(xla::XlaOp input, xla::XlaOp other);

xla::XlaOp BuildRoll(xla::XlaOp input, absl::Span<const int64_t> shifts,
                     absl::Span<const int64_t> dims);

xla::XlaOp BuildAddcdiv(xla::XlaOp input, xla::XlaOp t1, xla::XlaOp t2,
                        xla::XlaOp val);

xla::XlaOp BuildAddcmul(xla::XlaOp input, xla::XlaOp t1, xla::XlaOp t2,
                        xla::XlaOp val);

xla::XlaOp BuildCdistForward(xla::XlaOp x1, xla::XlaOp x2, xla::XlaOp p,
                             bool use_hamming, bool use_chebyshev);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_XLA_LOWER_UTIL_H_