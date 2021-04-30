#pragma once

#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "torch_xla/csrc/device.h"

namespace torch_xla {

xla::XlaOp PadToSize(xla::XlaOp input, absl::Span<const xla::int64> size,
                     absl::optional<xla::XlaOp> pad_value = absl::nullopt);

std::vector<xla::XlaOp> CreateKthValue(xla::XlaOp input, xla::int64 k,
                                       xla::int64 dim, bool keepdim);

std::vector<xla::XlaOp> CreateTopK(xla::XlaOp input, xla::int64 k,
                                   xla::int64 dim, bool largest, bool sorted);

xla::XlaOp CreateMatMul(xla::XlaOp lhs, xla::XlaOp rhs);

xla::XlaOp BuildGer(xla::XlaOp lhs, xla::XlaOp rhs);

xla::XlaOp BuildMatMul(xla::XlaOp lhs, xla::XlaOp rhs, xla::XlaOp bias);

xla::XlaOp BuildMatMulWithMultiplier(xla::XlaOp lhs, xla::XlaOp rhs,
                                     xla::XlaOp bias,
                                     xla::XlaOp product_multiplier,
                                     xla::XlaOp bias_multiplier);

xla::XlaOp BuildDot(xla::XlaOp lhs, xla::XlaOp rhs);

xla::XlaOp BuildBernoulli(xla::XlaOp probability, xla::XlaOp seed,
                          xla::PrimitiveType type);

xla::XlaOp BuildExponential(xla::XlaOp lambda, xla::XlaOp seed,
                            xla::PrimitiveType type);

xla::XlaOp BuildDropout(xla::XlaOp input, float probability, xla::XlaOp seed);

std::vector<xla::XlaOp> CreateBroadcastTensors(
    absl::Span<const xla::XlaOp> operands);

// Similar to tf.gather_nd, used to implement advanced indexing.
xla::XlaOp CreateIndex(xla::XlaOp input, xla::XlaOp indices,
                       xla::int64 start_dim);

// Similar to tf.scatter_nd, used to implement advanced indexing updates.
xla::XlaOp CreateIndexUpdate(
    xla::XlaOp buffer, xla::XlaOp indices, xla::int64 start_dim,
    xla::XlaOp updates,
    const std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>& combiner);

xla::XlaOp CreateIndexAdd(xla::XlaOp buffer, xla::int64 dim, xla::XlaOp index,
                          xla::XlaOp value);

xla::XlaOp CreateIndexCopy(xla::XlaOp buffer, xla::int64 dim, xla::XlaOp index,
                           xla::XlaOp value);

xla::XlaOp CreateIndexFill(xla::XlaOp buffer, xla::int64 dim, xla::XlaOp index,
                           xla::XlaOp values);

using XlaOpCombiner = std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>;

XlaOpCombiner NumericAddCombiner();

struct ScatterOptions {
  explicit ScatterOptions(XlaOpCombiner combiner)
      : combiner(std::move(combiner)) {}

  XlaOpCombiner combiner;
  absl::optional<xla::XlaOp> init_value;
  bool indices_are_unique = true;
};

xla::XlaOp CreateScatter(const Device& device, xla::XlaOp input,
                         xla::XlaOp index, xla::XlaOp source, xla::int64 dim,
                         const ScatterOptions& options);

xla::XlaOp CreatePut(const Device& device, xla::XlaOp input, xla::XlaOp index,
                     xla::XlaOp source, bool accumulate);

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

}  // namespace torch_xla
