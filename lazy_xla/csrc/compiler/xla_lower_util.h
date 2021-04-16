#pragma once

#include "lazy_xla/csrc/compiler/xla_lowering_context.h"

namespace torch_lazy_tensors {
namespace compiler {

xla::XlaOp PadToSize(xla::XlaOp input, absl::Span<const xla::int64> size,
                     absl::optional<xla::XlaOp> pad_value = absl::nullopt);

using XlaOpCombiner = std::function<xla::XlaOp(xla::XlaOp, xla::XlaOp)>;

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

XlaOpVector BuildAmpForeachNonFiniteCheckAndUnscale(
    const XlaOpVector& inputs, const xla::XlaOp& found_inf_float,
    const xla::XlaOp& inv_scale);

XlaOpVector BuildAmpUpdateScale(const xla::XlaOp& growth_tracker,
                                const xla::XlaOp& current_scale,
                                const xla::XlaOp& found_inf_float,
                                double scale_growth_factor,
                                double scale_backoff_factor,
                                int scale_growth_interval);

}  // namespace compiler
}  // namespace torch_lazy_tensors
