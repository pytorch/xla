#include "torch_xla/csrc/data_ops.h"

#include <algorithm>
#include <functional>
#include <numeric>

#include "absl/strings/str_join.h"
#include "lazy_tensors/compiler/xla/xla_client/sys_util.h"
#include "lazy_tensors/compiler/xla/xla_client/util.h"
#include "lazy_xla/csrc/compiler/convert_ops.h"
#include "lazy_xla/csrc/compiler/debug_macros.h"
#include "lazy_xla/csrc/compiler/helpers.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "torch_xla/csrc/reduction.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_lazy_tensors {
namespace {

bool IsSparseGather(const xla::Shape& input_shape,
                    const xla::Shape& index_shape, xla::int64 dim) {
  static int dense_gather_factor =
      lazy_tensors::sys_util::GetEnvInt("XLA_DENSE_GATHER_FACTOR", 100);
  xla::int64 input_elements = xla::ShapeUtil::ElementsIn(input_shape);
  xla::int64 index_elements = xla::ShapeUtil::ElementsIn(index_shape);
  // Simple heuristic. Might need fine tuning.
  return index_elements < input_elements / dense_gather_factor;
}

}  // namespace

bool IsSparseGather(xla::XlaOp input, xla::XlaOp index, xla::int64 dim) {
  return IsSparseGather(compiler::XlaHelpers::ShapeOfXlaOp(input),
                        compiler::XlaHelpers::ShapeOfXlaOp(index), dim);
}

xla::XlaOp BuildView(xla::XlaOp input,
                     absl::Span<const xla::int64> output_sizes) {
  const xla::Shape& input_shape = compiler::XlaHelpers::ShapeOfXlaOp(input);
  const auto complete_output_sizes =
      GetCompleteShape(output_sizes, input_shape.dimensions());
  return compiler::XlaHelpers::DynamicReshape(input, complete_output_sizes);
}

xla::XlaOp SqueezeTrivialDimension(xla::XlaOp input, xla::int64 dim) {
  const xla::Shape& input_shape = compiler::XlaHelpers::ShapeOfXlaOp(input);
  XLA_CHECK_LT(dim, input_shape.rank());
  if (input_shape.dimensions(dim) != 1) {
    return input;
  }
  auto output_sizes = BuildSqueezedDimensions(input_shape.dimensions(), dim);
  return compiler::XlaHelpers::DynamicReshape(input, output_sizes);
}

xla::XlaOp SqueezeAllTrivialDimensions(xla::XlaOp input) {
  const xla::Shape& input_shape = compiler::XlaHelpers::ShapeOfXlaOp(input);
  auto output_sizes =
      BuildSqueezedDimensions(input_shape.dimensions(), /*squeeze_dim=*/-1);
  return compiler::XlaHelpers::DynamicReshape(input, output_sizes);
}

xla::XlaOp BuildExpand(xla::XlaOp input,
                       absl::Span<const xla::int64> output_sizes) {
  auto input_sizes = compiler::XlaHelpers::SizesOfXlaOp(input);
  // Adjust the rank of the input to match the rank of the output.
  XLA_CHECK_LE(input_sizes.size(), output_sizes.size());
  input_sizes.insert(input_sizes.begin(),
                     output_sizes.size() - input_sizes.size(), 1);
  xla::XlaOp implicit_reshape =
      compiler::XlaHelpers::DynamicReshape(input, input_sizes);
  return xla::BroadcastInDim(
      implicit_reshape, output_sizes,
      lazy_tensors::util::Iota<xla::int64>(output_sizes.size()));
}

xla::XlaOp BuildUnsqueeze(xla::XlaOp input, xla::int64 dim) {
  auto dimensions =
      BuildUnsqueezeDimensions(compiler::XlaHelpers::SizesOfXlaOp(input), dim);
  return compiler::XlaHelpers::DynamicReshape(input, dimensions);
}

xla::XlaOp BuildCat(absl::Span<const xla::XlaOp> inputs, xla::int64 dim) {
  XLA_CHECK_GT(inputs.size(), 0);
  return xla::ConcatInDim(inputs[0].builder(), inputs, dim);
}

std::vector<xla::XlaOp> BuildSplit(xla::XlaOp input,
                                   absl::Span<const xla::int64> split_sizes,
                                   xla::int64 dim) {
  const auto input_sizes = compiler::XlaHelpers::SizesOfXlaOp(input);
  xla::int64 dim_size = input_sizes.at(dim);
  xla::int64 index = 0;
  std::vector<xla::XlaOp> splits;
  for (auto size : split_sizes) {
    if (index + size > dim_size) {
      break;
    }
    splits.emplace_back(xla::SliceInDim(input, index, index + size, 1, dim));
    index += size;
  }
  return splits;
}

xla::XlaOp BuildUpdateSlice(xla::XlaOp input, xla::XlaOp source,
                            absl::Span<const xla::int64> base_indices) {
  const xla::Shape& input_shape = compiler::XlaHelpers::ShapeOfXlaOp(input);
  const xla::Shape& source_shape = compiler::XlaHelpers::ShapeOfXlaOp(source);
  xla::XlaOp update_source = source;
  if (source_shape.element_type() != input_shape.element_type()) {
    update_source = ConvertTo(source, source_shape.element_type(),
                              input_shape.element_type(), /*device=*/nullptr);
  }
  xla::XlaOp reshaped_source =
      compiler::XlaHelpers::ReshapeToRank(update_source, input_shape.rank());
  std::vector<xla::XlaOp> start_indices;
  for (auto index : base_indices) {
    start_indices.push_back(
        compiler::XlaHelpers::ScalarValue<xla::int64>(index, input.builder()));
  }
  return xla::DynamicUpdateSlice(input, reshaped_source, start_indices);
}

xla::XlaOp BuildSlice(xla::XlaOp input,
                      absl::Span<const xla::int64> base_indices,
                      absl::Span<const xla::int64> sizes) {
  XLA_CHECK_EQ(base_indices.size(), sizes.size());
  std::vector<xla::int64> limit_indices(base_indices.begin(),
                                        base_indices.end());
  std::transform(limit_indices.begin(), limit_indices.end(), sizes.begin(),
                 limit_indices.begin(), std::plus<xla::int64>());
  std::vector<xla::int64> strides(base_indices.size(), 1);
  return xla::Slice(input, base_indices, limit_indices, strides);
}

xla::XlaOp BoundIndices(xla::XlaOp index, xla::XlaOp max_index) {
  const xla::Shape& index_shape = compiler::XlaHelpers::ShapeOfXlaOp(index);
  return xla::Select(
      xla::Ge(index, xla::Zero(index.builder(), index_shape.element_type())),
      index, index + max_index);
}

xla::XlaOp BuildResize(xla::XlaOp input, absl::Span<const xla::int64> size) {
  xla::Shape input_shape;
  xla::XlaOp r1_input = compiler::XlaHelpers::Flatten(input, &input_shape);
  xla::int64 num_elements = xla::ShapeUtil::ElementsIn(input_shape);
  xla::int64 new_num_elements = lazy_tensors::util::Multiply<xla::int64>(size);
  xla::XlaOp resized_input = input;
  if (num_elements > new_num_elements) {
    resized_input = xla::SliceInDim(r1_input, 0, new_num_elements, 1, 0);
  } else if (new_num_elements > num_elements) {
    xla::XlaOp zero = xla::Zero(input.builder(), input_shape.element_type());
    xla::PaddingConfig padding_config;
    auto* dims = padding_config.add_dimensions();
    dims->set_edge_padding_low(0);
    dims->set_interior_padding(0);
    dims->set_edge_padding_high(new_num_elements - num_elements);
    resized_input = xla::Pad(r1_input, zero, padding_config);
  }
  return compiler::XlaHelpers::DynamicReshape(resized_input, size);
}

xla::XlaOp BuildUnselect(xla::XlaOp target, xla::XlaOp source, xla::int64 dim,
                         xla::int64 start, xla::int64 end, xla::int64 stride) {
  const xla::Shape& target_shape = compiler::XlaHelpers::ShapeOfXlaOp(target);
  const xla::Shape& source_shape = compiler::XlaHelpers::ShapeOfXlaOp(source);
  if (target_shape.dimensions(dim) == source_shape.dimensions(dim)) {
    // Shortcut for unselects which are fully covering selects.
    XLA_CHECK_EQ(start, 0);
    XLA_CHECK_EQ(stride, 1);
    XLA_CHECK_EQ(end, target_shape.dimensions(dim));
    return source;
  }

  xla::PrimitiveType pred_type =
      compiler::XlaHelpers::XlaPrimitiveType(GetDevicePrimitiveType(
          lazy_tensors::PrimitiveType::PRED, /*device=*/nullptr));
  xla::XlaOp source_true = compiler::XlaHelpers::ScalarBroadcast(
      1, pred_type, source_shape.dimensions(), source.builder());
  xla::XlaOp pred_zero = xla::Zero(target.builder(), pred_type);
  xla::XlaOp zero = xla::Zero(target.builder(), target_shape.element_type());
  xla::PaddingConfig padding_config;
  for (xla::int64 i = 0; i < target_shape.rank(); ++i) {
    auto* dims = padding_config.add_dimensions();
    if (i == dim) {
      dims->set_edge_padding_low(start);
      dims->set_interior_padding(stride - 1);

      xla::int64 size = start + source_shape.dimensions(i) +
                        (source_shape.dimensions(i) - 1) * (stride - 1);
      dims->set_edge_padding_high(target_shape.dimensions(i) - size);
    } else {
      XLA_CHECK_EQ(target_shape.dimensions(i), source_shape.dimensions(i))
          << target_shape << " vs. " << source_shape;
      dims->set_edge_padding_low(0);
      dims->set_interior_padding(0);
      dims->set_edge_padding_high(0);
    }
  }
  xla::XlaOp padded_source = xla::Pad(source, zero, padding_config);
  xla::XlaOp mask = xla::Pad(source_true, pred_zero, padding_config);
  return xla::Select(mask, padded_source, target);
}

xla::XlaOp PadInDim(xla::XlaOp input, xla::int64 dim, xla::int64 pad_lo,
                    xla::int64 pad_hi, const xla::XlaOp* pad_value) {
  const xla::Shape& input_shape = compiler::XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp zero;
  if (pad_value == nullptr) {
    zero = xla::Zero(input.builder(), input_shape.element_type());
    pad_value = &zero;
  }
  xla::PaddingConfig padding_config;
  for (xla::int64 i = 0; i < input_shape.rank(); ++i) {
    auto* dims = padding_config.add_dimensions();
    dims->set_interior_padding(0);
    if (i == dim) {
      dims->set_edge_padding_low(pad_lo);
      dims->set_edge_padding_high(pad_hi);
    } else {
      dims->set_edge_padding_low(0);
      dims->set_edge_padding_high(0);
    }
  }
  return xla::Pad(input, *pad_value, padding_config);
}

}  // namespace torch_lazy_tensors
