#include "torch_xla/csrc/data_ops.h"

#include <torch/csrc/lazy/core/tensor_util.h>
#include <torch/csrc/lazy/core/util.h>

#include <algorithm>
#include <functional>
#include <numeric>

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/dtype.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/reduction.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/runtime/util.h"
#include "torch_xla/csrc/shape_helper.h"
#include "torch_xla/csrc/tensor_util.h"
#include "xla/client/lib/constants.h"
#include "xla/client/lib/slicing.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace torch_xla {
namespace {

bool IsSparseGather(const xla::Shape& input_shape,
                    const xla::Shape& index_shape, int64_t dim) {
  // Conservative sparsity check for multi-platform support
  // to avoid gather on a single float on TPU.
  XlaDeviceType hw_type =
      static_cast<XlaDeviceType>(bridge::GetCurrentDevice().type());
  if (CheckTpuDevice(hw_type) || CheckNeuronDevice(hw_type)) {
    // XLA_DENSE_GATHER_FACTOR can be used to finely control the
    // sparsity check.
    static int dense_gather_factor =
        runtime::sys_util::GetEnvInt("XLA_DENSE_GATHER_FACTOR", 8192);
    int64_t input_elements = input_shape.dimensions()[dim];
    // Use a very conservative check so that we run dense gather
    // most of the time on TPU.
    return input_elements > dense_gather_factor * 10;
  }
  // Use sparse gather for non-TPU platforms.
  return true;
}

}  // namespace

bool IsSparseGather(xla::XlaOp input, xla::XlaOp index, int64_t dim) {
  return IsSparseGather(ShapeHelper::ShapeOfXlaOp(input),
                        ShapeHelper::ShapeOfXlaOp(index), dim);
}

std::vector<int64_t> GetCompleteShape(absl::Span<const int64_t> output_sizes,
                                      absl::Span<const int64_t> input_sizes) {
  std::optional<size_t> incomplete_dim;
  int64_t incomplete_element_count = 1;
  for (size_t dim = 0; dim < output_sizes.size(); ++dim) {
    int64_t dim_size = output_sizes[dim];
    if (dim_size < 0) {
      XLA_CHECK(!incomplete_dim)
          << "More than one incomplete dimension found: " << *incomplete_dim
          << " and " << dim;
      incomplete_dim = dim;
    } else {
      incomplete_element_count *= dim_size;
    }
  }
  int64_t total_element_count = runtime::util::Multiply<int64_t>(input_sizes);
  if (!incomplete_dim) {
    XLA_CHECK_EQ(total_element_count,
                 runtime::util::Multiply<int64_t>(output_sizes))
        << "(" << absl::StrJoin(output_sizes, ", ") << ") vs. ("
        << absl::StrJoin(input_sizes, ", ") << ")";
    return torch::lazy::ToVector<int64_t>(output_sizes);
  }
  XLA_CHECK_GT(incomplete_element_count, 0)
      << "Cannot reshape tensor of 0 elements into shape "
      << "(" << absl::StrJoin(output_sizes, ", ")
      << ") because the unspecified dimension size -1 can be any value";
  XLA_CHECK_EQ(total_element_count % incomplete_element_count, 0)
      << "(" << absl::StrJoin(output_sizes, ", ") << ") vs. ("
      << absl::StrJoin(input_sizes, ", ") << ")";
  std::vector<int64_t> complete_output_sizes =
      torch::lazy::ToVector<int64_t>(output_sizes);
  complete_output_sizes[*incomplete_dim] =
      total_element_count / incomplete_element_count;
  return complete_output_sizes;
}

xla::XlaOp BuildView(xla::XlaOp input, absl::Span<const int64_t> output_sizes) {
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  const auto complete_output_sizes =
      GetCompleteShape(output_sizes, input_shape.dimensions());
  return XlaHelpers::DynamicReshape(input, complete_output_sizes);
}

xla::XlaOp BuildUnboundedDynamicView(
    xla::XlaOp input, const xla::Shape& input_shape,
    const absl::Span<const int64_t>& output_sizes) {
  // Only Support BS is dynamic now.
  const absl::Span<const int64_t> input_dims = input_shape.dimensions();
  XLA_CHECK(std::count(input_dims.cbegin(), input_dims.cend(),
                       xla::Shape::kUnboundedSize) == 1 &&
            input_shape.is_unbounded_dynamic_dimension(0))
      << "Only BS of the input to view op can be unbounded dynamic.";

  XLA_CHECK(std::accumulate(input_dims.cbegin() + 1, input_dims.cend(), 1,
                            std::multiplies<int64_t>()) ==
            std::accumulate(output_sizes.cbegin() + 1, output_sizes.cend(), 1,
                            std::multiplies<int64_t>()))
      << "Dimensions of view input and output don't match.";

  const int src_index = 0;
  const int target_index = 0;
  xla::XlaOp dynamic_dim =
      xla::Reshape(xla::GetDimensionSize(input, src_index), {1});

  std::vector<xla::XlaOp> concat_ops;
  concat_ops.push_back(dynamic_dim);
  std::vector<int32_t> static_input_dims_vec(output_sizes.begin() + 1,
                                             output_sizes.end());
  concat_ops.push_back(xla::ConstantR1(
      input.builder(), absl::Span<const int32_t>(static_input_dims_vec)));
  xla::XlaOp final_broadcast_dimensions =
      xla::ConcatInDim(input.builder(), absl::Span<xla::XlaOp>(concat_ops), 0);

  // Final shape
  std::vector<int64_t> output_sizes_vec(output_sizes.begin(),
                                        output_sizes.end());
  output_sizes_vec[target_index] = xla::Shape::kUnboundedSize;
  std::vector<bool> output_dynamic(output_sizes_vec.size(), false);
  output_dynamic[target_index] = true;
  xla::Shape final_shape = xla::ShapeUtil::MakeShape(
      input_shape.element_type(), output_sizes_vec, output_dynamic);

  xla::XlaOp result =
      xla::CustomCall(input.builder(), "mhlo.dynamic_reshape",
                      {input, final_broadcast_dimensions}, final_shape);
  return result;
}

xla::XlaOp SetDimensionSizes(xla::XlaOp input,
                             absl::Span<const xla::XlaOp> symbolic_output_sizes,
                             std::vector<bool> dynamic_dims) {
  size_t current_output_size_index = 0;
  size_t symbolic_output_sizes_len = symbolic_output_sizes.size();
  for (size_t i = 0; i < dynamic_dims.size(); i++) {
    if (dynamic_dims[i]) {
      // Dimension i is dynamic
      XLA_CHECK_LT(current_output_size_index, symbolic_output_sizes_len);
      input = xla::SetDimensionSize(
          input, symbolic_output_sizes[current_output_size_index++], i);
    }
  }
  // Number of symbolic_output_sizes should equal to number of dynamic
  // dimensions.
  XLA_CHECK_EQ(current_output_size_index, symbolic_output_sizes_len);
  return input;
}

xla::XlaOp SqueezeTrivialDimension(xla::XlaOp input, int64_t dim) {
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  XLA_CHECK_LT(dim, input_shape.rank());
  if (input_shape.dimensions(dim) != 1) {
    return input;
  }
  auto output_sizes = BuildSqueezedDimensions(input_shape.dimensions(), dim);
  return XlaHelpers::DynamicReshape(input, output_sizes);
}

xla::XlaOp SqueezeAllTrivialDimensions(xla::XlaOp input) {
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  auto output_sizes =
      BuildSqueezedDimensions(input_shape.dimensions(), /*squeeze_dim=*/-1);
  return XlaHelpers::DynamicReshape(input, output_sizes);
}

xla::XlaOp BuildExpand(xla::XlaOp input,
                       absl::Span<const int64_t> output_sizes) {
  auto input_sizes = XlaHelpers::SizesOfXlaOp(input);
  // Adjust the rank of the input to match the rank of the output.
  XLA_CHECK_LE(input_sizes.size(), output_sizes.size());
  input_sizes.insert(input_sizes.begin(),
                     output_sizes.size() - input_sizes.size(), 1);
  xla::XlaOp implicit_reshape = XlaHelpers::DynamicReshape(input, input_sizes);
  return xla::BroadcastInDim(implicit_reshape, output_sizes,
                             torch::lazy::Iota<int64_t>(output_sizes.size()));
}

xla::XlaOp BuildMaskedFillScalar(xla::XlaOp input, xla::XlaOp mask,
                                 xla::XlaOp scalar) {
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  const xla::Shape& mask_shape = ShapeHelper::ShapeOfXlaOp(mask);

  if (!xla::ShapeUtil::Compatible(input_shape, mask_shape)) {
    xla::Shape shape = XlaHelpers::GetPromotedShape(input_shape, mask_shape);
    input = BuildExpand(input, shape.dimensions());
    mask = BuildExpand(mask, shape.dimensions());
  }

  xla::XlaOp zero = xla::Zero(mask.builder(), XlaHelpers::TypeOfXlaOp(mask));
  xla::XlaOp mask_pred = xla::Ne(mask, zero);
  xla::XlaOp update_scalar =
      ConvertTo(scalar, ShapeHelper::ShapeOfXlaOp(scalar).element_type(),
                ShapeHelper::ShapeOfXlaOp(input).element_type());
  return xla::Select(mask_pred, update_scalar, input);
}

std::vector<int64_t> BuildSqueezedDimensions(
    absl::Span<const int64_t> dimensions, int64_t squeeze_dim) {
  std::vector<int64_t> squeeze_dims({squeeze_dim});
  return BuildSqueezedDimensions(dimensions, squeeze_dims);
}

std::vector<int64_t> BuildSqueezedDimensions(
    absl::Span<const int64_t> dimensions, std::vector<int64_t>& squeeze_dims) {
  std::sort(squeeze_dims.begin(), squeeze_dims.end());
  std::vector<int64_t> output_dimensions;
  size_t i = 0;
  for (size_t j = 0; j < dimensions.size(); j++) {
    auto dim = dimensions[j];
    if (squeeze_dims.size() == 1 && squeeze_dims[0] == -1) {
      // Special case where squeeze_dims = {-1}.
      if (dim != 1) {
        output_dimensions.push_back(dim);
      }
      continue;
    }
    if (i == squeeze_dims.size() || j < squeeze_dims[i]) {
      output_dimensions.push_back(dim);
      continue;
    }
    // Checks to see if we need to squeeze the dim or not.
    if (dim != 1) {
      output_dimensions.push_back(dim);
    }
    i++;
  }
  return output_dimensions;
}

std::vector<int64_t> BuildUnsqueezeDimensions(
    absl::Span<const int64_t> dimensions, int64_t dim) {
  XLA_CHECK_LE(dim, dimensions.size());
  auto unsqueeze_dimensions = torch::lazy::ToVector<int64_t>(dimensions);
  unsqueeze_dimensions.insert(unsqueeze_dimensions.begin() + dim, 1);
  return unsqueeze_dimensions;
}

xla::XlaOp BuildUnsqueeze(xla::XlaOp input, int64_t dim) {
  auto dimensions =
      BuildUnsqueezeDimensions(XlaHelpers::SizesOfXlaOp(input), dim);
  return XlaHelpers::DynamicReshape(input, dimensions);
}

xla::XlaOp BuildStack(absl::Span<const xla::XlaOp> inputs, int64_t dim) {
  // Reshape inputs along the dim axis.
  XLA_CHECK_GT(inputs.size(), 0);
  std::vector<xla::XlaOp> reshaped_inputs;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const xla::XlaOp& input = inputs[i];
    const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(inputs[i]);
    const std::vector<int64_t> input_sizes =
        XlaHelpers::SizesOfXlaOp(inputs[i]);
    std::vector<int64_t> output_sizes = input_sizes;
    output_sizes.insert(output_sizes.begin() + dim, 1);
    reshaped_inputs.push_back(XlaHelpers::DynamicReshape(input, output_sizes));
  }
  return xla::ConcatInDim(inputs[0].builder(), reshaped_inputs, dim);
}

xla::XlaOp BuildCat(absl::Span<const xla::XlaOp> inputs, int64_t dim,
                    at::ScalarType dtype) {
  XLA_CHECK_GT(inputs.size(), 0);
  std::vector<xla::XlaOp> casted_inputs;
  for (const auto& op : inputs) {
    casted_inputs.push_back(CastToScalarType(op, dtype));
  }
  return xla::ConcatInDim(inputs[0].builder(), casted_inputs, dim);
}

xla::XlaOp BuildRepeat(xla::XlaOp input, absl::Span<const int64_t> repeats) {
  const auto input_sizes = XlaHelpers::SizesOfXlaOp(input);
  XLA_CHECK_GE(repeats.size(), input_sizes.size())
      << "Number of dimensions of repeat dims can not be smaller than number "
         "of dimensions of tensor";
  size_t broadcast_dims = repeats.size() - input_sizes.size();
  xla::XlaOp repeated = input;
  for (size_t dim = 0; dim < input_sizes.size(); ++dim) {
    std::vector<xla::XlaOp> repeated_inputs(repeats[broadcast_dims + dim],
                                            repeated);
    repeated = xla::ConcatInDim(input.builder(), repeated_inputs, dim);
  }
  if (repeats.size() > input_sizes.size()) {
    std::vector<int64_t> remaining_repeats(repeats.begin(),
                                           repeats.begin() + broadcast_dims);
    repeated = xla::Broadcast(repeated, remaining_repeats);
  }
  return repeated;
}

size_t ComputeSplitCount(int64_t dim_size,
                         absl::Span<const int64_t> split_sizes) {
  size_t count = 0;
  for (auto size : split_sizes) {
    if (size > dim_size) {
      break;
    }
    dim_size -= size;
    ++count;
  }
  return count;
}

std::vector<xla::XlaOp> BuildSplit(xla::XlaOp input,
                                   absl::Span<const int64_t> split_sizes,
                                   int64_t dim) {
  const auto input_sizes = XlaHelpers::SizesOfXlaOp(input);
  int64_t dim_size = input_sizes.at(dim);
  int64_t index = 0;
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
                            absl::Span<const int64_t> base_indices) {
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  const xla::Shape& source_shape = ShapeHelper::ShapeOfXlaOp(source);
  xla::XlaOp update_source = source;
  if (source_shape.element_type() != input_shape.element_type()) {
    update_source = ConvertTo(source, source_shape.element_type(),
                              input_shape.element_type());
  }
  xla::XlaOp reshaped_source =
      XlaHelpers::ReshapeToRank(update_source, input_shape.rank());
  std::vector<xla::XlaOp> start_indices;
  for (auto index : base_indices) {
    start_indices.push_back(
        XlaHelpers::ScalarValue<int64_t>(index, input.builder()));
  }
  return xla::DynamicUpdateSlice(input, reshaped_source, start_indices);
}

xla::XlaOp BuildSlice(xla::XlaOp input, absl::Span<const int64_t> base_indices,
                      absl::Span<const int64_t> sizes) {
  XLA_CHECK_EQ(base_indices.size(), sizes.size());
  std::vector<int64_t> limit_indices(base_indices.begin(), base_indices.end());
  std::transform(limit_indices.begin(), limit_indices.end(), sizes.begin(),
                 limit_indices.begin(), std::plus<int64_t>());
  std::vector<int64_t> strides(base_indices.size(), 1);
  return xla::Slice(input, base_indices, limit_indices, strides);
}

xla::XlaOp BoundIndices(xla::XlaOp index, xla::XlaOp max_index) {
  const xla::Shape& index_shape = ShapeHelper::ShapeOfXlaOp(index);
  return xla::Select(
      xla::Ge(index, xla::Zero(index.builder(), index_shape.element_type())),
      index, index + max_index);
}

xla::XlaOp BuildTake(xla::XlaOp input, xla::XlaOp index) {
  static const int take_dim = 0;
  xla::Shape input_shape;
  xla::XlaOp r1_input = XlaHelpers::Flatten(input, &input_shape);
  xla::Shape index_shape;
  xla::XlaOp r1_index = XlaHelpers::Flatten(index, &index_shape);
  xla::XlaOp max_index =
      XlaHelpers::ScalarValue(xla::ShapeUtil::ElementsIn(input_shape),
                              index_shape.element_type(), index.builder());
  xla::XlaOp bound_index = BoundIndices(r1_index, max_index);
  xla::XlaOp r1_result =
      xla::TorchGather(r1_input, bound_index, take_dim,
                       IsSparseGather(input_shape, index_shape, take_dim));
  return XlaHelpers::DynamicReshape(r1_result, index_shape.dimensions());
}

xla::XlaOp BuildResize(xla::XlaOp input, absl::Span<const int64_t> size) {
  xla::Shape input_shape;
  xla::XlaOp r1_input = XlaHelpers::Flatten(input, &input_shape);
  int64_t num_elements = xla::ShapeUtil::ElementsIn(input_shape);
  int64_t new_num_elements = runtime::util::Multiply<int64_t>(size);
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
  return XlaHelpers::DynamicReshape(resized_input, size);
}

xla::XlaOp BuildUnselect(xla::XlaOp target, xla::XlaOp source, int64_t dim,
                         int64_t start, int64_t end, int64_t stride) {
  const xla::Shape& target_shape = ShapeHelper::ShapeOfXlaOp(target);
  const xla::Shape& source_shape = ShapeHelper::ShapeOfXlaOp(source);
  if (target_shape.dimensions(dim) == source_shape.dimensions(dim)) {
    // Shortcut for unselects which are fully covering selects.
    XLA_CHECK_EQ(start, 0);
    XLA_CHECK_EQ(stride, 1);
    XLA_CHECK_EQ(end, target_shape.dimensions(dim));
    return source;
  }

  xla::PrimitiveType pred_type =
      GetXlaPrimitiveTypeForCurrentDevice(xla::PrimitiveType::PRED);
  xla::XlaOp source_true = XlaHelpers::ScalarBroadcast(
      1, pred_type, source_shape.dimensions(), source.builder());
  xla::XlaOp pred_zero = xla::Zero(target.builder(), pred_type);
  xla::XlaOp zero = xla::Zero(target.builder(), target_shape.element_type());
  xla::PaddingConfig padding_config;
  for (int64_t i = 0; i < target_shape.rank(); ++i) {
    auto* dims = padding_config.add_dimensions();
    if (i == dim) {
      dims->set_edge_padding_low(start);
      dims->set_interior_padding(stride - 1);

      int64_t size = start + source_shape.dimensions(i) +
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

xla::XlaOp BuildReflectionPad2d(xla::XlaOp input,
                                absl::Span<const int64_t> padding) {
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  XLA_CHECK_GE(2 * input_shape.rank(), padding.size());
  XLA_CHECK_EQ(padding.size() % 2, 0) << "Uneven padding: " << padding.size();
  xla::XlaOp result = input;
  for (size_t i = 0; i < padding.size(); i += 2) {
    int64_t dim = input_shape.rank() - 1 - i / 2;
    int64_t dim_size = input_shape.dimensions(dim);
    int64_t lhs_padding = padding[i];
    int64_t rhs_padding = padding[i + 1];

    XLA_CHECK(lhs_padding >= 0 && lhs_padding <= dim_size - 1);
    XLA_CHECK(rhs_padding >= 0 && rhs_padding <= dim_size - 1);

    xla::XlaOp reverse = xla::Rev(result, {dim});
    xla::XlaOp lhs_pad = xla::SliceInDim(reverse, dim_size - 1 - lhs_padding,
                                         dim_size - 1, 1, dim);
    xla::XlaOp rhs_pad = xla::SliceInDim(reverse, 1, 1 + rhs_padding, 1, dim);
    result = xla::ConcatInDim(input.builder(), {lhs_pad, result, rhs_pad}, dim);
  }
  return result;
}

xla::XlaOp BuildReflectionPadBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                      absl::Span<const int64_t> padding) {
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  const xla::Shape& grad_output_shape = ShapeHelper::ShapeOfXlaOp(grad_output);
  XLA_CHECK_GE(2 * grad_output_shape.rank(), padding.size());
  XLA_CHECK_EQ(padding.size() % 2, 0) << "Uneven padding: " << padding.size();

  xla::XlaOp grad = grad_output;
  for (size_t i = 0; i < padding.size(); i += 2) {
    int64_t dim = grad_output_shape.rank() - 1 - i / 2;
    int64_t dim_size = grad_output_shape.dimensions(dim);
    int64_t lhs_padding = padding[i];
    int64_t rhs_padding = padding[i + 1];

    XLA_CHECK(lhs_padding >= 0 && lhs_padding <= dim_size - 1);
    XLA_CHECK(rhs_padding >= 0 && rhs_padding <= dim_size - 1);

    xla::XlaOp lhs_pad = xla::SliceInDim(grad, 0, lhs_padding, 1, dim);
    xla::XlaOp reverse_lhs_pad = xla::Rev(lhs_pad, {dim});
    xla::XlaOp padded_lhs_pad =
        PadInDim(reverse_lhs_pad, dim,
                 /*pad_lo=*/1,
                 /*pad_hi=*/input_shape.dimensions(dim) - lhs_padding - 1);

    xla::XlaOp rhs_pad =
        xla::SliceInDim(grad, dim_size - rhs_padding, dim_size, 1, dim);
    xla::XlaOp reverse_rhs_pad = xla::Rev(rhs_pad, {dim});
    xla::XlaOp padded_rhs_pad =
        PadInDim(reverse_rhs_pad, dim,
                 /*pad_lo=*/input_shape.dimensions(dim) - rhs_padding - 1,
                 /*pad_hi=*/1);

    xla::XlaOp grad_core =
        xla::SliceInDim(grad, lhs_padding, dim_size - rhs_padding, 1, dim);
    grad = padded_lhs_pad + grad_core + padded_rhs_pad;
  }
  return grad;
}

xla::XlaOp BuildReplicationPad(xla::XlaOp input,
                               absl::Span<const int64_t> padding) {
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  XLA_CHECK_GE(2 * input_shape.rank(), padding.size());
  XLA_CHECK_EQ(padding.size() % 2, 0) << "Uneven padding: " << padding.size();
  xla::XlaOp result = input;
  for (size_t i = 0; i < padding.size(); i += 2) {
    int64_t dim = input_shape.rank() - 1 - i / 2;
    if ((padding[i] != 0 || padding[i + 1] != 0) &&
        input_shape.dimensions(dim) > 0) {
      std::vector<xla::XlaOp> parts;
      if (padding[i] != 0) {
        xla::XlaOp pad1 = xla::SliceInDim(result, 0, 1, 1, dim);
        parts.push_back(
            XlaHelpers::BroadcastDimensions(pad1, {dim}, {padding[i]}));
      }
      parts.push_back(result);
      if (padding[i + 1] != 0) {
        xla::XlaOp pad1 =
            xla::SliceInDim(result, input_shape.dimensions(dim) - 1,
                            input_shape.dimensions(dim), 1, dim);
        parts.push_back(
            XlaHelpers::BroadcastDimensions(pad1, {dim}, {padding[i + 1]}));
      }
      result = xla::ConcatInDim(result.builder(), parts, dim);
    }
  }
  return result;
}

xla::XlaOp BuildReplicationPadBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                       absl::Span<const int64_t> padding) {
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  const xla::Shape& grad_output_shape = ShapeHelper::ShapeOfXlaOp(grad_output);
  XLA_CHECK_GE(2 * grad_output_shape.rank(), padding.size());
  XLA_CHECK_EQ(padding.size() % 2, 0) << "Uneven padding: " << padding.size();

  xla::XlaOp grad = grad_output;
  for (size_t i = 0; i < padding.size(); i += 2) {
    int64_t dim = grad_output_shape.rank() - 1 - i / 2;
    int64_t dim_size = grad_output_shape.dimensions(dim);
    int64_t lhs_padding = padding[i];
    int64_t rhs_padding = padding[i + 1];

    XLA_CHECK(lhs_padding >= 0 && lhs_padding <= dim_size - 1);
    XLA_CHECK(rhs_padding >= 0 && rhs_padding <= dim_size - 1);

    xla::XlaOp lhs_pad = xla::SliceInDim(grad, 0, lhs_padding, 1, dim);
    xla::XlaOp reduced_lhs_pad =
        BuildSum(lhs_pad, {dim}, /*keep_reduced_dimensions=*/true);
    xla::XlaOp padded_lhs_pad =
        PadInDim(reduced_lhs_pad, dim,
                 /*pad_lo=*/0,
                 /*pad_hi=*/input_shape.dimensions(dim) - 1);

    xla::XlaOp rhs_pad =
        xla::SliceInDim(grad, dim_size - rhs_padding, dim_size, 1, dim);
    xla::XlaOp reduced_rhs_pad =
        BuildSum(rhs_pad, {dim}, /*keep_reduced_dimensions=*/true);
    xla::XlaOp padded_rhs_pad =
        PadInDim(reduced_rhs_pad, dim,
                 /*pad_lo=*/input_shape.dimensions(dim) - 1,
                 /*pad_hi=*/0);

    xla::XlaOp grad_core =
        xla::SliceInDim(grad, lhs_padding, dim_size - rhs_padding, 1, dim);
    grad = padded_lhs_pad + grad_core + padded_rhs_pad;
  }
  return grad;
}

xla::XlaOp PadInDim(xla::XlaOp input, int64_t dim, int64_t pad_lo,
                    int64_t pad_hi, const xla::XlaOp* pad_value) {
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  xla::XlaOp zero;
  if (pad_value == nullptr) {
    zero = xla::Zero(input.builder(), input_shape.element_type());
    pad_value = &zero;
  }
  xla::PaddingConfig padding_config;
  for (int64_t i = 0; i < input_shape.rank(); ++i) {
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

}  // namespace torch_xla
