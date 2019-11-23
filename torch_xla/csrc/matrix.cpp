#include "torch_xla/csrc/matrix.h"

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/helpers.h"

namespace torch_xla {
namespace {

struct DiagonalMask {
  xla::XlaOp source;
  xla::XlaOp mask;
};

xla::PaddingConfig CreateDiagonalPaddingConfig(const xla::Shape& target_shape,
                                               const xla::Shape& input_shape,
                                               xla::int64 offset) {
  xla::int64 rank = target_shape.rank();
  xla::PaddingConfig padding_config;
  for (xla::int64 i = 0; i < rank - 2; ++i) {
    auto* dims = padding_config.add_dimensions();
    dims->set_edge_padding_low(0);
    dims->set_interior_padding(0);
    dims->set_edge_padding_high(0);
  }
  auto* dims = padding_config.add_dimensions();
  dims->set_interior_padding(target_shape.dimensions(rank - 1));
  if (offset >= 0) {
    dims->set_edge_padding_low(offset);
  } else {
    dims->set_edge_padding_low(target_shape.dimensions(rank - 1) * (-offset));
  }
  xla::int64 num_elements =
      target_shape.dimensions(rank - 2) * target_shape.dimensions(rank - 1);
  xla::int64 num_interior_paddings =
      input_shape.dimensions(input_shape.rank() - 1) - 1;
  dims->set_edge_padding_high(num_elements - dims->edge_padding_low() -
                              num_interior_paddings * dims->interior_padding() -
                              (num_interior_paddings + 1));
  return padding_config;
}

DiagonalMask CreateDiagonalMask(const xla::XlaOp& input,
                                const xla::Shape& target_shape,
                                xla::int64 offset) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::PaddingConfig padding_config =
      CreateDiagonalPaddingConfig(target_shape, input_shape, offset);
  xla::XlaOp zero_scalar =
      XlaHelpers::ScalarValue(0, input_shape.element_type(), input.builder());
  xla::XlaOp source = xla::Reshape(xla::Pad(input, zero_scalar, padding_config),
                                   target_shape.dimensions());

  xla::XlaOp false_scalar =
      XlaHelpers::ScalarValue(false, xla::PrimitiveType::PRED, input.builder());
  xla::XlaOp empty_mask =
      XlaHelpers::ScalarBroadcast(true, xla::PrimitiveType::PRED,
                                  input_shape.dimensions(), input.builder());
  xla::XlaOp mask =
      xla::Reshape(xla::Pad(empty_mask, false_scalar, padding_config),
                   target_shape.dimensions());
  return {source, mask};
}

std::vector<xla::int64> GetDiagonalPermutation(xla::int64 rank, xla::int64 dim1,
                                               xla::int64 dim2) {
  std::vector<xla::int64> permutation;
  for (xla::int64 dim = 0; dim < rank; ++dim) {
    if (dim != dim1 && dim != dim2) {
      permutation.push_back(dim);
    }
  }
  permutation.push_back(dim1);
  permutation.push_back(dim2);
  return permutation;
}

}  // namespace

xla::XlaOp BuildTriu(const xla::XlaOp& input, xla::int64 diagonal) {
  return xla::Select(xla::TriangleMask(input, diagonal - 1),
                     xla::ZerosLike(input), input);
}

xla::XlaOp BuildTril(const xla::XlaOp& input, xla::int64 diagonal) {
  return xla::Select(xla::TriangleMask(input, diagonal), input,
                     xla::ZerosLike(input));
}

xla::XlaOp BuildDiagonal(const xla::XlaOp& input, xla::int64 offset,
                         xla::int64 dim1, xla::int64 dim2) {
  xla::XlaOp diag_input = input;
  if (dim1 != 0 || dim2 != 1) {
    const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
    auto permutation = GetDiagonalPermutation(input_shape.rank(), dim1, dim2);
    diag_input = xla::Transpose(diag_input, permutation);
  }
  return xla::GetMatrixDiagonal(diag_input, offset);
}

xla::XlaOp BuildDiagonalViewUpdate(const xla::XlaOp& target,
                                   const xla::XlaOp& input, xla::int64 offset,
                                   xla::int64 dim1, xla::int64 dim2) {
  const xla::Shape* target_shape = &XlaHelpers::ShapeOfXlaOp(target);
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp diag_input = input;
  if (target_shape->element_type() != input_shape.element_type()) {
    diag_input = ConvertTo(input, input_shape.element_type(),
                           target_shape->element_type(), /*device=*/nullptr);
  }
  std::vector<xla::int64> permutation;
  xla::XlaOp diag_target = target;
  if (dim1 != 0 || dim2 != 1) {
    permutation = GetDiagonalPermutation(target_shape->rank(), dim1, dim2);
    diag_target = xla::Transpose(diag_target, permutation);
    target_shape = &XlaHelpers::ShapeOfXlaOp(diag_target);
  }
  DiagonalMask dmask = CreateDiagonalMask(diag_input, *target_shape, offset);
  xla::XlaOp result = xla::Select(dmask.mask, dmask.source, diag_target);
  if (!permutation.empty()) {
    result = xla::Transpose(result, xla::InversePermutation(permutation));
  }
  return result;
}

}  // namespace torch_xla
