#include "torch_xla/csrc/matrix.h"

#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/shape_helper.h"
#include "xla/client/lib/constants.h"
#include "xla/client/lib/matrix.h"
#include "xla/hlo/builder/lib/qr.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace torch_xla {
namespace {

struct DiagonalMask {
  xla::XlaOp source;
  xla::XlaOp mask;
};

xla::PaddingConfig CreateDiagonalPaddingConfig(const xla::Shape& target_shape,
                                               const xla::Shape& input_shape,
                                               int64_t offset) {
  int64_t rank = target_shape.rank();
  xla::PaddingConfig padding_config;
  for (int64_t i = 0; i < rank - 2; ++i) {
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
  int64_t num_elements =
      target_shape.dimensions(rank - 2) * target_shape.dimensions(rank - 1);
  int64_t num_interior_paddings =
      input_shape.dimensions(input_shape.rank() - 1) - 1;
  dims->set_edge_padding_high(num_elements - dims->edge_padding_low() -
                              num_interior_paddings * dims->interior_padding() -
                              (num_interior_paddings + 1));
  return padding_config;
}

DiagonalMask CreateDiagonalMask(xla::XlaOp input,
                                const xla::Shape& target_shape,
                                int64_t offset) {
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  xla::PaddingConfig padding_config =
      CreateDiagonalPaddingConfig(target_shape, input_shape, offset);
  xla::XlaOp zero_scalar =
      xla::Zero(input.builder(), input_shape.element_type());
  xla::XlaOp source = xla::Reshape(xla::Pad(input, zero_scalar, padding_config),
                                   target_shape.dimensions());

  xla::XlaOp false_scalar =
      xla::Zero(input.builder(), xla::PrimitiveType::PRED);
  xla::XlaOp empty_mask =
      XlaHelpers::ScalarBroadcast(true, xla::PrimitiveType::PRED,
                                  input_shape.dimensions(), input.builder());
  xla::XlaOp mask =
      xla::Reshape(xla::Pad(empty_mask, false_scalar, padding_config),
                   target_shape.dimensions());
  return {source, mask};
}

std::vector<int64_t> GetDiagonalPermutation(int64_t rank, int64_t dim1,
                                            int64_t dim2) {
  std::vector<int64_t> permutation;
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (dim != dim1 && dim != dim2) {
      permutation.push_back(dim);
    }
  }
  permutation.push_back(dim1);
  permutation.push_back(dim2);
  return permutation;
}

}  // namespace

xla::XlaOp BuildTriu(xla::XlaOp input, int64_t diagonal) {
  return xla::Select(xla::TriangleMask(input, diagonal - 1),
                     xla::ZerosLike(input), input);
}

xla::XlaOp BuildTril(xla::XlaOp input, int64_t diagonal) {
  return xla::Select(xla::TriangleMask(input, diagonal), input,
                     xla::ZerosLike(input));
}

xla::XlaOp BuildDiagonal(xla::XlaOp input, int64_t offset, int64_t dim1,
                         int64_t dim2) {
  xla::XlaOp diag_input = input;
  if (dim1 != 0 || dim2 != 1) {
    const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
    auto permutation = GetDiagonalPermutation(input_shape.rank(), dim1, dim2);
    diag_input = xla::Transpose(diag_input, permutation);
  }
  return xla::GetMatrixDiagonal(diag_input, offset);
}

xla::XlaOp BuildDiagonalViewUpdate(xla::XlaOp target, xla::XlaOp input,
                                   int64_t offset, int64_t dim1, int64_t dim2) {
  const xla::Shape* target_shape = &ShapeHelper::ShapeOfXlaOp(target);
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  xla::XlaOp diag_input = input;
  if (target_shape->element_type() != input_shape.element_type()) {
    diag_input = ConvertTo(input, input_shape.element_type(),
                           target_shape->element_type());
  }
  std::vector<int64_t> permutation;
  xla::XlaOp diag_target = target;
  if (dim1 != 0 || dim2 != 1) {
    permutation = GetDiagonalPermutation(target_shape->rank(), dim1, dim2);
    diag_target = xla::Transpose(diag_target, permutation);
    target_shape = &ShapeHelper::ShapeOfXlaOp(diag_target);
  }
  DiagonalMask dmask = CreateDiagonalMask(diag_input, *target_shape, offset);
  xla::XlaOp result = xla::Select(dmask.mask, dmask.source, diag_target);
  if (!permutation.empty()) {
    result = xla::Transpose(result, xla::InversePermutation(permutation));
  }
  return result;
}

xla::XlaOp BuildInverse(xla::XlaOp input) {
  xla::XlaOp q, r;
  xla::QrExplicit(input, /*full_matrices=*/false, q, r);

  return xla::TriangularSolve(r, xla::TransposeInMinorDims(q),
                              /*left_side=*/true,
                              /*lower=*/false, /*unit_diagonal=*/false,
                              /*transpose_a=*/
                              xla::TriangularSolveOptions::NO_TRANSPOSE);
}

}  // namespace torch_xla
