#include "tensorflow/compiler/xla/client/lib/matrix.h"

#include "lazy_xla/csrc/compiler/helpers.h"

using namespace torch::jit::tensorexpr;

namespace xla {

XlaOp GetMatrixDiagonal(XlaOp x, int k) {
  const auto& shape = torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(x);
  int64 rank = shape.rank();
  XLA_CHECK_GE(rank, 2);
  std::vector<int64> output_sizes;
  output_sizes.reserve(shape.rank() - 1);
  for (size_t i = 0; i < static_cast<size_t>(rank - 2); ++i) {
    output_sizes.push_back(shape.dimensions(i));
  }
  int M = shape.dimensions(rank - 2);
  int N = shape.dimensions(rank - 1);
  output_sizes.push_back(k >= 0 ? std::min(M, N - k) : std::min(M + k, N));
  return XlaOp(
      Compute("get_matrix_diagonal", x, output_sizes,
              [&](const std::vector<ExprHandle>& indices) {
                XLA_CHECK_GE(indices.size(), size_t(1));
                auto diag_it = indices.end() - 1;
                std::vector<ExprHandle> expr_indices(indices.begin(), diag_it);
                if (k >= 0) {
                  expr_indices.emplace_back(*diag_it);
                  expr_indices.emplace_back(*diag_it + ExprHandle(k));
                } else {
                  expr_indices.emplace_back(*diag_it - ExprHandle(k));
                  expr_indices.emplace_back(*diag_it);
                }
                return expr_indices;
              }),
      std::make_unique<Shape>(shape.element_type(), output_sizes), x.builder());
}

}  // namespace xla
