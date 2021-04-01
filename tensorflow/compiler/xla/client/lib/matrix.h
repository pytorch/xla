#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace xla {

inline XlaOp IdentityMatrix(XlaBuilder* builder, PrimitiveType type, int64 m,
                            int64 n) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

// Get the diagonals of the last two dimensions. Use k>0 for diagonals above the
// main diagonal, and k<0 for diagonals below the main diagonal.
//
// If 'x' has shape [..., M, N]
//  If k >= 0: then the output has shape [..., min(M, N - k)], containing the
//            diagonal elements (i.e., with indices [..., i, i + k]).
//  If k < 0: then the output has shape [..., min(M + k, N)], containing the
//            diagonal elements (i.e., with indices [..., i - k, i]).
XlaOp GetMatrixDiagonal(XlaOp x, int k = 0);

inline XlaOp TriangleMask(XlaOp x, int diagonal) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp Triangle(XlaOp x, bool lower) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp TransposeInMinorDims(XlaOp x) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

}  // namespace xla
