#pragma once

#include <cstdint>
#include <ostream>
#include <string>

#include "lazy_tensors/compiler/xla/xla_client/tf_logging.h"

namespace xla {

enum class PrimitiveType {
  PRED,
  S8,
  S16,
  S32,
  S64,
  U8,
  U16,
  U32,
  U64,
  F16,
  F32,
  BF16,
  F64,
  C64,
  C128,
  TUPLE,
  INVALID
};

inline std::string PrimitiveTypeName(PrimitiveType primitive_type) {
  switch (primitive_type) {
    case PrimitiveType::PRED: {
      return "pred";
    }
    case PrimitiveType::S8: {
      return "s8";
    }
    case PrimitiveType::S16: {
      return "s16";
    }
    case PrimitiveType::S32: {
      return "s32";
    }
    case PrimitiveType::S64: {
      return "s64";
    }
    case PrimitiveType::U8: {
      return "u8";
    }
    case PrimitiveType::U16: {
      return "u16";
    }
    case PrimitiveType::U32: {
      return "u32";
    }
    case PrimitiveType::U64: {
      return "u64";
    }
    case PrimitiveType::F16: {
      return "f16";
    }
    case PrimitiveType::F32: {
      return "f32";
    }
    case PrimitiveType::BF16: {
      return "bf16";
    }
    case PrimitiveType::F64: {
      return "f64";
    }
    case PrimitiveType::C64: {
      return "c64";
    }
    case PrimitiveType::C128: {
      return "c128";
    }
    case PrimitiveType::TUPLE: {
      return "tuple";
    }
    default: { return "invalid"; }
  }
}

class PaddingConfig {
 public:
  class PaddingConfigDimension {
   public:
    int64_t edge_padding_low() const { return edge_padding_low_; }
    void set_edge_padding_low(int64_t value) { edge_padding_low_ = value; }
    int64_t edge_padding_high() const { return edge_padding_high_; }
    void set_edge_padding_high(int64_t value) { edge_padding_high_ = value; }
    int64_t interior_padding() const { return interior_padding_; }
    void set_interior_padding(int64_t value) { interior_padding_ = value; }

   private:
    int64_t edge_padding_low_ = 0;
    int64_t edge_padding_high_ = 0;
    int64_t interior_padding_ = 0;
  };

  PaddingConfigDimension* add_dimensions() {
    dimensions_.emplace_back(new PaddingConfigDimension());
    return dimensions_.back().get();
  }

  const std::vector<std::unique_ptr<PaddingConfigDimension>>& dimensions()
      const {
    return dimensions_;
  }

 private:
  std::vector<std::unique_ptr<PaddingConfigDimension>> dimensions_;
};

class PrecisionConfig {
 public:
  enum class Precision { DEFAULT, HIGH, HIGHEST };

  static const Precision DEFAULT = Precision::DEFAULT;
  static const Precision HIGHEST = Precision::HIGHEST;

  struct RepeatedFieldPrecision {
    void Resize(int num, Precision val) {
      TF_LOG(FATAL) << "Not implemented yet.";
    }
  };

  RepeatedFieldPrecision* mutable_operand_precision() {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
};

class OpMetadata {
 public:
  void set_op_type(const std::string& value) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
  void set_op_name(const std::string& value) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
  void set_source_file(const std::string& value) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
  void set_source_line(int value) { TF_LOG(FATAL) << "Not implemented yet."; }
};

inline std::ostream& operator<<(std::ostream& os,
                                PrimitiveType primitive_type) {
  os << PrimitiveTypeName(primitive_type);
  return os;
}

class ScatterDimensionNumbers {
 public:
  void set_index_vector_dim(int64_t value) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
  void add_update_window_dims(int64_t value) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
  void add_inserted_window_dims(int64_t value) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
  void add_scatter_dims_to_operand_dims(int64_t value) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
  void add_lhs_batch_dimensions(int64_t value) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
  void add_rhs_batch_dimensions(int64_t value) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
  void add_lhs_contracting_dimensions(int64_t value) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
  void add_rhs_contracting_dimensions(int64_t value) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
  void add_collapsed_slice_dims(int64_t value) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
  void add_offset_dims(int64_t value) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
  void add_start_index_map(int64_t value) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
};

class DotDimensionNumbers {
 public:
  void add_lhs_batch_dimensions(int64_t value) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
  void add_rhs_batch_dimensions(int64_t value) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
  void add_lhs_contracting_dimensions(int64_t value) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
  void add_rhs_contracting_dimensions(int64_t value) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
};

class GatherDimensionNumbers {
 public:
  void add_collapsed_slice_dims(int64_t value) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
  void add_offset_dims(int64_t value) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
  void set_index_vector_dim(int64_t value) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
  void add_start_index_map(int64_t value) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
};

class ConvolutionDimensionNumbers {};

class TriangularSolveOptions {
 public:
  enum class Transpose { NO_TRANSPOSE, TRANSPOSE };

  static const Transpose NO_TRANSPOSE = Transpose::NO_TRANSPOSE;
  static const Transpose TRANSPOSE = Transpose::TRANSPOSE;
};

enum class RandomAlgorithm {
  RNG_DEFAULT = 0,
};

class ReplicaGroup {
 public:
  void add_replica_ids(int64_t value) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
};

class ChannelHandle {};

}  // namespace xla
