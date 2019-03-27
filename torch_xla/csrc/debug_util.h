#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/xla_client/tf_logging.h"
#include "torch_xla/csrc/tensor.h"

namespace torch_xla {
namespace internal {

struct StreamSink : public std::basic_ostringstream<char> {};

class Streamer {
 public:
  Streamer(std::ostream& output_stream = std::cerr)
      : output_stream_(output_stream) {}

  void operator&(const std::basic_ostream<char>& oss) const {
    const StreamSink& sink = dynamic_cast<const StreamSink&>(oss);
    output_stream_ << sink.str();
  }

 private:
  std::ostream& output_stream_;
};

}  // namespace internal

#define XLA_DEBUG(level)                 \
  TF_PREDICT_TRUE(!TF_VLOG_IS_ON(level)) \
  ? (void)0                              \
  : ::torch_xla::internal::Streamer() & ::torch_xla::internal::StreamSink()

class DebugUtil {
 public:
  enum GraphFormat {
    kText,
    kDot,
  };

  // Dumps the current Python frame and the IR Graph whose roots are the IR
  // values held at the tensors. If indices is not nullptr, it selects the
  // indices of the tensors whose graph will be emitted.
  static std::string GetTensorsGraphInfo(
      const std::vector<XLATensor>& tensors, const std::vector<size_t>* indices,
      GraphFormat format = GraphFormat::kText);
};

}  // namespace torch_xla
