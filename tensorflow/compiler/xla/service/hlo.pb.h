#pragma once

#include <string>

#include "tensorflow/compiler/xla/xla_client/tf_logging.h"

namespace xla {

class HloModuleProto {
 public:
  std::string SerializeAsString() const {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
};

}  // namespace xla
