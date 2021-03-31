#pragma once

#include <string>

#include "lazy_tensors/computation_client/tf_logging.h"

namespace xla {

class HloModuleProto {
 public:
  std::string SerializeAsString() const {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
};

}  // namespace xla
