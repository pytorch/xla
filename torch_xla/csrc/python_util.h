#pragma once

#include <iostream>
#include <string>
#include <vector>

#include <c10/util/Optional.h>

namespace torch_xla {

struct SourceLocation {
  std::string file;
  std::string function;
  int line = -1;
};

c10::optional<SourceLocation> GetPythonFrameTop();

std::vector<SourceLocation> GetPythonFrames();

std::ostream& operator<<(std::ostream& stream,
                         const std::vector<SourceLocation>& frames);

}  // namespace torch_xla
