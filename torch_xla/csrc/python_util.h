#pragma once

#include <string>
#include <vector>

#include <c10/util/Optional.h>

namespace torch_xla {

// Represents a location within a source code file.
struct SourceLocation {
  std::string file;
  std::string function;
  int line = -1;
};

// Extracts the top Python frame which ended up calling into the C++ side. If no
// such frame is available, nullopt is returned.
c10::optional<SourceLocation> GetPythonFrameTop();

// Returns the stack track of the Python side. Frame 0 is the closer to the C++
// side.
std::vector<SourceLocation> GetPythonFrames();

}  // namespace torch_xla
