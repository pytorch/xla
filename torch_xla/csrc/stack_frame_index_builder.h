#pragma once

#include <map>
#include <string_view>
#include <tuple>

#include <torch/csrc/lazy/core/ir_metadata.h>  // SourceLocation

#include "xla/service/hlo.pb.h"
#include "xla/types.h"

namespace torch_xla {

// TODO: Deduplicate with
// https://github.com/openxla/xla/blob/952d3cf39c3e3eeaa790cc1dd53423c8eb27d473/xla/translate/mhlo_to_hlo/stack_frame_index_builder.cc#L40
// in openxla/xla
class StackFrameIndexBuilder {
 public:
  StackFrameIndexBuilder() {}

  void AddStackFrameLocations(const std::vector<torch::lazy::SourceLocation>& f,
                              int max_stack_depth,
                              xla::OpMetadata& metadata_to_populate);

  const xla::StackFrameIndexProto& stack_frame_index() const {
    return indexes_;
  }

 private:
  int AddStackFrameLocation(const torch::lazy::SourceLocation& source,
                            int parent_id);

  // Stack frame index tables - we accumulate and write these to the HloModule
  xla::StackFrameIndexProto indexes_;

  std::map<std::string_view, int> function_name_to_id_;
  std::map<std::string_view, int> file_name_to_id_;
  std::map<std::tuple<int, int, int, int>, int> file_location_to_id_;
  std::map<std::tuple<int, int>, int> frame_to_id_;
};  // StackFrameIndexBuilder

}  // namespace torch_xla