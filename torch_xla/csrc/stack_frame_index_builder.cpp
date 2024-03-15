#include "torch_xla/csrc/stack_frame_index_builder.h"

namespace torch_xla {

// Invalid stack frame id - used for stack frame population
static int kInvalidIndex = 0;

int FindId(std::string_view key, std::map<std::string_view, int>& index) {
  auto entry_iterator = index.find(key);
  if (entry_iterator == index.end()) {
    return 0;
  } else {
    return entry_iterator->second;
  }
}

void StackFrameIndexBuilder::AddStackFrameLocations(
    const std::vector<torch::lazy::SourceLocation>& frame_info,
    int max_stack_depth, xla::OpMetadata& metadata_to_populate) {
  if (!frame_info.empty()) {
    auto frame_it = frame_info.rbegin();
    int parent_frame_id = kInvalidIndex;
    int depth = 0;
    for (; frame_it != frame_info.rend() && depth < max_stack_depth;
         ++frame_it) {
      parent_frame_id = AddStackFrameLocation(*frame_it, parent_frame_id);
      ++depth;
    }

    // Point to first entry / deepest call / top frame in call stack
    --frame_it;

    metadata_to_populate.set_source_file(frame_it->file);
    metadata_to_populate.set_source_line(frame_it->line);
    metadata_to_populate.set_stack_frame_id(parent_frame_id);
  }
}

int StackFrameIndexBuilder::AddStackFrameLocation(
    const torch::lazy::SourceLocation& frame, int parent_frame_id) {
  int line = frame.line;
  int column = 0;  // Not provided in torch lazy source location - set to zero
  std::string filename = frame.file;
  std::string function_name = frame.function;

  int filename_id = FindId(filename, file_name_to_id_);
  if (filename_id == 0) {
    indexes_.add_file_names(std::move(filename));
    filename_id = indexes_.file_names_size();
    file_name_to_id_[indexes_.file_names(filename_id - 1)] = filename_id;
  }

  int function_name_id = FindId(function_name, function_name_to_id_);
  if (function_name_id == 0) {
    indexes_.add_function_names(std::move(function_name));
    function_name_id = indexes_.function_names_size();
    function_name_to_id_[indexes_.function_names(function_name_id - 1)] =
        function_name_id;
  }

  auto location_tuple =
      std::make_tuple(filename_id, function_name_id, line, column);
  auto file_location_iterator = file_location_to_id_.find(location_tuple);
  int file_location_id = 0;
  if (file_location_iterator == file_location_to_id_.end()) {
    auto file_location = indexes_.add_file_locations();
    file_location->set_file_name_id(filename_id);
    file_location->set_function_name_id(function_name_id);
    file_location->set_line(line);
    file_location->set_column(column);

    file_location_id = indexes_.file_locations_size();
    file_location_to_id_[location_tuple] = file_location_id;
  } else {
    file_location_id = file_location_iterator->second;
  }

  auto frame_tuple = std::make_tuple(file_location_id, parent_frame_id);
  auto stack_frame_iterator = frame_to_id_.find(frame_tuple);
  int stack_frame_id = 0;
  if (stack_frame_iterator == frame_to_id_.end()) {
    auto frame = indexes_.add_stack_frames();
    frame->set_file_location_id(file_location_id);
    frame->set_parent_frame_id(parent_frame_id);

    stack_frame_id = indexes_.stack_frames_size();
    frame_to_id_[frame_tuple] = stack_frame_id;
  } else {
    stack_frame_id = stack_frame_iterator->second;
  }

  return stack_frame_id;
}

}  // namespace torch_xla