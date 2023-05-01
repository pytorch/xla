#ifndef XLA_TORCH_XLA_CSRC_FUNCTION_CALL_TRACKER_H_
#define XLA_TORCH_XLA_CSRC_FUNCTION_CALL_TRACKER_H_

namespace torch_xla {
namespace fn_tracker {

#define XLA_FN_TRACK(level) \
  torch_xla::fn_tracker::TrackFunction(__FUNCTION__, level)

void TrackFunction(const char* tag, int level);

}  // namespace fn_tracker
}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_FUNCTION_CALL_TRACKER_H_