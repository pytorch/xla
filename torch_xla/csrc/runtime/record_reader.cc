#include "torch_xla/csrc/runtime/record_reader.h"

#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/strcat.h"
#include "torch_xla/csrc/runtime/debug_macros.h"

namespace torch_xla {
namespace runtime {
namespace util {

RecordReader::RecordReader(std::string path, const std::string& compression,
                           int64_t buffer_size)
    : path_(std::move(path)) {
  tsl::Env* env = tsl::Env::Default();
  XLA_CHECK_OK(env->NewRandomAccessFile(path_, &file_));
  tsl::io::RecordReaderOptions options =
      tsl::io::RecordReaderOptions::CreateRecordReaderOptions(compression);
  options.buffer_size = buffer_size;
  reader_.reset(new tsl::io::RecordReader(file_.get(), options));
}

bool RecordReader::Read(Data* value) {
  std::lock_guard<std::mutex> slock(lock_);
  xla::Status status = reader_->ReadRecord(&offset_, value);
  if (tsl::errors::IsOutOfRange(status)) {
    return false;
  }
  XLA_CHECK_OK(status) << path_ << " offset " << offset_;
  return true;
}

}  // namespace util
}  // namespace runtime
}  // namespace torch_xla
