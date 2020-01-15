#include "tensorflow/compiler/xla/xla_client/record_reader.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"

namespace xla {
namespace util {

RecordReader::RecordReader(std::string path, const string& compression,
                           int64 buffer_size)
    : path_(std::move(path)) {
  tensorflow::Env* env = tensorflow::Env::Default();
  XLA_CHECK_OK(env->NewRandomAccessFile(path_, &file_));
  tensorflow::io::RecordReaderOptions options =
      tensorflow::io::RecordReaderOptions::CreateRecordReaderOptions(
          compression);
  options.buffer_size = buffer_size;
  reader_.reset(new tensorflow::io::RecordReader(file_.get(), options));
}

bool RecordReader::Read(std::string* value) {
  // We need to pass a tensorflow::tstring here, which will ultimately result in
  // making a copy. Hopefully the tensorflow string story will end with a nice
  // outcome.
  tensorflow::tstring tvalue;
  std::lock_guard<std::mutex> slock(lock_);
  xla::Status status = reader_->ReadRecord(&offset_, &tvalue);
  if (tensorflow::errors::IsOutOfRange(status)) {
    return false;
  }
  XLA_CHECK_OK(status) << path_ << " offset " << offset_;
  *value = tvalue;
  return true;
}

}  // namespace util
}  // namespace xla
