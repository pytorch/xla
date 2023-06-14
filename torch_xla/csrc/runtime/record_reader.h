#ifndef XLA_CLIENT_RECORD_READER_H_
#define XLA_CLIENT_RECORD_READER_H_

#include <memory>
#include <mutex>
#include <string>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/tsl/lib/io/record_reader.h"

namespace torch_xla {
namespace runtime {
namespace util {

class RecordReader {
 public:
  using Data = tsl::tstring;

  RecordReader(std::string path, const std::string& compression,
               int64_t buffer_size);

  const std::string& path() const { return path_; }

  bool Read(Data* value);

 private:
  std::string path_;
  std::mutex lock_;
  uint64_t offset_ = 0;
  std::unique_ptr<tsl::RandomAccessFile> file_;
  std::unique_ptr<tsl::io::RecordReader> reader_;
};

}  // namespace util
}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_CLIENT_RECORD_READER_H_
