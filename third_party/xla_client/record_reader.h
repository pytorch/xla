#ifndef TENSORFLOW_COMPILER_XLA_RPC_RECORD_READER_H_
#define TENSORFLOW_COMPILER_XLA_RPC_RECORD_READER_H_

#include <memory>
#include <mutex>
#include <string>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/io/record_reader.h"

namespace xla {
namespace util {

class RecordReader {
 public:
  RecordReader(std::string path, const std::string& compression,
               int64 buffer_size);

  const std::string& path() const { return path_; }

  bool Read(std::string* value);

 private:
  std::string path_;
  std::mutex lock_;
  uint64 offset_ = 0;
  std::unique_ptr<tensorflow::RandomAccessFile> file_;
  std::unique_ptr<tensorflow::io::RecordReader> reader_;
};

}  // namespace util
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RPC_RECORD_READER_H_
