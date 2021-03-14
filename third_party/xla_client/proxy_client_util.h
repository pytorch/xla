#pragma once

#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/tpu/topology.pb.h"

#include <cassert>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <strstream>
#include <vector>

namespace xla {

class HloModuleProto;

#if 1  // TF appears to disable assert.
// TODO: use the TF_CHECK macros rather than generic assert.
#undef assert
#undef __ASSERT_FUNCTION
static void my_assert_fail(const char *a, const char *b, unsigned int cc,
                           const char *d) {
  std::cerr << "ASSERTION FAILED: " << a << " " << b << ":" << cc << " " << d
            << std::endl
            << std::flush;
  raise(SIGTRAP);
}

#define assert(expr)                                    \
  (static_cast<bool>(expr)                              \
       ? void(0)                                        \
       : xla::my_assert_fail(#expr, __FILE__, __LINE__, \
                             __extension__ __PRETTY_FUNCTION__))
#endif

void print_environment_config();

std::vector<std::string> split(const std::string &str, const char delim);

std::string join(const std::vector<std::string> &pieces,
                 const std::string &delimiter);

std::string get_proxy_device(const xla::HloModuleProto &module);

std::unique_ptr<xla::HloModuleProto> get_proxy_hlo_module(
    const xla::HloModuleProto &module);

void set_frontend_attribute(xla::HloModuleProto &module,
                            const std::string &attribute_name,
                            std::string attribute_value);

template <typename MSG>
inline std::string msg_to_json(const MSG &msg) {
  std::string json;
  google::protobuf::util::JsonPrintOptions op;
  op.add_whitespace = true;
  google::protobuf::util::MessageToJsonString(msg, &json, op);
  return std::move(json);
}

template <typename MSG>
inline bool save_msg(const MSG &msg, const std::string &file) {
  const std::string json = msg_to_json(msg);

  FILE *f = fopen(file.c_str(), "wt");
  if (f) {
    fwrite(json.c_str(), json.size(), sizeof(std::string::value_type), f);
    fclose(f);
    return true;
  } else {
    LOG(ERROR) << "Could not open file: " << file
               << ", reason: " << strerror(errno) << std::endl;
    return false;
  }
}

template <typename PType>
void *get_data_pointer(xla::Literal &literal) {
  return literal.data<PType>().data();
}

template <typename PType>
const void *get_data_pointer(const xla::Literal &literal) {
  return literal.data<PType>().data();
}

/**
 * @brief Probably an incorrect copy function. See tensor_util.cpp
 *        Note: It's possible that this copy isn't correct in
 *              some cases.
 */
template <typename L>
inline void *get_data_ptr(L &literal) {
  switch (literal.shape().element_type()) {
    case xla::PrimitiveType::PRED:
      return get_data_pointer<bool>(literal);
    case xla::PrimitiveType::F16:
      return get_data_pointer<xla::half>(literal);
    case xla::PrimitiveType::BF16:
      return get_data_pointer<tensorflow::bfloat16>(literal);
    case xla::PrimitiveType::F32:
      return get_data_pointer<float>(literal);
    case xla::PrimitiveType::F64:
      return get_data_pointer<double>(literal);
    case xla::PrimitiveType::U8:
      return get_data_pointer<xla::uint8>(literal);
    case xla::PrimitiveType::S8:
      return get_data_pointer<xla::int8>(literal);
    case xla::PrimitiveType::S16:
      return get_data_pointer<xla::int16>(literal);
    case xla::PrimitiveType::U16:
      return get_data_pointer<xla::uint16>(literal);
    case xla::PrimitiveType::S32:
      return get_data_pointer<xla::int32>(literal);
    case xla::PrimitiveType::U32:
      return get_data_pointer<xla::uint32>(literal);
    case xla::PrimitiveType::S64:
      return get_data_pointer<xla::int64>(literal);
    case xla::PrimitiveType::U64:
      return get_data_pointer<xla::uint64>(literal);
    case xla::PrimitiveType::C64:
      return get_data_pointer<xla::complex64>(literal);
    case xla::PrimitiveType::C128:
      return get_data_pointer<xla::complex128>(literal);
    default:
      XLA_ERROR() << "Unsupported literal type: " << literal.shape();
  }
}

/// \brief Convert a TensorSource object to an xla::Literal object
inline xla::Literal tensor_to_literal(
    const ComputationClient::TensorSource &tensor_source) {
  xla::Literal literal(tensor_source.shape);
  tensor_source.populate_fn(tensor_source, get_data_ptr(literal),
                            literal.size_bytes());
  return literal;
}

/// \brief Convert an xla::Literal to a TensorSource object
inline ComputationClient::TensorSource literal_to_tensor(xla::Literal &&literal,
                                                         std::string device) {
  // make a copy of the shape before the move
  xla::Shape shape = literal.shape();
  ComputationClient::TensorSource td(
      /*shape=*/std::move(shape),
      /*device=*/std::move(device),
      /*populate_fn=*/
      [l = std::make_shared<xla::Literal>(std::move(literal))](
          const ComputationClient::TensorSource &src, void *buff, size_t size) {
        memcpy(buff, get_data_ptr(*l), size);
      });
  return td;
}

/// \brief Build a topology without the TPIU configuration OP (deprecated)
tensorflow::tpu::TopologyProto InitializeAndFetchTopologyLocal(
    const std::string &job, int task_no, const std::string &worker_host_port,
    const tensorflow::ConfigProto &config);

}  // namespace xla
