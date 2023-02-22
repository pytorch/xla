#include "third_party/xla_client/xla_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <set>
#include <unordered_map>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status_matchers.h"
#include "tensorflow/tsl/protobuf/error_codes.pb.h"
#include "xla_util.h"

namespace xla {
namespace util {

using ::testing::AllOf;
using ::testing::HasSubstr;
using ::tsl::testing::IsOk;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

TEST(XlaUtilTest, ShapeHash) {
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {2, 2});
  EXPECT_EQ(ShapeHash(shape), ShapeHash(shape));
}

template <typename MessageType>
StatusOr<MessageType> ParseTextProto(const std::string& text_proto) {
  tensorflow::protobuf::TextFormat::Parser parser;
  MessageType parsed_proto;
  tensorflow::protobuf::io::ArrayInputStream input_stream(text_proto.data(),
                                                          text_proto.size());
  if (!parser.Parse(&input_stream, &parsed_proto)) {
    return tensorflow::errors::InvalidArgument("Could not parse text proto: ",
                                               text_proto);
  }
  return parsed_proto;
}

TEST(XlaUtilrest, CreateModule) {
  TF_ASSERT_OK_AND_ASSIGN(
      HloModuleProto hlo_module_proto,
      ParseTextProto<HloModuleProto>(
          R"pb(
            name: "myname"
            id: 7
            entry_computation_name: "mycomp"
            entry_computation_id: 0
            computations {
              id: 0
              name: "c1"
              instructions: {
                name: "i1"
                id: 1
                opcode: "constant"
                shape: {
                  element_type: S32
                  layout {}
                }
                literal: {
                  shape: {
                    element_type: S32
                    layout {}
                  }
                  s32s: 0
                }
              }
              instructions: {
                name: "constant.3"
                id: 0
                opcode: "constant"
                shape: {
                  element_type: S32
                  layout {}
                }
                literal: {
                  shape: {
                    element_type: S32
                    layout {}
                  }
                  s32s: 0
                }
              }
              root_id: 1
            }
            host_program_shape: { result: { element_type: 4 } }
          )pb"));

  HloModule m("cool_module", {});
  auto got = CreateModuleFromProto(hlo_module_proto);
  EXPECT_THAT(got, IsOk());
  EXPECT_EQ((*got)->name(), "myname");
  EXPECT_EQ((*got)->computation_count(), 1);
}

TEST(XlaUtilrest, XlaToHlo) {
  xla::Shape input_shape =
      xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {2, 2});
  xla::XlaBuilder builder("AddComputation");
  xla::XlaOp x = xla::Parameter(&builder, 0, input_shape, "x");
  xla::XlaOp y = xla::Parameter(&builder, 1, input_shape, "y");
  xla::XlaOp sum = xla::Add(x, y);
  ASSERT_THAT(GetComputationHloText(*builder.Build()),
              IsOkAndHolds(AllOf(
                  HasSubstr("HloModule AddComputation.4"),
                  HasSubstr("%AddComputation.4 (x.1: f32[2,2], y.2: f32[2,2])"),
                  HasSubstr("ROOT %add.3"))));
}

}  // namespace util
}  // namespace xla
