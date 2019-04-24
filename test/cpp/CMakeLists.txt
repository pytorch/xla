cmake_minimum_required(VERSION 3.0)

set(GTEST_DIR "${CMAKE_BINARY_DIR}/gtest")

get_filename_component(PTXLA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
get_filename_component(PT_DIR "${PTXLA_DIR}/.." ABSOLUTE)
set(TFDIR "${PTXLA_DIR}/third_party/tensorflow")

file(GLOB PTXLA_LIBDIRS "${PTXLA_DIR}/build/lib.*")
list(GET PTXLA_LIBDIRS 0 PTXLA_LIBDIR)
message("Selected PT/XLA library folder ${PTXLA_LIBDIR}")

project(ptxla_test)

find_package(PythonLibs)

include(ExternalProject)
set_directory_properties(PROPERTIES EP_PREFIX "${GTEST_DIR}")

ExternalProject_Add(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG master
  SOURCE_DIR "${GTEST_DIR}/src/googletest-src"
  BINARY_DIR "${GTEST_DIR}/src/googletest-build"
  # Disable install step
  INSTALL_COMMAND ""
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON)

ExternalProject_Get_Property(googletest SOURCE_DIR)

include_directories(
  "${SOURCE_DIR}/googletest/include"
  "${PTXLA_DIR}"
  "${PTXLA_DIR}/torch_xla/csrc"
  "${TFDIR}/bazel-tensorflow"
  "${TFDIR}/bazel-genfiles"
  "${TFDIR}/bazel-tensorflow/external/protobuf_archive/src"
  "${TFDIR}/bazel-tensorflow/external/eigen_archive"
  "${TFDIR}/bazel-tensorflow/external/com_google_absl"
  "${PT_DIR}"
  "${PT_DIR}/torch/csrc"
  "${PT_DIR}/torch/include"
  "${PT_DIR}/torch/lib/tmp_install/include"
  "${PYTHON_INCLUDE_DIR}"
)

set(TORCH_XLA_TEST_SOURCES
  main.cpp
  cpp_test_util.cpp
  test_async_task.cpp
  test_aten_xla_tensor.cpp
  test_ir.cpp
  test_mayberef.cpp
  test_op_by_op_executor.cpp
  test_replication.cpp
  test_tensor.cpp
  test_xla_util_cache.cpp
  torch_xla_test.cpp
)

add_executable(test_ptxla ${TORCH_XLA_TEST_SOURCES})

set(TGT_OPTS
  -Wno-sign-compare
  -Wno-deprecated-declarations
  -Wno-return-type
)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # The -fsized-deallocation is required for Clang to prevent an error on
  # pytorch pybind11 about the operator delete being called inappropriately.
  list(APPEND TGT_OPTS
    -Wno-macro-redefined
    -Wno-return-std-move
    -fsized-deallocation)
endif()

target_compile_options(test_ptxla PRIVATE ${TGT_OPTS})

add_dependencies(test_ptxla googletest)

ExternalProject_Get_Property(googletest BINARY_DIR)

file(GLOB XLAC_LIBS "${PTXLA_LIBDIR}/_XLAC.*.so")
list(GET XLAC_LIBS 0 XLAC_LIBRARY)
message("Selected XLAC library ${XLAC_LIBRARY}")

# The linker does not like the _XLAC.cpython-36m-x86_64-linux-gnu.so name.
execute_process(COMMAND "ln" "-s" "-f"
  "${XLAC_LIBRARY}"
  "${PTXLA_LIBDIR}/libptxla.so")

find_library(PTXLA_LIB "libptxla.so"
  HINTS "${PTXLA_LIBDIR}")
find_library(PT_LIB "libtorch.so"
  HINTS "${PT_DIR}/build/lib")
find_library(PTPY_LIB "libtorch_python.so"
  HINTS "${PT_DIR}/build/lib")
find_library(C10_LIB "libc10.so"
  HINTS "${PT_DIR}/build/lib")
find_library(CAFFE_LIB "libcaffe2.so"
  HINTS "${PT_DIR}/build/lib")

# Use --unresolved-symbols=ignore-all to get around the c10::Half::from_bits
# undefined symbol error at link time. At runtime everything resolves correctly.
target_link_libraries(
  test_ptxla
  -Wl,--unresolved-symbols=ignore-in-shared-libs
  "${PTXLA_LIB}"
  "${PTXLA_LIBDIR}/torch_xla/lib/libxla_computation_client.so"
  "${PTPY_LIB}"
  "${PT_LIB}"
  "${CAFFE_LIB}"
  "${C10_LIB}"
  "${BINARY_DIR}/lib/${CMAKE_FIND_LIBRARY_PREFIXES}gtest.a"
  "${PYTHON_LIBRARY}"
  -lutil
  -pthread
  -lstdc++
  -ldl)
