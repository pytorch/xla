cmake_minimum_required(VERSION 3.0)

if( NOT CXX )
  set(CMAKE_CXX_COMPILER clang++)
endif()

set(GTEST_DIR "${CMAKE_BINARY_DIR}/gtest")

get_filename_component(PTXLA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
get_filename_component(PT_DIR "${PTXLA_DIR}/.." ABSOLUTE)
set(TFDIR "${PTXLA_DIR}/third_party/tensorflow")

file(GLOB PTXLA_LIBDIRS "${PTXLA_DIR}/build/lib.*")
list(GET PTXLA_LIBDIRS 0 PTXLA_LIBDIR)
message("Selected PT/XLA library folder ${PTXLA_LIBDIR}")

execute_process(COMMAND
  python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  OUTPUT_VARIABLE PT_CXX_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

project(ptxla_test)

find_package(PythonLibs)
set(Torch_DIR "${PT_DIR}/torch/share/cmake/Torch")
find_package(Torch REQUIRED)

include(ExternalProject)
set_directory_properties(PROPERTIES EP_PREFIX "${GTEST_DIR}")

ExternalProject_Add(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG e2239ee6043f73722e7aa812a459f54a28552929
  SOURCE_DIR "${GTEST_DIR}/src/googletest-src"
  BINARY_DIR "${GTEST_DIR}/src/googletest-build"
  # Disable install step
  INSTALL_COMMAND ""
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  CMAKE_ARGS
    "-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${PT_CXX_ABI}")

ExternalProject_Get_Property(googletest SOURCE_DIR)

set(TORCH_XLA_TEST_SOURCES
  main.cpp
  cpp_test_util.cpp
  metrics_snapshot.cpp
  test_async_task.cpp
  test_aten_xla_tensor.cpp
  test_ir.cpp
  test_mayberef.cpp
  test_op_by_op_executor.cpp
  test_replication.cpp
  test_tensor.cpp
  test_xla_util_cache.cpp
  torch_xla_test.cpp
  test_xla_backend_intf.cpp
)

add_executable(test_ptxla ${TORCH_XLA_TEST_SOURCES})

set(TGT_OPTS
  -D_GLIBCXX_USE_CXX11_ABI=${PT_CXX_ABI}
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

target_include_directories(
  test_ptxla
  PRIVATE
  "${PTXLA_DIR}"
  "${PTXLA_DIR}/torch_xla/csrc"
)
target_include_directories(
  test_ptxla
  SYSTEM PUBLIC
  "${SOURCE_DIR}/googletest/include"
  "${TFDIR}/bazel-tensorflow"
  "${TFDIR}/bazel-bin"
  "${TFDIR}/bazel-tensorflow/external/protobuf_archive/src"
  "${TFDIR}/bazel-tensorflow/external/com_google_protobuf/src"
  "${TFDIR}/bazel-tensorflow/external/eigen_archive"
  "${TFDIR}/bazel-tensorflow/external/com_google_absl"
  "${PYTHON_INCLUDE_DIR}"
)

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
find_library(PTPY_LIB "libtorch_python.so"
  HINTS "${PT_DIR}/torch/lib")

# Use --unresolved-symbols=ignore-all to get around the c10::Half::from_bits
# undefined symbol error at link time. At runtime everything resolves correctly.
target_link_libraries(
  test_ptxla
  -Wl,--unresolved-symbols=ignore-in-shared-libs
  "${TORCH_LIBRARIES}"
  "${PTXLA_LIB}"
  "${PTXLA_LIBDIR}/torch_xla/lib/libxla_computation_client.so"
  "${PTPY_LIB}"
  "${BINARY_DIR}/lib/${CMAKE_FIND_LIBRARY_PREFIXES}gtest.a"
  "${PYTHON_LIBRARY}"
  -lutil
  -pthread
  -lstdc++
  -ldl)
