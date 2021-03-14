cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(pytorch_xla CXX C)
#set(CMAKE_CXX_STANDARD 17)

option(BUILD_PTXLA_LIB "Build ptxla lib and link _XLAC to it as private, rather than making _XLAC one big library" OFF)

include_directories(${PYTHON_INCLUDE_DIR})

message(STATUS "PYTHON_VERSION_STRING=${PYTHON_VERSION_STRING}")

execute_process( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )
execute_process( COMMAND uname -s COMMAND tr -d '\n' OUTPUT_VARIABLE KERNEL_NAME_MIXED_CASE )
string(TOLOWER "${KERNEL_NAME_MIXED_CASE}" KERNEL_NAME)
message( STATUS "Architecture: ${ARCHITECTURE}-${KERNEL_NAME}" )
set(ARCH_BUILD_DIR_NAME "lib.${KERNEL_NAME}-${ARCHITECTURE}-${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
set(CPYTHON_STR "cpython-${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}m-${ARCHITECTURE}-${KERNEL_NAME}-gnu")

message(STATUS "CPYTHON_STR=${CPYTHON_STR}")

file(GLOB
        XLA_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/torch_xla/csrc/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/torch_xla/csrc/ops/*.cpp)

file(GLOB
        XLA_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/torch_xla/csrc/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/torch_xla/csrc/ops/*.h)

if(NOT TORCH_PATH)
    message(ERROR "TORCH_DIR needs to point to the pytorch source directory")
endif()
message(STATUS "TORCH_PATH: ${TORCH_PATH}")

add_compile_definitions(XLA_CUDA=1)

include_directories(${TORCH_PATH})
include_directories(${TORCH_PATH}/torch/csrc)
include_directories(${TORCH_PATH}/torch/lib/tmp_install/include)
include_directories(${TORCH_PATH}/torch/include)
include_directories(SYSTEM ${TORCH_PATH}/torch/include/TH)

include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorflow/bazel-tensorflow)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorflow/bazel-bin)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorflow/tensorflow)
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

set(TFDIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorflow)

include_directories(BEFORE ${TFDIR}/bazel-tensorflow/external/com_google_protobuf/src)
include_directories(BEFORE ${TFDIR}/bazel-tensorflow/external/eigen_archive)
include_directories(BEFORE ${TFDIR}/bazel-tensorflow/external/com_google_absl/)
include_directories(BEFORE ${TFDIR}/bazel-tensorflow/external/com_github_grpc_grpc/include)
include_directories(BEFORE ${TFDIR}/bazel-tensorflow/external/llvm-project/llvm/include)
include_directories(BEFORE ${TFDIR}/bazel-out/host/bin/external/llvm-project/llvm/include)
include_directories(BEFORE ${TFDIR}/bazel-out/k8-opt/bin/external/llvm-project/llvm/include)
include_directories(BEFORE ${TFDIR}/bazel-tensorflow/external/llvm-project/mlir/include)
include_directories(BEFORE ${TFDIR}/bazel-out/host/bin/external/llvm-project/mlir/include)
include_directories(BEFORE ${TFDIR}/bazel-out/k8-opt/bin/external/llvm-project/mlir/include)
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorflow/tensorflow/compiler/mlir/hlo/include)
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorflow)

message(STATUS "${XLAC_EXTENSION_TARGET} CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}")

add_compile_options(-O0)
add_compile_options(-ggdb3)
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=1)
add_compile_options(-DTORCH_API_INCLUDE_EXTENSION_H)
add_compile_options(-DTORCH_EXTENSION_NAME=_XLAC)
add_compile_options(-D_LARGEFILE_SOURCE)
add_compile_options(-D_LARGEFILE64_SOURCE)
add_compile_options(-D_FILE_OFFSET_BITS=64)
set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")

# -Namespaces
add_compile_options(-Dtensorflow=ptxla_tf)
add_compile_options(-Dtensorflow=ptxla_tf)
add_compile_options(-DEigen=ptxla_Eigen)
add_compile_options(-Dgrpc=ptxla_grpc)
add_compile_options(-Dgrpc_impl=ptxla_grpc_impl)
add_compile_options(-Dgoogle=ptxla_google)
add_compile_options(-Dllvm=ptxla_llvm)
add_compile_options(-Dmlir=ptxla_mlir)
# Namespaces-

set(CMAKE_CXX_FLAGS "-fPIC ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-fwrapv ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-Wno-sign-compare ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-Wno-return-type ${CMAKE_CXX_FLAGS}")
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "-Wno-macro-redefined ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "-Wno-return-std-move ${CMAKE_CXX_FLAGS}")
else()
    set(CMAKE_CXX_FLAGS "-Wignored-qualifiers ${CMAKE_CXX_FLAGS}")
endif()
set(CMAKE_CXX_FLAGS "-DTORCH_API_INCLUDE_EXTENSION_H ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-DTORCH_EXTENSION_NAME=_XLAC ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1 ${CMAKE_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "-DTF_HAS_WSE_DEVICE=1 ${CMAKE_CXX_FLAGS}")

#add_compile_options(-fvisibility=hidden)
#add_link_options(-fvisibility=hidden)
add_compile_options(-Wno-ignored-qualifiers)


set(BUILD_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build/${ARCH_BUILD_DIR_NAME})


#
# ptxla library
#

if (EXISTS ${CMAKE_CURRENT_LIST_DIR}/sources.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/sources.cmake)
endif()

link_directories(${CMAKE_BINARY_DIR}/lib)
link_directories(${TORCH_PATH}/torch/lib)

set(XLAC_EXTENSION_TARGET _XLAC.${CPYTHON_STR})

set(USE_TORCH_LIBRARIES torch torch_python c10 shm torch_global_deps)
set(OTHER_TORCH_DEPS torch_shm_manager _C _dl)

if (BUILD_PTXLA_LIB)

  target_link_libraries(
          ptxla
          PUBLIC
          -Wl,--unresolved-symbols=ignore-in-shared-libs
          "${USE_TORCH_LIBRARIES}"
          "${PYTHON_LIBRARY}"
          -lutil
          -pthread
          -lstdc++
          -latomic
  )

  add_library(ptxla SHARED ${XLA_SOURCES} ${XLA_HEADERS})
  add_dependencies(ptxla ${OTHER_TORCH_DEPS})

  set_target_properties(
          ptxla
          PROPERTIES
          LIBRARY_OUTPUT_DIRECTORY
          ${BUILD_OUTPUT_DIR}/torch_xla/lib
  )
  
  # Link into develop dir position
  add_custom_command(
          TARGET ptxla POST_BUILD
          COMMAND mkdir -p ${CMAKE_CURRENT_SOURCE_DIR}/build/lib.linux-x86_64-3.7
          COMMAND mkdir -p ${CMAKE_CURRENT_SOURCE_DIR}/torch_xla/lib
          COMMAND ln -sf $<TARGET_FILE:ptxla> ${CMAKE_CURRENT_SOURCE_DIR}/torch_xla/lib/
  )

  #
  # _XLAC Python Extension Library
  #
  add_library(
          ${XLAC_EXTENSION_TARGET}
          ${CMAKE_CURRENT_SOURCE_DIR}/torch_xla/csrc/stub.cpp
  )

  target_link_libraries(
          ${XLAC_EXTENSION_TARGET}
          PUBLIC
          ptxla
  )
else()
  add_library(${XLAC_EXTENSION_TARGET} SHARED ${XLA_SOURCES} ${XLA_HEADERS})
  target_link_libraries(
          ${XLAC_EXTENSION_TARGET}
          PRIVATE
          -Wl,--unresolved-symbols=ignore-in-shared-libs
          "${USE_TORCH_LIBRARIES}"
          "${PYTHON_LIBRARY}"
          -lutil
          -pthread
          -lstdc++
          -latomic
  )
if (BUILD_PYTORCH)
  add_dependencies(${XLAC_EXTENSION_TARGET} ${OTHER_TORCH_DEPS})
endif()
endif(BUILD_PTXLA_LIB)

set_target_properties(
        ${XLAC_EXTENSION_TARGET}
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
        ${BUILD_OUTPUT_DIR}
)

set_target_properties(${XLAC_EXTENSION_TARGET} PROPERTIES BUILD_RPATH "$ORIGIN")
set_target_properties(${XLAC_EXTENSION_TARGET} PROPERTIES BUILD_RPATH "$ORIGIN/lib")

#
# _XLAC extension
#

set(CMAKE_VERBOSE_MAKEFILE ON)

set_target_properties(${XLAC_EXTENSION_TARGET} PROPERTIES OUTPUT_NAME "${XLAC_EXTENSION_TARGET}")
set_target_properties(${XLAC_EXTENSION_TARGET} PROPERTIES PREFIX "")

add_custom_command(
        TARGET ${XLAC_EXTENSION_TARGET} POST_BUILD
        COMMAND ln -sf $<TARGET_FILE:${XLAC_EXTENSION_TARGET}> ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(
        xlac ALL
        COMMAND echo  "xlac built"
        DEPENDS ${XLAC_EXTENSION_TARGET}
)

set(CMAKE_EXE_LINKER_FLAGS_DEBUG:STRING="-Wl,--gdb-index" ${CMAKE_EXE_LINKER_FLAGS_DEBUG})

if (BUILD_TENSORFLOW)
    set(LIBXLA_COMPUTATION_CLIENT_LIB_FILE_NAME "libxla_computation_client.so")
    set(LIBXLA_COMPUTATION_CLIENT_LIB_ACTUAL_FILE_NAME ${LIBXLA_COMPUTATION_CLIENT_LIB_FILE_NAME})
    list(APPEND TENSORFLOW_TARGETS "//tensorflow/compiler/xla/xla_client:${LIBXLA_COMPUTATION_CLIENT_LIB_FILE_NAME}" PARENT_SCOPE)

    set(LIBXLA_COMPUTATION_CLIENT_LIB "${CMAKE_CURRENT_SOURCE_DIR}/torch_xla/lib/${LIBXLA_COMPUTATION_CLIENT_LIB_ACTUAL_FILE_NAME}")
    message(STATUS "LIBXLA_COMPUTATION_CLIENT_LIB: ${LIBXLA_COMPUTATION_CLIENT_LIB}")

    target_link_libraries(${XLAC_EXTENSION_TARGET} PRIVATE ${LIBXLA_COMPUTATION_CLIENT_LIB})
    target_link_libraries(${XLAC_EXTENSION_TARGET} PRIVATE ${FRAMEWORK_LIBS})

    set(TF_SOURCE_FILE "${TF_DIRECTORY}/bazel-bin/tensorflow/compiler/xla/xla_client/${LIBXLA_COMPUTATION_CLIENT_LIB_ACTUAL_FILE_NAME}")
    set(TF_DEST_FILE "${CMAKE_CURRENT_LIST_DIR}/torch_xla/lib/${LIBXLA_COMPUTATION_CLIENT_LIB_ACTUAL_FILE_NAME}")

    message(STATUS "TO LINK: ${TF_SOURCE_FILE} -> ${TF_DEST_FILE}")

    add_custom_target(
            libxla_computation_client_mkdir
            COMMAND "mkdir" "-p"
            "${CMAKE_CURRENT_SOURCE_DIR}/torch_xla/lib"
            DEPENDS libxla_computation_client_build
    )

    add_custom_target(
            libxla_computation_client_make_writable
            COMMAND "chmod" "+w" "${TF_SOURCE_FILE}"
            DEPENDS libxla_computation_client_build
    )
    add_custom_target(
            libxla_computation_client_gdb_add_index
            COMMAND "gdb-add-index"
            "${TF_SOURCE_FILE}"
            DEPENDS libxla_computation_client_make_writable
    )
    add_custom_target(
            libxla_computation_client
            BYPRODUCTS "${TF_DEST_FILE}"
            COMMAND "ln" "-s" "-f"
            "${TF_SOURCE_FILE}"
            "${TF_DEST_FILE}"
            DEPENDS libxla_computation_client_gdb_add_index libxla_computation_client_mkdir libxla_computation_client_build
    )

    add_dependencies(${XLAC_EXTENSION_TARGET} libxla_computation_client)

else()
    message(STATUS "Not building TensorFlow -- assuming the library exists: ${LIBXLA_COMPUTATION_CLIENT_LIB}")
endif()

#add_subdirectory(test/cpp)
#add_dependencies(xlac test_ptxla)
