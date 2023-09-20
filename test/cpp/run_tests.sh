#!/bin/bash
set -ex
RUNDIR="$(cd "$(dirname "$0")" ; pwd -P)"
BUILDTYPE="opt"
VERB=
FILTER=
LOGFILE=/tmp/pytorch_cpp_test.log
XLA_EXPERIMENTAL="nonzero:masked_select"
BAZEL_REMOTE_CACHE="0"
BAZEL_VERB="test"

# See Note [Keep Going]
CONTINUE_ON_ERROR=false
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  set +e
fi

if [ "$DEBUG" == "1" ]; then
  BUILDTYPE="dbg"
fi

while getopts 'VLDKBF:X:RC' OPTION
do
  case $OPTION in
    V)
      VERB="VERBOSE=1"
      ;;
    L)
      LOGFILE=
      ;;
    D)
      BUILDTYPE="dbg"
      ;;
    F)
      FILTER="--test_filter=$OPTARG"
      ;;
    X)
      XLA_EXPERIMENTAL="$OPTARG"
      ;;
    R)
      BAZEL_REMOTE_CACHE="1"
      ;;
    C)
      BAZEL_VERB="coverage"
      ;;
  esac
done
shift $(($OPTIND - 1))

# Set XLA_EXPERIMENTAL var to subsequently executed commands.
export XLA_EXPERIMENTAL

EXTRA_FLAGS=""

if [[ "$TPUVM_MODE" == "1" ]]; then
  EXTRA_FLAGS="$EXTRA_FLAGS --config=tpu"
fi

# Match CXX_ABI flags with XLAC.so build
if [[ -n "${CXX_ABI}" ]]; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=${CXX_ABI}"
fi

# Override BAZEL_REMOTE_CACHE if GLOUD_SERVICE_KEY_FILE is 1 byte
if [ ! -z "$GCLOUD_SERVICE_KEY_FILE" ]; then
  file_size=$(stat -c%s "$GCLOUD_SERVICE_KEY_FILE")
  if [ "$file_size" -le 1 ]; then
    BAZEL_REMOTE_CACHE=0
  fi
fi
# Handle remote builds and remote cache. Use a CI-private cache silo to avoid cache pollution.
if [[ "$BAZEL_REMOTE_CACHE" == "1" ]]; then
  EXTRA_FLAGS="$EXTRA_FLAGS --config=remote_cache"
  if [[ ! -z "$GCLOUD_SERVICE_KEY_FILE" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --google_credentials=$GCLOUD_SERVICE_KEY_FILE"
  fi
  if [[ ! -z "$SILO_NAME" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --remote_default_exec_properties=cache-silo-key=$SILO_NAME"
  fi
fi
if [[ "$XLA_CUDA" == "1" ]]; then
  EXTRA_FLAGS="$EXTRA_FLAGS --config=cuda"
fi
if [[ "$BAZEL_VERB" == "coverage" ]]; then
  EXTRA_FLAGS="$EXTRA_FLAGS --remote_download_outputs=all" # for lcov symlink
fi

test_names=("all")
if [[ "$RUN_CPP_TESTS1" == "cpp_tests1" ]]; then
  test_names=("test_aten_xla_tensor_1"
              "test_aten_xla_tensor_2"
              "test_aten_xla_tensor_3"
              "test_aten_xla_tensor_4")
elif [[ "$RUN_CPP_TESTS2" == "cpp_tests2" ]]; then
  test_names=("test_aten_xla_tensor_5"
              "test_aten_xla_tensor_6"
              "test_ir"
              "test_lazy"
              "test_replication"
              "test_tensor"
              "test_xla_backend_intf"
              "test_xla_sharding")
fi
for name in "${test_names[@]}"; do
  if [ "$LOGFILE" != "" ]; then
    bazel $BAZEL_VERB $EXTRA_FLAGS //torch_xla/csrc/runtime:all //test/cpp:$name --test_timeout 1000 ${FILTER:+"$FILTER"} 2> $LOGFILE
  else
    bazel $BAZEL_VERB $EXTRA_FLAGS //torch_xla/csrc/runtime:all //test/cpp:$name --test_timeout 1000 ${FILTER:+"$FILTER"}
  fi
done

