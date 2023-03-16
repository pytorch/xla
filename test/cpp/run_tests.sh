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

while getopts 'VLDKBF:X:R' OPTION
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
    K)
      RMBUILD=0
      ;;
    B)
      BUILD_ONLY=1
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

# Inherit env flags for tests.
EXTRA_FLAGS="--test_env=XRT_DEVICE_MAP --test_env=XRT_WORKERS --test_env=XRT_TPU_CONFIG --test_env=GPU_NUM_DEVICES --test_env=PJRT_DEVICE"

# Set up coverage flags.
if [[ "$BAZEL_VERB" == "coverage" ]]; then
  EXTRA_FLAGS="$EXTRA_FLAGS --combined_report=lcov"
fi

# Handle remote builds and remote cache. Use a CI-private cache silo to avoid cachce polution.
if [[ "$BAZEL_REMOTE_CACHE" == "1" ]]; then
  EXTRA_FLAGS="$EXTRA_FLAGS --config=remote_cache"
  if [[ ! -z "$GCLOUD_SERVICE_KEY_FILE" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --google_credentials=$GCLOUD_SERVICE_KEY_FILE"
  fi
  if [[ ! -z "$SILO_NAME" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --remote_default_exec_properties=cache-silo-key=$SILO_NAME"
    EXTRA_FLAGS="$EXTRA_FLAGS --host_platform_remote_properties_override=properties:{name:\"cache-silo-key\" value:\"$SILO_NAME\"}"
  fi
fi

if [ $BUILD_ONLY -eq 0 ]; then
  if [ "$LOGFILE" != "" ]; then
    bazel $BAZEL_VERB $EXTRA_FLAGS --test_output=all //third_party/xla_client:all //test/cpp:all ${FILTER:+"$FILTER"} 2> $LOGFILE
  else 
    bazel $BAZEL_VERB $EXTRA_FLAGS --test_output=all //third_party/xla_client:all //test/cpp:all ${FILTER:+"$FILTER"}
  fi
fi
