#!/bin/bash
set -ex
RUNDIR="$(cd "$(dirname "$0")" ; pwd -P)"
BUILDDIR="$RUNDIR/build"
BUILDTYPE="Release"
VERB=
FILTER=
BUILD_ONLY=0
RMBUILD=1
LOGFILE=/tmp/pytorch_cpp_test.log
XLA_EXPERIMENTAL="nonzero:masked_select"

if [ "$DEBUG" == "1" ]; then
  BUILDTYPE="Debug"
fi

while getopts 'VLDKBF:X:' OPTION
do
  case $OPTION in
    V)
      VERB="VERBOSE=1"
      ;;
    L)
      LOGFILE=
      ;;
    D)
      BUILDTYPE="Debug"
      ;;
    K)
      RMBUILD=0
      ;;
    B)
      BUILD_ONLY=1
      ;;
    F)
      FILTER="--gtest_filter=$OPTARG"
      ;;
    X)
      XLA_EXPERIMENTAL="$OPTARG"
      ;;
  esac
done
shift $(($OPTIND - 1))

if [[ "$TPUVM_MODE" != "1" ]]; then
  # Dynamic shape is not supported on the tpuvm.
  export XLA_EXPERIMENTAL
fi

rm -rf "$BUILDDIR"
mkdir "$BUILDDIR" 2>/dev/null
pushd "$BUILDDIR"
cmake "$RUNDIR" \
  -DCMAKE_BUILD_TYPE=$BUILDTYPE \
  -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR') + '/' + sysconfig.get_config_var('LDLIBRARY'))")
make -j $VERB

if [ $BUILD_ONLY -eq 0 ]; then
   BAZEL_FLAGS=
  if [ "$LOGFILE" != "" ]; then
    ./test_ptxla ${FILTER:+"$FILTER"} 2> $LOGFILE
     bazel test --spawn_strategy=sandboxed --sandbox_base=/dev/shm --show_progress_rate_limit=20 --define framework_shared_object=false -c opt --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=1 --config=tpu //third_party/xla_client/... 2> $LOGFILE
  else
    ./test_ptxla ${FILTER:+"$FILTER"} &
     bazel test --spawn_strategy=sandboxed --sandbox_base=/dev/shm --show_progress_rate_limit=20 --define framework_shared_object=false -c opt --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=1 --config=tpu //third_party/xla_client/...
  fi
fi

popd
if [ $RMBUILD -eq 1 -a $BUILD_ONLY -eq 0 ]; then
  rm -rf "$BUILDDIR"
fi
