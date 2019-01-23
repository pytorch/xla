#!/bin/sh

VERB=
RMBUILD=1
LOGFILE=/tmp/pytorch_cpp_test.log

while getopts 'VLK' OPTION
do
  case $OPTION in
    V)
      VERB="VERBOSE=1"
      ;;
    L)
      LOGFILE=
      ;;
    K)
      RMBUILD=0
      ;;
  esac
done
shift $(($OPTIND - 1))

mkdir build 2>/dev/null
pushd build
cmake ..
make $VERB
if [ "$LOGFILE" != "" ]; then
  ./test_ptxla 2>$LOGFILE
else
  ./test_ptxla
fi
popd
if [ $RMBUILD -eq 1 ]; then
  rm -rf build
fi
