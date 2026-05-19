#!/bin/bash
echo "$PATH"
exec llvm-cov gcov "$@"
