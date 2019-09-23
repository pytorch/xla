## C++ style guide

`pytorch/xla` uses `clang-format-7` with a customized style config.
If your PR touches the C++ source files, please run the following command before submmiting a PR.

```Shell
# If your PR only changes foo.cpp, run the following in xla/ folder
clang-format-7 -i -style /PATH/TO/foo.cpp
# To format all cpp files, run the follwoing in xla/ folder
find -name '*.cpp' -o -name '*.h' | xargs clang-format-7 -i -style=file
```

## Python style guide

`pytorch/xla` uses `yapf` with a customized style config.
If your PR touches the Python source files, please run the following command before submmiting a PR.

```Shell
#TODO:
```

