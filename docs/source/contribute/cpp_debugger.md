# C++ Debugging

PyTorch has good support for C++ debugging. This guide shows you how build with 
debugging symbols, and how to debug PyTorch using GDB or LLDB with VSCode. 

## Why would you need this?

This guide is for contributors to PyTorch or PyTorch/XLA who use C++ to build features like [custom C++ operations](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html). 

## Install GDB: The GNU Project Debugger

The official instructions are [here](https://www.sourceware.org/gdb/) but if you are in a conda environment, you can
install it with the following command:

``` sh
conda install -c conda-forge gdb
```

## Building PyTorch with Debugging Symbols

Looking into [`setup.py`]((https://github.com/pytorch/pytorch/blob/300e0ee13c08ef77e88f32204a2e0925c17ce216/setup.py#L2C1-L11C53)), we see

``` python
# Environment variables you are probably interested in:
#
#   DEBUG
#     build with -O0 and -g (debug symbols)
#
#   REL_WITH_DEB_INFO
#     build with optimizations and -g (debug symbols)
#
#   USE_CUSTOM_DEBINFO="path/to/file1.cpp;path/to/file2.cpp"
#     build with debug info only for specified files
```

Setting `DEBUG` will result in a slow binary, since every single file will
be unoptimized (`-O0`) and built with debugging symbols. 

In practice, you'll want to set the `USE_CUSTOM_DEBINFO` instead. 

The challenge becomes knowing which files to build with debugging symbols. A common issue will be debugging a file, then realizing you want to look at data structures or another
function call in a file that's not built with debugging symbols. 

Your basic flow as you get started is:

1. Identify the source file you want to debug.
2. Build that file with debugger symbols using a command similar to `USE_CUSTOM_DEBINFO="aten/src/ATen/native/Linear.cpp" python setup.py develop`.
3. Start a debugger session with a breakpoint. You'll discover additional files you want to debug. 
4. Add those files to the `USE_CUSTOM_DEBINFO`.
5. Rebuild those files.
6. Rebuild with a command similar to `USE_CUSTOM_DEBINFO="aten/src/ATen/native/Linear.cpp;newfile.cpp" python setup.py develop`.
7. Start your debugger session again. 

Again, the key line environment variable to set is `USE_CUSTOM_DEBINFO`. At this point, your PyTorch
is built with debugging symbols and ready for GDB command line to debug! However, we recommend
debugging with VSCode, and we will review the instructions to integrate with VSCode shortly. 

### Ensuring your file is built

For PyTorch, in our experience, you'll need to do a full clean and rebuild each time. Unfortunately, touching a file to update its timestamp does not seem to reliably rebuild that file. 

In the output of the `python setup.py develop` command, look for a line that starts with `Source files with custom debug infos`
to make sure your file is correctly being built with debugger symbols. For example:

``` sh
--   USE_ROCM_KERNEL_ASSERT : OFF
-- Performing Test HAS_WMISSING_PROTOTYPES
-- Performing Test HAS_WMISSING_PROTOTYPES - Failed
-- Performing Test HAS_WERROR_MISSING_PROTOTYPES
-- Performing Test HAS_WERROR_MISSING_PROTOTYPES - Failed
-- Source files with custom debug infos: aten/src/ATen/Utils.cpp aten/src/ATen/ScalarOps.cpp aten/src/ATen/EmptyTensor.cpp aten/src/ATen/core/Tensor.cpp aten/src/ATen/native/Linear.cpp
```

It may also be helpful to ensure that your previous debugger session is shut down.  

## VSCode Configuration

Integrating with VSCode will require installing the C/C++ Extension Pack, published by Microsoft. Search in the Extensions tab of VSCode with the identifier `ms-vscode.cpptools-extension-pack`.

![Microsoft C/C++ Extension Pack](../_static/img/debugger0_pack.png)

Next, you'll need to create a `launch.json` in the `pytorch/.vscode` folder. Here is
a sample file to start from. You'll need to adjust the file paths to match your specific installation.

``` json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "GDB a python interpreter",
            "type": "cppdbg",
            "request": "launch",
            "program": "/home/USERNAME/miniconda3/envs/torch310/bin/python", // Replace with your executable's path
            "args": [],
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                },
                {
                    "description": "Set Disassembly flavor to Intel",
                    "text": "-gdb-set disassembly-flavor intel",
                    "ignoreFailures": true
                }
            ],
            // "preLaunchTask": "C/C++: g++ build active file",
            "miDebuggerPath": "/home/USERNAME/miniconda3/envs/torch310/bin/gdb" // Replace with your gdb location
        }
    ]
 }
 ```

## LLDB

You can also debug with LLDB instead of GDB. In our experience, LLDB is faster. But both have their quirks, and you may benefit from switching between the two.

Install LLDB for VSCode by searching in the Extensions tab of VSCode with the identifier `vadimcn.vscode-lldb`.

Then add this to your launch.json file:

``` json
    {
        "name": "LLDB Python",
        "type": "lldb",
        "request": "launch",
        "program": "/home/yho_google_com/miniconda3/envs/torch310/bin/python",
        
    }
```


## Using VSCode's Visual Debugger

First, open the file you want to debug.

![Opened File](../_static/img/debugger1_file.png)

Second, click to the left of the line number you want to set a breakpoint. You'll see a red dot there, and, in the lower left of the debugger tab.

![Breakpoint](../_static/img/debugger2_breakpoint.png)

Third, you will start the debugger session. Select the GDB or LLDB configuration using the dropdown to the right of the green play button. Now press the green play button. This triggers the rules we
set in the launch.json file. 

Notice that the red button now fades to an empty white circle. That's because you have an active session, but you have not imported torch yet, so the torch library you built with debugger symbols is not yet loaded. 

![Debugger Session](../_static/img/debugger3_session.png)

In the python interpretor, `import torch`. This loads the torch library you built. Now, the breakpoints become bright red again to indicate that they are active. 

![Active torch](../_static/img/debugger4_active.png)

Run a command that will trigger the breakpoint, namely, `torch.einsum()`. Notice the yellow arrow indicating the current file location, and the variables and call stack information on the left. You are debugging! 

![Breakpoint](../_static/img/debugger5_break.png)

## A realistic build command

Over time, you'll collect additional flags to your build command, such as not building certain libraries to speed up builds, or adding diagnostics. Here is one example that turns off building CUDA and turns on handling an environment variable to print dispatcher traces. 

``` sh
USE_CUSTOM_DEBINFO="aten/src/ATen/native/Linear.cpp" USE_CUDA=0 LD_LIBRARY_PATH=/home/USERNAME/miniconda3/envs/torch310/lib CFLAGS="-DHAS_TORCH_SHOW_DISPATCH_TRACE" python setup.py develop
```

## Future Work

This guide has demonstrated how to build debugging symbols for PyTorch C++ files. Future work will demonstrate adding debugging symbols to PyTorch/XLA C++ files. 