# "@xla//tensorflow:tensorflow.bzl",
def ptxla_cc_shared_object(
        name,
        srcs = [],
        deps = [],
        data = [],
        linkopts = lrt_if_needed(),
        framework_so = tf_binary_additional_srcs(),
        soversion = None,
        kernels = [],
        per_os_targets = False,  # Generate targets with SHARED_LIBRARY_NAME_PATTERNS
        visibility = None,
        **kwargs):
    """Configure the shared object (.so) file for PyTorch/XLA."""
    if soversion != None:
        suffix = "." + str(soversion).split(".")[0]
        longsuffix = "." + str(soversion)
    else:
        suffix = ""
        longsuffix = ""

    if per_os_targets:
        names = [
            (
                pattern % (name, ""),
                pattern % (name, suffix),
                pattern % (name, longsuffix),
            )
            for pattern in SHARED_LIBRARY_NAME_PATTERNS
        ]
    else:
        names = [(
            name,
            name + suffix,
            name + longsuffix,
        )]

    testonly = kwargs.pop("testonly", False)

    for name_os, name_os_major, name_os_full in names:
        # Windows DLLs cant be versioned
        if name_os.endswith(".dll"):
            name_os_major = name_os
            name_os_full = name_os

        if name_os != name_os_major:
            native.genrule(
                name = name_os + "_sym",
                outs = [name_os],
                srcs = [name_os_major],
                output_to_bindir = 1,
                cmd = "ln -sf $$(basename $<) $@",
            )
            native.genrule(
                name = name_os_major + "_sym",
                outs = [name_os_major],
                srcs = [name_os_full],
                output_to_bindir = 1,
                cmd = "ln -sf $$(basename $<) $@",
            )

        soname = name_os_major.split("/")[-1]

        data_extra = []
        if framework_so != []:
            data_extra = tf_binary_additional_data_deps()

        cc_binary(
            exec_properties = if_google({"cpp_link.mem": "16g"}, {}),
            name = name_os_full,
            srcs = srcs + framework_so,
            deps = deps,
            linkshared = 1,
            data = data + data_extra,
            linkopts = linkopts + _rpath_linkopts(name_os_full) + select({
                clean_dep("//tensorflow:ios"): [
                    "-Wl,-install_name,@rpath/" + soname,
                ],
                clean_dep("//tensorflow:macos"): [
                    "-Wl,-install_name,@rpath/" + soname,
                ],
                clean_dep("//tensorflow:windows"): [],
                "//conditions:default": [
                    "-Wl,-soname," + soname,
                ],
            }),
            testonly = testonly,
            visibility = visibility,
            **kwargs
        )

    flat_names = [item for sublist in names for item in sublist]
    if name not in flat_names:
        native.filegroup(
            name = name,
            srcs = select({
                clean_dep("//tensorflow:windows"): [":%s.dll" % (name)],
                clean_dep("//tensorflow:macos"): [":lib%s%s.dylib" % (name, longsuffix)],
                "//conditions:default": [":lib%s.so%s" % (name, longsuffix)],
            }),
            visibility = visibility,
            testonly = testonly,
        )
#####