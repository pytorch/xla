# copy from "@xla//tensorflow:tensorflow.bzl",
load(
    "@tsl/tsl/platform/default:rules_cc.bzl",
    "cc_binary",
}

ptxla_cc_shared_object = rule(
    implementation = _ptxla_cc_shared_objectl,
)

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
    """Configure the shared object (.so) file for TensorFlow."""
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

    # names = [(name, name, name)]

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
        # soname = name

        data_extra = []
        if framework_so != []:
            data_extra = tf_binary_additional_data_deps()
        # data_extra = 

        cc_binary(
            exec_properties = if_google({"cpp_link.mem": "16g"}, {}),
            name = name_os_full, # name
            srcs = srcs + framework_so,
            deps = deps,
            linkshared = 1,
            data = data + data_extra,
            linkopts = linkopts + _rpath_linkopts(name_os_full) + ["-Wl,-soname," + soname,],
            testonly = testonly,
            visibility = visibility,
            **kwargs
        )

    flat_names = [item for sublist in names for item in sublist]
    if name not in flat_names:
        native.filegroup(
            name = name,
            srcs = [":lib%s.so%s" % (name, longsuffix)],
                # ":libname.so"
            visibility = visibility,
            testonly = testonly,
        )


# Bazel-generated shared objects which must be linked into TensorFlow binaries
# to define symbols from //tensorflow/core:framework and //tensorflow/core:lib.
def tf_binary_additional_srcs(fullversion = False):
    if fullversion:
        suffix = "." + VERSION
    else:
        suffix = "." + VERSION_MAJOR
    # suffix = .2

    return if_static(
        extra_deps = [],
        macos = [
            clean_dep("//tensorflow:libtensorflow_framework.2.dylib"),
        ],
        otherwise = [
            clean_dep("//tensorflow:libtensorflow_framework.so.2"),
        ],
    )

def tf_binary_additional_data_deps():
    return if_static(
        extra_deps = [],
        macos = [
            clean_dep("//tensorflow:libtensorflow_framework.dylib"),
            clean_dep("//tensorflow:libtensorflow_framework.2.dylib"),
            clean_dep("//tensorflow:libtensorflow_framework.2.13.0.dylib"),
        ],
        otherwise = [
            clean_dep("//tensorflow:libtensorflow_framework.so"),
            clean_dep("//tensorflow:libtensorflow_framework.so.2"),
            clean_dep("//tensorflow:libtensorflow_framework.so.2.13.0"),
        ],
    )

def _make_search_paths(prefix, levels_to_root):
    return ",".join(
        [
            "-rpath,%s/%s" % (prefix, "/".join([".."] * search_level))
            for search_level in range(levels_to_root + 1)
        ],
    )

def _rpath_linkopts(name):
    # Search parent directories up to the TensorFlow root directory for shared
    # object dependencies, even if this op shared object is deeply nested
    # (e.g. tensorflow/contrib/package:python/ops/_op_lib.so). tensorflow/ is then
    # the root and tensorflow/libtensorflow_framework.so should exist when
    # deployed. Other shared object dependencies (e.g. shared between contrib/
    # ops) are picked up as long as they are in either the same or a parent
    # directory in the tensorflow/ tree.

    levels_to_root = native.package_name().count("/") + name.count("/")
    return ["-Wl,%s" % (_make_search_paths("$$ORIGIN", levels_to_root),),

#####
