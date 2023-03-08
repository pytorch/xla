"""Macros for working with tensorflow deps."""

def if_with_tpu_support(if_true, if_false = []):
    """Shorthand for select()ing whether to build API support for TPUs when building TensorFlow"""
    return select({
        "@org_tensorflow//tensorflow:with_tpu_support": if_true,
        "//conditions:default": if_false,
    })
