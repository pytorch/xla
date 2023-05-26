"""Macros for working with openxla deps."""

def if_with_tpu_support(if_true, if_false = []):
    """Shorthand for select()ing whether to build API support for TPUs when building OpenXLA"""
    return select({
        "//conditions:default": if_true,
        "//conditions:default": if_false,
    })
