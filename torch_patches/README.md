# Guidelines For Patch File Names

The only files which are considered by the apply script are the ones with extension '.diff'.

A file for PyTorch PR _N_ needs to be named 'N.diff'.

Patch files which are not related to PyTorch PRs, should begin with an 'X' character,
followed by a two digit number, followed by a dash ('-'), a name, and '.diff'.
Example:

```
X10-optimizer.diff
```

Patch file are alphabetically ordered, so PyTorch PR patches are always applied
before the non PyTorch ones.

