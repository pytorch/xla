#!/usr/bin/python

from __future__ import print_function

import argparse
import collections
import lark
import os
import re
import sys

FuncGen = collections.namedtuple(
    'FuncGen', 'func, xfunc, code, sig, cppsig, funsig, mapsig')

_PARSER = lark.Lark(
    r"""
    start: type fnname "(" params ")"
    type: CONST? core_type refspec?
    fnname: CNAME
    refspec: REF
           | PTR
    core_type: template
        | TNAME
    template: TNAME "<" typelist ">"
    typelist: type
            | type "," typelist
    REF: "&"
    PTR: "*"
    CONST: "const"
    TNAME: /[a-zA-Z0-9_:]+/
    params: param
          | param "," params
    param: type CNAME

    %import common.CNAME -> CNAME
    %import common.WS
    %ignore WS
    """,
    parser='lalr',
    propagate_positions=True)

_FN_BLACKLIST = set([
    # ATEN functions
    'toBackend',
    'toScalarType',
    'copy',
    'copy_',
    'backward',
    'tensorFromBlob',
    'tensorWithAllocator',
    'storageFromBlob',
    'storageWithAllocator',
    'unsafeStorageFromTH',
    'unsafeTensorFromTH',
    # XLA/TPU functions
])


def type_core(t):
  assert isinstance(t, lark.tree.Tree)
  for c in t.children:
    if isinstance(c, lark.tree.Tree) and c.data == 'core_type':
      c = c.children[0]
      if isinstance(c, lark.lexer.Token):
        return c.value
      assert isinstance(c, lark.tree.Tree) and c.data == 'template'
      return c.children[0].value
  raise RuntimeError('Not a type tree: {}'.format(t))


def type_is_const(t):
  assert isinstance(t, lark.tree.Tree)
  c = t.children[0]
  return isinstance(c, lark.lexer.Token) and c.value == 'const'


def type_is_refptr(t, kind):
  assert isinstance(t, lark.tree.Tree)
  c = t.children[-1]
  if not isinstance(c, lark.tree.Tree) or c.data != 'refspec':
    return False
  c = c.children[0]
  return isinstance(c, lark.lexer.Token) and c.value == kind


def extract_list(t, l):
  assert isinstance(t, lark.tree.Tree)
  l.append(t.children[0])
  if len(t.children) == 2:
    c = t.children[1]
    if isinstance(c, lark.tree.Tree) and c.data == t.data:
      extract_list(c, l)
  return l


def tuple_type_list(t):
  assert isinstance(t, lark.tree.Tree)
  c = t.children[0]
  assert isinstance(c, lark.tree.Tree) and c.data == 'core_type'
  c = c.children[0]
  assert isinstance(c, lark.tree.Tree) and c.data == 'template'
  types = []
  return extract_list(c.children[1], types)


def get_function_name(t):
  assert isinstance(t, lark.tree.Tree)
  fname = t.children[1]
  assert isinstance(fname, lark.tree.Tree)
  assert fname.data == 'fnname'
  return fname.children[0].value


def get_function_signature(t, orig_sig, namefn):
  assert isinstance(t, lark.tree.Tree)
  fname = t.children[1]
  assert isinstance(fname, lark.tree.Tree)
  assert fname.data == 'fnname'
  token = fname.children[0]
  assert isinstance(token, lark.lexer.Token)
  return (orig_sig[0:token.column - 1] + namefn(token.value) +
          orig_sig[token.end_column - 1:]), token.value


def get_parameters(t):
  assert isinstance(t, lark.tree.Tree)
  c = t.children[2]
  assert isinstance(c, lark.tree.Tree)
  assert c.data == 'params'
  params = []
  extract_list(c, params)
  return params


def list_get(l, n):
  return l[n] if n < len(l) else None


def get_return_value(rtype, rname, param, var, ref_param=None):
  if type_is_const(rtype) or type_is_refptr(rtype, '&'):
    assert param
    return param.children[1].value
  else:
    return 'CreateXlaTensor({}, DeviceFromTensor({}))'.format(
        rname, ref_param or param.children[1].value)


def get_tuple_return(rtype, rtype_str, rname, params, param_vars):
  types = tuple_type_list(rtype)
  ref_param = 'self'
  retstr = '{}('.format(rtype_str)
  for i, ttype in enumerate(types):
    if i > 0:
      retstr += ', '
    tuple_var = '{}.get<{}>()'.format(rname, i)
    retstr += get_return_value(
        ttype,
        tuple_var,
        list_get(params, i),
        list_get(param_vars, i),
        ref_param=ref_param)
  return retstr + ')'


def get_return_type_str(t, orig_sig):
  assert isinstance(t, lark.tree.Tree)
  fname = t.children[1]
  assert isinstance(fname, lark.tree.Tree)
  assert fname.data == 'fnname'
  token = fname.children[0]
  assert isinstance(token, lark.lexer.Token)
  return orig_sig[0:token.column - 2]


def generate_return_stmt(t, orig_sig, fname, rname, params, param_vars):
  assert isinstance(t, lark.tree.Tree)
  rtype = t.children[0]
  ctype = type_core(rtype)
  if ctype == 'std::tuple':
    rtype_str = get_return_type_str(t, orig_sig)
    retstr = get_tuple_return(rtype, rtype_str, rname, params, param_vars)
  elif ctype == 'Tensor':
    retstr = get_return_value(rtype, rname, params[0], param_vars[0])
  else:
    retstr = rname
  return '  return {};\n'.format(retstr)


def get_xla_wrapper(t, orig_sig):
  params = get_parameters(t)
  sig, fname = get_function_signature(t, orig_sig, lambda x: 'xla_' + x)
  code = 'static {} {{\n'.format(sig)
  param_vars = []
  for p in params:
    ptype = p.children[0]
    pname = p.children[1].value
    if type_core(ptype) == 'TensorList':
      xname = '_l_{}'.format(pname)
      code += '  auto {} = XlaCreateTensorList({});\n'.format(xname, pname)
      param_vars.append(xname)
    elif type_core(ptype) != 'Tensor':
      param_vars.append(pname)
    elif type_is_const(ptype):
      xname = '_r_{}'.format(pname)
      code += '  auto {} = {}.alias().ToTensor();\n'.format(xname, pname)
      param_vars.append(xname)
    else:
      xname = '_w_{}'.format(pname)
      code += '  auto {} = {}.alias().ToMutableTensor();\n'.format(xname, pname)
      param_vars.append(xname)
  code += '  auto&& __result = {}('.format(fname)
  for i, v in enumerate(param_vars):
    if i > 0:
      code += ', '
    code += v
  code += ');\n'
  code += '  (void) __result; // Avoid warnings in case not used\n'
  code += generate_return_stmt(t, orig_sig, fname, '__result', params,
                               param_vars)
  code += '}'
  return FuncGen(
      func=fname,
      xfunc='xla_{}'.format(fname),
      code=code,
      sig=orig_sig,
      cppsig=sig,
      funsig=None,
      mapsig=None)


def extract_functions(path):
  functions = []
  for line in open(path, 'r'):
    m = re.match(r'\s*([^\s].*) const override;', line)
    if not m:
      continue
    fndef = m.group(1)
    try:
      tree = _PARSER.parse(fndef)
      fname = get_function_name(tree)
      if fname not in _FN_BLACKLIST:
        functions.append(fndef)
    except:
      pass
  return functions


def generate(args, files):
  ofile = sys.stdout
  if args.output:
    ofile = open(args.output, 'w')
  for fname in files:
    fndefs = extract_functions(fname)
    print(
        'Extracted {} functions from {}'.format(len(fndefs), fname),
        file=sys.stderr)
    for ts in fndefs:
      tree = _PARSER.parse(ts)
      fgen = get_xla_wrapper(tree, ts)
      print('{}\n'.format(fgen.code), file=ofile)


if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--output', type=str)
  args, files = arg_parser.parse_known_args()
  generate(args, files)
