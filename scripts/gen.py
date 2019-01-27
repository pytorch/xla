#!/usr/bin/python

from __future__ import print_function

import argparse
import collections
import lark
import os
import re
import sys

FuncGen = collections.namedtuple(
    'FuncGen', 'tree, xtree, func, xfunc, code, sig, cppsig, funsig, mapsig')

_GRAMMAR = r"""
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
    param: type param_name
    param_name: CNAME

    %import common.CNAME -> CNAME
    %import common.WS
    %ignore WS
    """

_PARSER = lark.Lark(_GRAMMAR, parser='lalr', propagate_positions=True)
_XPARSER = lark.Lark(
    _GRAMMAR, parser='lalr', propagate_positions=True, keep_all_tokens=True)

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


def first_match(t):
  if isinstance(t, lark.lexer.Token):
    return t.column - 1
  assert isinstance(t, lark.tree.Tree)
  return first_match(t.children[0])


def last_match(t):
  if isinstance(t, lark.lexer.Token):
    return t.end_column - 1
  assert isinstance(t, lark.tree.Tree)
  return last_match(t.children[-1])


class StringEmit(object):

  def __init__(self, sref):
    self.sref = sref
    self.sval = ''
    self.pos = -1

  def __repr__(self):
    return self.sval

  def advance(self, t):
    start = first_match(t)
    end = last_match(t)
    pos = self.pos if self.pos >= 0 else start
    self.sval += self.sref[pos:end]
    self.pos = end

  def skip(self, t):
    self.pos = last_match(t) if self.pos >= 0 else -1

  def append(self, s):
    self.sval += s
    self.pos = -1


def list_get(l, n):
  return l[n] if n < len(l) else None


def for_every_token(t, fn):
  if isinstance(t, lark.lexer.Token):
    fn(t)
  else:
    assert isinstance(t, lark.tree.Tree)
    for c in t.children:
      for_every_token(c, fn)


def emit_string(t, emit, emit_fn):
  status = emit_fn(t)
  if status > 0:

    def do_emit(tok):
      emit.advance(tok)

    for_every_token(t, do_emit)
  elif status == 0:
    if isinstance(t, lark.lexer.Token):
      emit.advance(t)
    else:
      assert isinstance(t, lark.tree.Tree)
      for c in t.children:
        emit_string(c, emit, emit_fn)
  else:
    emit.skip(t)


def typed_child(t, n, ttype):
  assert isinstance(t, lark.tree.Tree)
  assert n < len(t.children)
  c = t.children[n]
  assert isinstance(c, lark.tree.Tree)
  assert c.data == ttype, t.pretty()
  return c


def create_stdfunc_sig(tree, orig_sig):

  def emit_fn(t):
    if isinstance(t, lark.lexer.Token):
      return 0
    return -1 if t.data == 'param_name' else 0

  emit = StringEmit(orig_sig)
  # Emit full function return type.
  emit_string(typed_child(tree, 0, 'type'), emit, emit_fn)
  emit.append('(')
  # Emit parameter list w/out parameter names.
  emit_string(typed_child(tree, 3, 'params'), emit, emit_fn)
  emit.append(')')
  return str(emit)


def create_map_sig(tree, orig_sig):

  def emit_fn(t):
    if isinstance(t, lark.lexer.Token):
      return -1 if t.type in ['CONST', 'REF', 'PTR'] else 0
    return -1 if t.data == 'param_name' else 0

  emit = StringEmit(orig_sig)
  # Emit full function return type.
  emit_string(typed_child(tree, 1, 'fnname'), emit, emit_fn)
  emit.append('(')
  # Emit parameter list w/out parameter names.
  emit_string(typed_child(tree, 3, 'params'), emit, emit_fn)
  emit.append(') -> ')
  emit_string(typed_child(tree, 0, 'type'), emit, emit_fn)
  return str(emit)


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


def param_name(t):
  assert isinstance(t, lark.tree.Tree)
  c = t.children[1]
  assert isinstance(c, lark.tree.Tree)
  assert c.data == 'param_name'
  token = c.children[0]
  assert isinstance(token, lark.lexer.Token)
  return token.value


def get_return_value(rtype, rname, param, var, ref_param=None):
  if type_is_const(rtype) or type_is_refptr(rtype, '&'):
    assert param
    return param_name(param)
  else:
    return 'CreateXlaTensor({}, DeviceFromTensor({}))'.format(
        rname, ref_param or param_name(param))


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


def get_xla_wrapper(orig_sig):
  tree = _PARSER.parse(orig_sig)
  xtree = _XPARSER.parse(orig_sig)
  params = get_parameters(tree)
  sig, fname = get_function_signature(tree, orig_sig, lambda x: 'xla_' + x)
  code = 'static {} {{\n'.format(sig)
  param_vars = []
  for p in params:
    ptype = p.children[0]
    pname = param_name(p)
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
  code += generate_return_stmt(tree, orig_sig, fname, '__result', params,
                               param_vars)
  code += '}'
  return FuncGen(
      tree=tree,
      xtree=xtree,
      func=fname,
      xfunc='xla_{}'.format(fname),
      code=code,
      sig=orig_sig,
      cppsig=sig,
      funsig=create_stdfunc_sig(xtree, orig_sig),
      mapsig=create_map_sig(xtree, orig_sig))


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
    fgens = []
    for ts in fndefs:
      fgens.append(get_xla_wrapper(ts))

    for fgen in fgens:
      print('{}\n'.format(fgen.code), file=ofile)
    print('\nstatic void RegisterFunctions() {', file=ofile)
    for fgen in fgens:
      print(
          '  register_extension_backend_op(\n    Backend::TPU,\n    "{}",\n    &{});'
          .format(fgen.mapsig, fgen.xfunc),
          file=ofile)
    print('}\n', file=ofile)


if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument('--output', type=str)
  args, files = arg_parser.parse_known_args()
  generate(args, files)
