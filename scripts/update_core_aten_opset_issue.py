#!/usr/bin/env python3

"""

"""

from github import Github
from github import Auth
import urllib3

AUTH_KEY = "YOUR_AUTH_KEY_HERE"

def parse_test_file():
  issues_by_op = {}
  http = urllib3.PoolManager()
  resp = http.request('GET', 'https://raw.githubusercontent.com/pytorch/xla/5e63756c3438e0d25e32ba5dceac68d82d23993a/test/test_core_aten_ops.py')
  lines = resp.data.decode('utf-8').split('\n')
  prev_line = ''
  for line in lines:
    if '@unittest.skip' in prev_line:
      test_name = line.split()[1][:-7]
      op_name = test_name[:-2]
      op_number = test_name[-1:]
      # print(op_name, op_number)
      if op_name not in issues_by_op:
        issues_by_op[op_name] = []
      issues_by_op[op_name].append(op_name + "_" + op_number)
    prev_line = line
  return issues_by_op

def create_issues(issues: dict[str, list[str]]):
  auth = Auth.Token(AUTH_KEY)
  client = Github(auth=auth)
  pytorch_xla_repo = client.get_repo("PyTorch/XLA")

  count = 0
  for op_name in issues:
    if count == 1:
      break

    title = f'[Core ATen Opset] Lower {op_name[6:]}'
    op_unit_test_bullet = ''
    for op_test in issues[op_name]:
      op_unit_test_bullet += f'      - {op_test}\n'
    body = f'In order for PyTorch/XLA to support the PyTorch core ATen opset, it requires lowering each core ATen op in PyTorch/XLA. This issue is used to track the PyTorch/XLA lowering for **{op_name[6:]}**.\n\n' \
           f'Here are some general guidelines to lowering this op:\n' \
           f'  - Read through the [PyTorch/XLA op lowering guide](https://github.com/pytorch/xla/blob/master/OP_LOWERING_GUIDE.md)\n' \
           f'  - Lower the op\n' \
           f'  - To verify the correctness of the lowering, uncomment \'@unittest.skip\' and run the corresponding unit test at [test_core_aten_ops.py](https://github.com/pytorch/xla/blob/5e63756c3438e0d25e32ba5dceac68d82d23993a/test/test_core_aten_ops.py)\n' \
           f'    - There may be multiple unit tests for a single op. For this op, the corresponding unit tests are:\n{op_unit_test_bullet}' \
           f'    - Note that sometimes the fix may be to fix the unit tests itself. Please take a look at the corresponding unit tests to make sure the tests are valid.\n' \
           f'  - Submit the PR!\n\n' \
           f'For any questions, feel free to leave a comment in this PR.'

    labels = []
    labels.append(pytorch_xla_repo.get_label('good first issue'))
    labels.append(pytorch_xla_repo.get_label('core aten opset'))
    
    print(title)
    print(body)
    print(labels)

    pytorch_xla_repo.create_issue(
        title = title,
        body = body,
        labels = labels,
    )
    
    count += 1


if __name__ == '__main__':
  issues_to_create = parse_test_file()
  create_issues(issues_to_create)
