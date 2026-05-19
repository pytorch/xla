#!/usr/bin/env python3
"""
This script uses GitHub API to create or update issues for lowering core ATen opset.
"""

from github import Github
from github import Auth
import sys
import urllib3

AUTH_KEY = 'YOUR_AUTH_KEY_HERE'
TEST_CORE_ATEN_OPS_LINK = 'https://raw.githubusercontent.com/pytorch/xla/master/test/test_core_aten_ops.py'
LOWER_CORE_ATEN_OPS_README_LINK = 'https://github.com/pytorch/xla/blob/master/FIX_LOWERING_FOR_CORE_ATEN_OPS.md'


def parse_test_file():
  issues_by_op = {}
  http = urllib3.PoolManager()
  resp = http.request('GET', TEST_CORE_ATEN_OPS_LINK)
  print(resp)
  lines = resp.data.decode('utf-8').split('\n')
  prev_line = ''
  for line in lines:
    if '@unittest.skip' in prev_line:
      if '@unittest.skip' in line:
        continue
      test_name = line.split()[1][:-7]
      op_name = test_name[:-2]
      op_number = test_name[-1:]
      # print(op_name, op_number)
      if op_name not in issues_by_op:
        issues_by_op[op_name] = []
      issues_by_op[op_name].append(op_name + "_" + op_number)
    prev_line = line
  return issues_by_op


def initialize_client():
  auth = Auth.Token(AUTH_KEY)
  return Github(auth=auth)


def create_issues(client, issues: dict[str, list[str]]):
  while True:
    user_answer = input(
        f'Running this script will create {len(issues)} issues in PyTorch/XLA. Continue? [Y/N]: '
    ).lower()
    if user_answer in ['y', 'yes']:
      break
    elif user_answer in ['n', 'no']:
      sys.exit()
    else:
      print('Unexpected input. Please enter [Y/N]')

  pytorch_xla_repo = client.get_repo('PyTorch/XLA')

  count = 0
  for op_name in issues:
    title = f'[Core ATen Opset] Lower {op_name[5:]}'
    op_unit_test_bullet = ''
    for op_test in issues[op_name]:
      op_unit_test_bullet += f'      - {op_test}\n'
    body = f'In order for PyTorch/XLA to support the PyTorch core ATen opset, it requires lowering each core ATen op in PyTorch/XLA. This issue is used to track the PyTorch/XLA lowering for **{op_name[5:]}**.\n\n' \
           f'Here are some general guidelines to lowering this op:\n' \
           f'  - Uncomment `@unittest.skip` or `@unittest.expectFailure` and run the unit test at [test_core_aten_ops.py]({TEST_CORE_ATEN_OPS_LINK}). Eg: `pytest test/test_core_aten_ops.py -k {op_name}_0`\n' \
           f'  - Make code changes until the test passes. Read and follow [fix_lowering_for_core_aten_ops.md]({LOWER_CORE_ATEN_OPS_README_LINK}) for ideas to fix.\n' \
           f'    - There may be multiple unit tests for a single op. For this op, the corresponding unit tests are:\n{op_unit_test_bullet}' \
           f'    - Please also uncomment the skips for all these tests and ensure all tests are fixed.\n' \
           f'    - Note that sometimes the fix may be to fix the unit tests itself. Please take a look at the corresponding unit tests to make sure the tests are valid.\n' \
           f'  - Submit the PR!\n\n' \
           f'For any questions, feel free to leave a comment in this PR.'

    labels = []
    labels.append(pytorch_xla_repo.get_label('good first issue'))
    labels.append(pytorch_xla_repo.get_label('core aten opset'))

    pytorch_xla_repo.create_issue(
        title=title,
        body=body,
        labels=labels,
    )

    count += 1


def check_for_new_issues(client, issues: dict[str, list[str]]):
  pytorch_xla_repo = client.get_repo('PyTorch/XLA')
  existing_issues = client.search_issues(query='label:"core aten opset"')
  print(existing_issues.totalCount)
  new_issues_to_create = {}

  for issue in issues:
    op_name = issue[5:]
    issue_exists = False
    for existing_issue in existing_issues:
      if op_name in existing_issue.title:
        issue_exists = True
        break
    if not issue_exists:
      new_issues_to_create[issue] = issues[issue]

  print(new_issues_to_create)
  return new_issues_to_create


if __name__ == '__main__':
  issues_to_create = parse_test_file()

  g_client = initialize_client()
  new_issues_to_create = check_for_new_issues(g_client, issues_to_create)
  create_issues(g_client, new_issues_to_create)
