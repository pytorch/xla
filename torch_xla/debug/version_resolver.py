from __future__ import division
from __future__ import print_function

import json
import urllib.request
import torch_xla

_COMMIT_MAPPING = {
    'commit.author.name': 'author',
    'commit.author.email': 'author_email',
    'commit.committer.date': 'date',
    'commit.message': 'message',
}


def get_github_commit_api_url(owner, repo, tail=''):
  return 'https://api.github.com/repos/{}/{}/commits{}'.format(
      owner, repo, tail)


def get_commit_info(url_template, sha):
  url = url_template.format(sha)
  response = urllib.request.urlopen(url)
  return json.loads(response.read())


def nested_lookup(d, name):
  value = d
  for part in name.split('.'):
    value = value.get(part, None)
    if value is None:
      break
  return value


def build_commit_dict(data):
  result = dict()
  for k, v in _COMMIT_MAPPING.items():
    value = nested_lookup(data, k)
    if value:
      result[v] = value
  return result


def get_xla_version_info():
  revs = torch_xla._XLAC._get_git_revs()
  xla_data = get_commit_info(
      get_github_commit_api_url('pytorch', 'xla', tail='/{}'), revs['xla'])
  xla_info = build_commit_dict(xla_data)
  if revs['torch']:
    torch_data = get_commit_info(
        get_github_commit_api_url('pytorch', 'pytorch', tail='/{}'),
        revs['torch'])
    torch_info = build_commit_dict(torch_data)
  else:
    torch_info = None
  return xla_info, torch_info


def get_info_string(info, fullmsg=False):
  result = '  Author      : {}\n'.format(info['author'])
  result += '  Author Email: {}\n'.format(info['author_email'])
  result += '  Date        : {}\n'.format(info['date'])
  msg = info['message']
  if not fullmsg:
    msg = msg.split('\n', 1)[0]
  result += '  Message     : {}\n'.format(msg)
  return result


def get_xla_version_string(fullmsg=False):
  info = get_xla_version_info()
  result = 'XLA Commit:\n'
  result += get_info_string(info[0], fullmsg=fullmsg)
  if info[1]:
    result += 'Torch Commit:\n'
    result += get_info_string(info[1], fullmsg=fullmsg)
  return result


if __name__ == '__main__':
  print(get_xla_version_string())
