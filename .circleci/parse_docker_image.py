import yaml
from urllib.request import urlopen

base_config = 'pytorch_linux_build'
name = 'pytorch_xla_linux_bionic_py3_6_clang9_build'
url = 'https://raw.githubusercontent.com/pytorch/pytorch/master/.circleci/config.yml'

def parse_config_yml():
    with urlopen(url) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
        for job in configs['workflows']['build']['jobs']:
            if base_config in job:
                if job[base_config]['name'] == name:
                    print(job[base_config]['docker_image'])
                    return

if __name__ == '__main__':
    parse_config_yml()
