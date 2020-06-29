#!/usr/bin/env bash

set -euo pipefail

declare -r REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"

source "$REPO_ROOT"'/.circleci/pkg_mgr.sh'

function pkg_names() {
    case "$PKG_MGR" in
        brew) PKGS='git sccache gnu-sed python@3 cmake' ;;
        apt-get) PKGS='git build-essential python3-dev python3 python3-venv cmake' ;;
    esac
}

pkg_names
eval "${SUDO}"' '"${PKG_MGR_EXEC}"' install '"${PKGS}"
eval "${SUDO}"' '"${PKG_MGR_EXEC}"' '"${PKG_MGR_INSTALL_CMD}"' '"${PKGS}"
