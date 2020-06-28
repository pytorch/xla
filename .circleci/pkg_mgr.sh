#!/usr/bin/env sh

function get_pkg_mgr() {
  export SUDO='sudo '
  if PKG_MGR_EXEC="$(command -v apt-get 2>/dev/null)"; then
    export PKG_MGR='apt-get'
  elif PKG_MGR_EXEC="$(command -v dnf 2>/dev/null)"; then
    export PKG_MGR='dnf'
  elif PKG_MGR_EXEC="$(command -v zypper 2>/dev/null)"; then
    export PKG_MGR='zypper'
  elif PKG_MGR_EXEC="$(command -v pacman 2>/dev/null)"; then
    export PKG_MGR='pacman'
  elif PKG_MGR_EXEC="$(command -v port 2>/dev/null)"; then
    export PKG_MGR='port'
    export SUDO=''
  elif PKG_MGR_EXEC="$(command -v brew 2>/dev/null)"; then
    export PKG_MGR='brew'
    export SUDO=''
  fi
  export PKG_MGR_EXEC="$PKG_MGR_EXEC"
}

get_pkg_mgr
