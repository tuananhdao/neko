#!/bin/sh
set -eu

if test -z "${PIXI_PROJECT_ROOT:-}"; then
  echo "Neko development environment is not active."
  echo "Enter it with: pixi shell"
  exit 1
fi

project_root=$PIXI_PROJECT_ROOT
expected_prefix=$project_root/install
status=0

check_value() {
  variable_name=$1
  actual_value=$2
  expected_value=$3

  if test "$actual_value" != "$expected_value"; then
    echo "$variable_name mismatch:"
    echo "  actual:   $actual_value"
    echo "  expected: $expected_value"
    status=1
  fi
}

check_command() {
  command_name=$1
  expected_path=$2
  actual_path=$(command -v "$command_name" 2>/dev/null || true)

  if test "$actual_path" != "$expected_path"; then
    echo "$command_name resolves incorrectly:"
    echo "  actual:   ${actual_path:-not found}"
    echo "  expected: $expected_path"
    status=1
  fi
}

check_value PREFIX "${PREFIX:-}" "$expected_prefix"
check_value NEKO_EXEC "${NEKO_EXEC:-}" "$expected_prefix/bin/neko"
check_value MAKENEKO_EXEC "${MAKENEKO_EXEC:-}" \
  "$expected_prefix/bin/makeneko"
check_value GENMESHBOX_EXEC "${GENMESHBOX_EXEC:-}" \
  "$expected_prefix/bin/genmeshbox"

check_command neko "$expected_prefix/bin/neko"
check_command makeneko "$expected_prefix/bin/makeneko"
check_command genmeshbox "$expected_prefix/bin/genmeshbox"

if ! command -v mpirun >/dev/null 2>&1; then
  echo "mpirun was not found in the Pixi environment."
  status=1
fi

build_library=$project_root/src/.libs/libneko.a
installed_library=$expected_prefix/lib/libneko.a
build_neko=$project_root/src/neko
installed_neko=$expected_prefix/bin/neko

if test ! -f "$installed_library" || test ! -x "$installed_neko"; then
  echo "The repository-local Neko installation is incomplete."
  echo "Build it with: pixi run build-neko-cpu dp"
  status=1
else
  if test -f "$build_library" && test "$build_library" -nt "$installed_library"; then
    echo "The build-tree libneko is newer than the installed libneko."
    echo "Synchronize it with: make dev"
    status=1
  fi
  if test -x "$build_neko" && test "$build_neko" -nt "$installed_neko"; then
    echo "The build-tree neko is newer than the installed neko."
    echo "Synchronize it with: make dev"
    status=1
  fi
fi

if test "$status" -ne 0; then
  exit "$status"
fi

echo "Neko development environment is consistent."
echo "  prefix:   $expected_prefix"
echo "  makeneko: $(command -v makeneko)"
echo "  neko:     $(command -v neko)"
echo "  mpirun:   $(command -v mpirun)"
