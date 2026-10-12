#!/bin/sh
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
exec sh "$task_root/launch-kd.sh" "$@"
