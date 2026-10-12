#!/bin/sh
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
unset PYTHONHOME PYTHONPATH
if [ -n "${KD_PYTHON:-}" ]; then
    if ! "$KD_PYTHON" -I -X utf8 -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12) or sys.maxsize <= 2**32)' >/dev/null 2>&1; then
        echo 'KD_PYTHON must point to a working Python 3.12 executable.' >&2
        exit 1
    fi
    exec "$KD_PYTHON" -I -X utf8 "$task_root/scripts/launch_kd.py" "$@"
fi
if [ "${KD_USE_MANAGED_PYTHON:-0}" != '1' ]; then
    for candidate in "$task_root/.venv/bin/python" python3.12 python3 python; do
        if "$candidate" -I -X utf8 -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12) or sys.maxsize <= 2**32)' >/dev/null 2>&1; then
            exec "$candidate" -I -X utf8 "$task_root/scripts/launch_kd.py" "$@"
        fi
    done
fi
if [ "$(uname -s)" != 'Linux' ] || [ "$(uname -m)" != 'x86_64' ]; then
    echo 'Automatic runtime setup supports Linux x86-64; provide Python 3.12 with KD_PYTHON.' >&2
    exit 1
fi
tool_dir="$task_root/.local/uv-tools/0.13.0"
for directory in "$task_root/.local" "$task_root/.local/uv-tools" "$tool_dir" "$task_root/.local/uv-python" "$task_root/.local/uv-cache"; do
    if [ -L "$directory" ]; then echo 'Runtime directories must not be symbolic links.' >&2; exit 1; fi
    mkdir -p "$directory"
done
uv="$tool_dir/uv"
archive_hash='1468ebd5a5541121837c5a2817b9972ba6090fa6caa3d142620850a47fb75154'
binary_hash='3e801df892439f5cde65d8b51ea69b69abff199ef37cc3cae4481be15d592df0'
valid=0
if [ -x "$uv" ] && printf '%s  %s\n' "$binary_hash" "$uv" | sha256sum -c - >/dev/null 2>&1; then valid=1; fi
if [ "$valid" != '1' ]; then
    echo 'Knowledge distillation: Downloading the verified Python bootstrap tool (first setup).'
    url='https://github.com/astral-sh/uv/releases/download/0.13.0/uv-x86_64-unknown-linux-gnu.tar.gz'
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 2 --connect-timeout 20 --max-time 180 "$url" --output "$tool_dir/uv-download.tar.gz"
    elif command -v wget >/dev/null 2>&1; then
        wget --timeout=30 --tries=3 "$url" -O "$tool_dir/uv-download.tar.gz"
    else
        echo 'Install curl or wget for the first Python download.' >&2; exit 1
    fi
    printf '%s  %s\n' "$archive_hash" "$tool_dir/uv-download.tar.gz" | sha256sum -c -
    tar -xzf "$tool_dir/uv-download.tar.gz" -C "$tool_dir" --strip-components=1 uv-x86_64-unknown-linux-gnu/uv
    printf '%s  %s\n' "$binary_hash" "$uv" | sha256sum -c -
fi
unset UV_PYTHON UV_CONFIG_FILE UV_PROJECT UV_WORKING_DIR UV_NO_MANAGED_PYTHON UV_PYTHON_DOWNLOADS UV_OFFLINE UV_PYTHON_INSTALL_MIRROR UV_PYTHON_DOWNLOADS_JSON_URL
export UV_PYTHON_INSTALL_DIR="$task_root/.local/uv-python"
export UV_CACHE_DIR="$task_root/.local/uv-cache"
"$uv" --no-config python install --no-bin 3.12.15
runtime=$("$uv" --no-config python find --managed-python 3.12.15)
exec "$runtime" -I -X utf8 "$task_root/scripts/launch_kd.py" "$@"
