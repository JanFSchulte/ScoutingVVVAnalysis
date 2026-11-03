#!/usr/bin/env bash
# Mirror the code, data and pixi environment to /work for Dask Gateway workers.
#
# /depot/cms/private is not world-readable, and the gateway needs an environment
# (and the editable vvv package with its data/) that everyone can read. Run the
# `vvv` commands that use --gateway from the mirror; outputs can still go anywhere.
#
# Usage: scripts/sync_work_mirror.sh [DEST]   (default /work/users/$USER/vvv/code)
set -euo pipefail
SRC="$(cd -P "$(dirname "$0")/.." && pwd -P)"
DEST="${1:-/work/users/${NB_USER:-$USER}/vvv/code}"
[ "$(realpath -m "$DEST")" != "$SRC" ] || { echo "DEST is the source tree" >&2; exit 1; }
mkdir -p "$DEST"
# No rsync in the session image: replace the tracked-code part of the mirror, keep its .pixi.
find "$DEST" -mindepth 1 -maxdepth 1 ! -name .pixi -exec rm -rf {} +
tar -C "$SRC" --exclude ./.pixi --exclude ./.git --exclude '*.parquet' --exclude '*.pkl' \
    --exclude ./metadata --exclude __pycache__ -cf - . | tar -C "$DEST" -xf -
# --manifest-path: the pixi wrapper refuses a cwd that resolves through the /home symlinks.
pixi install --manifest-path "$DEST/pixi.toml"
chmod -R a+rX "$DEST"
echo "mirror ready: $DEST"
