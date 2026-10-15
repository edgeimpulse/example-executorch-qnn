#!/usr/bin/env bash
# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Deploy this repository from the local machine to the specified EVK.
#
# Run this script from the repository root:
# ./sync.sh user@evk-host [remote-install-directory]
#
# Ignored files are skipped. Files absent locally are deleted from the EVK
# only after the dry-run preview is shown and you explicitly confirm. The
# deployment directory is created under the remote SSH user's home directory.
set -euo pipefail

EVK_HOST="${1:?Usage: $0 user@evk-host [remote-install-directory]}"
REMOTE_INSTALL_DIR="${2:-dev/demo-rock-paper-scissors-amazing-hand}"
LOCAL_SOURCE="$PWD/"

TEMP_DIR="$(mktemp -d)"
SSH_CONTROL_PATH="$TEMP_DIR/ssh-control"
RSYNC_PROTECT_FILTER="$TEMP_DIR/rsync-protect-filter"
SSH_COMMAND=(
  ssh
  -o ControlMaster=auto
  -o ControlPersist=60
  -o "ControlPath=$SSH_CONTROL_PATH"
)

cleanup() {
  "${SSH_COMMAND[@]}" -O exit "$EVK_HOST" >/dev/null 2>&1 || true
  rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

REMOTE_HOME="$("${SSH_COMMAND[@]}" "$EVK_HOST" 'printf %s "$HOME"')"
if [[ "$REMOTE_INSTALL_DIR" = /* ]]; then
  REMOTE_PROJECT_DIR="$REMOTE_INSTALL_DIR"
else
  REMOTE_PROJECT_DIR="$REMOTE_HOME/$REMOTE_INSTALL_DIR"
fi
REMOTE_DESTINATION="$EVK_HOST:$REMOTE_PROJECT_DIR/"

"${SSH_COMMAND[@]}" "$EVK_HOST" "mkdir -p -- '$REMOTE_PROJECT_DIR'"

while IFS= read -r pattern || [[ -n "$pattern" ]]; do
  if [[ -n "$pattern" && "$pattern" != \#* && "$pattern" != \!* ]]; then
    printf 'P %s\n' "$pattern" >> "$RSYNC_PROTECT_FILTER"
  fi
done < .gitignore

RSYNC_OPTIONS=(
  --archive
  --verbose
  --delete-after
  --exclude=.git/
  --filter=':- .gitignore'
  "--filter=merge $RSYNC_PROTECT_FILTER"
  "--rsh=${SSH_COMMAND[*]}"
)

echo "Dry run: previewing deployment to the device."
echo "No files will be copied or deleted during this preview."
echo "======================================================================"
rsync "${RSYNC_OPTIONS[@]}" --dry-run "$LOCAL_SOURCE" "$REMOTE_DESTINATION"

echo "======================================================================"
echo "Dry run complete. The changes above will now be applied."
read -r -p "This action updates existing files and will delete device files that are missing locally. Continue? [y/N] " response
if [[ "$response" != "y" && "$response" != "Y" ]]; then
  echo "Sync cancelled."
  exit 0
fi

echo "Applying deployment to the device."
rsync "${RSYNC_OPTIONS[@]}" "$LOCAL_SOURCE" "$REMOTE_DESTINATION"
