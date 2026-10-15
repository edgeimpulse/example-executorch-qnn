#!/usr/bin/env bash
# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Download the GGUF used for round commentary into models/.
#
#   ./scripts/fetch-llm-model.sh
#
# Built with Llama. The default model is Meta's Llama 3.2 1B Instruct, under
# the Llama 3.2 Community License (https://www.llama.com/llama3_2/license/) --
# source-available, not an OSI open-source licence. It is downloaded here, not
# redistributed by this repository. Products built with it should carry a
# "Built with Llama" notice.
#
# Q4_0 is deliberate, not incidental: every Hexagon example upstream uses Q4_0,
# and the HTP kernels are built around that block format. K-quants (Q4_K_M and
# friends) are better quality per bit but are not the tested path on the NPU --
# most publishers only ship K-quants, which is why the repo below is named
# explicitly rather than left to a search.
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_REPO="${MODEL_REPO:-unsloth/Llama-3.2-1B-Instruct-GGUF}"
MODEL_FILE="${MODEL_FILE:-Llama-3.2-1B-Instruct-Q4_0.gguf}"
DEST_DIR="${DEST_DIR:-models}"
MIN_BYTES="${MIN_BYTES:-600000000}"   # ~773 MB expected; anything tiny is an error page

URL="https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}"
DEST="${DEST_DIR}/${MODEL_FILE}"

mkdir -p "$DEST_DIR"

if [[ -f "$DEST" ]]; then
  size=$(wc -c < "$DEST")
  if (( size >= MIN_BYTES )); then
    echo "Already present: $DEST ($(du -h "$DEST" | cut -f1))"
    exit 0
  fi
  echo "Removing truncated download ($size bytes): $DEST" >&2
  rm -f "$DEST"
fi

echo "Downloading ${MODEL_FILE}"
echo "  from ${MODEL_REPO}"
echo "  to   ${DEST}"
echo ""

# -L follows redirects (Hugging Face redirects to a CDN); -C - resumes a partial
# file so a dropped connection does not mean starting over.
curl -L -C - --fail --progress-bar -o "$DEST" "$URL" || {
  echo "" >&2
  echo "Download failed. If the repository is gated you will need to accept its" >&2
  echo "licence on huggingface.co and pass a token:" >&2
  echo "  curl -L -H \"Authorization: Bearer \$HF_TOKEN\" -o $DEST $URL" >&2
  exit 1
}

size=$(wc -c < "$DEST")
if (( size < MIN_BYTES )); then
  echo "" >&2
  echo "Downloaded only $size bytes -- that is an HTML error page, not a model." >&2
  echo "This normally means the repository is gated. First lines:" >&2
  head -c 300 "$DEST" >&2
  rm -f "$DEST"
  exit 1
fi

echo ""
echo "Done: $DEST ($(du -h "$DEST" | cut -f1))"
echo ""
echo "Verify it runs on the NPU before enabling commentary:"
echo "  ./scripts/check-llama-hexagon.sh $DEST"