# Binary and document provenance

Every non-source file this repository ships has an origin, a license, and an
analysis note. This exists so a license or supply-chain audit has a written
answer for each binary instead of an unexplained blob.

Scope: files that are not plain source text. All project-owned source carries
a `Copyright (c) 2026 EdgeImpulse Inc.` line and the SPDX identifier
`BSD-3-Clause-Clear` in its header; it is not repeated here.

The authoritative statement of what is shipped versus what is fetched at
install time is [NOTICE](../NOTICE).

## Summary

| Category | Files | Bytes | Owner |
|---|---|---|---|
| Documentation images | 4 | 1,481,531 | This project |
| Porting patches | 3 | 68,070 | This project |

**Every shipped binary in this repository is owned by this project.** There are
no third-party files of any kind: no fonts, no model weights, no archives, no
`.so`, `.a`, `.dll`, `.jar`, `.whl`, `.gguf`, `.onnx`, `.tflite` or `.eim`
files, and no vendored third-party source tree.

## Documentation images

| Path | Bytes | SHA-256 (first 16) |
|---|---|---|
| `blocks/learning-image-classifier/assets/create-impulse.png` | 563,486 | `ec4340ad12961570` |
| `blocks/learning-image-classifier/assets/deployment.png` | 384,796 | `9d9641b24a38f816` |
| `blocks/learning-image-classifier/assets/model-testing.png` | 198,662 | `a3227c74c878f03c` |
| `blocks/learning-image-classifier/assets/training.png` | 334,587 | `e4b41dc12053cbee` |

**Category:** screenshots for documentation.

**Origin:** captured from Edge Impulse Studio for this repository, showing the
steps of building and training an impulse with this learning block.

**License:** `BSD-3-Clause-Clear`, as works of this project.

**Analysis:** the original captures included the signed-in account's avatar.
Those regions were redacted before publication; the record of what was redacted
is in
[`blocks/learning-image-classifier/assets/README.md`](../blocks/learning-image-classifier/assets/README.md).
No personal data, credentials, project identifiers or API keys remain visible.

## Porting patches

| Path | Bytes | SHA-256 (first 16) |
|---|---|---|
| `porting/patches/executorch-pytorch-kws-block.patch` | 22,474 | `dfebb6d2a432d7b2` |
| `porting/patches/executorch-pytorch-object-detection-fomo-block.patch` | 23,096 | `51b809bd57fdd916` |
| `porting/patches/executorch-pytorch-timeseries-block.patch` | 22,500 | `e678fbcfb9356aee` |

**Category:** unified diffs, text.

**Origin:** authored by this project. Each adds the Qualcomm NPU export path to
one of Edge Impulse's public `executorch-pytorch-*-block` repositories.

**License:** `BSD-3-Clause-Clear`, as works of this project.

**Analysis:** these contain only this project's own additions, expressed as a
diff. They carry context lines from the upstream blocks, as any diff does, but
no substantive third-party code. They deliberately carry **no** SPDX header,
because a header inserted into a patch file changes the diff's line offsets and
stops it applying.

## Not in this repository

For completeness, since a reader of the related **demo-executorch-qnn**
repository will see these there:

- **Web fonts** (SIL OFL 1.1) — used by the demo's web UI, not by anything here.
- **A trained `model.pte`** — this repository produces models but ships none.
  The example model is on the Releases page, not in the source tree.
- **Android test fixtures** — belong to the demo's Android app.

Each is documented in that repository's own provenance file.
