# Third-party components and licenses

The third-party components this project uses, and their licenses.

This project's own code is under The Clear BSD License ([LICENSE](LICENSE)).
Everything below is somebody else's and keeps its own terms.

**Nothing listed here is redistributed by this repository.** Every component is
fetched from its own distributor at build or install time. That is the line
between "we point at it" and "we ship it", and in this repository nothing is on
the "we ship it" side — see [NOTICE](NOTICE) for the full statement.

## Runtime and build dependencies

| Component | License | Shipped here? |
|---|---|---|
| ExecuTorch, PyTorch, torchvision | BSD-3-Clause | no — pip |
| `onnx2torch`, `onnxruntime` | MIT | no — pip |
| `numpy` | BSD | no — pip |
| `opencv-python` | Apache-2.0 | no — pip |
| Qualcomm QNN SDK / `libqnn1` | Qualcomm proprietary | no — Ubuntu IoT PPA, or fetched by ExecuTorch |

## Pretrained weights

| Artifact | License | Notes |
|---|---|---|
| torchvision ImageNet MobileNetV2 | BSD-3-Clause (torchvision) | Downloaded by torchvision at training time. The learning block fine-tunes from these weights; they are never redistributed here. |

## Worth reading before you build on this

### The Qualcomm SDK is proprietary, and it is baked into the Docker image

`blocks/learning-image-classifier/Dockerfile` triggers ExecuTorch's download of
the Qualcomm AI Runtime SDK during `docker build`, because custom blocks have
no network access while they run. That SDK is fetched under Qualcomm's own
license, by ExecuTorch, onto your machine.

This matters if you publish the built image: an image built from this
Dockerfile **contains** the Qualcomm SDK, even though this repository does not.
The repository is clean; a built image is not necessarily. Check Qualcomm's
terms before pushing one to a public registry.

### Models you train carry their data's provenance

The blocks produce `.pte` files from your Studio project's data. A trained
model is a derivative of its training data, so whatever consent and licensing
position applies to your images applies to the model you publish. This
repository ships no trained model and takes no position on yours.

### This repository ships no third-party files

Worth stating plainly, because it is unusual and it simplifies redistribution:
there are no fonts, no model weights, no vendored source, no compiled
libraries. The only binary files are four documentation screenshots and three
patch files, all authored by this project and itemized in
[docs/binary-provenance.md](docs/binary-provenance.md).

A separate repository, **demo-executorch-qnn**, holds applications built on
these artifacts. That one *does* redistribute third-party files (web fonts) and
uses GPL and source-available components at runtime. Its own notices cover
them; none of it applies here.
