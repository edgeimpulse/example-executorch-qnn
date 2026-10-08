# Third-party components and licences

The third-party components this project uses, and their licences.

This project's own code is under The Clear BSD License ([LICENSE](LICENSE)).
Everything below is somebody else's and keeps its own terms.

Nothing listed is redistributed by this repository except where the "Shipped
here?" column says so — the rest is fetched by scripts at install time. If you
redistribute this project or build on it, check that column first: it is the
line between "we point at it" and "we ship it".

## Runtime and build dependencies

| Component | Licence | Shipped here? |
|---|---|---|
| ExecuTorch, PyTorch, torchvision | BSD-3-Clause | no — pip |
| llama.cpp (commentary engine) | MIT | no — built by `scripts/build-llama-hexagon.sh` |
| Piper (TTS engine, `piper-tts`) | **GPL-3.0-or-later** | no — pip, see below |
| Amazing Hand — software | Apache-2.0 | no |
| Amazing Hand — mechanical design | CC BY 4.0 | no — attribution required if you publish the CAD or photos of it |
| `rustypot` (servo driver) | Apache-2.0 | no — pip |
| FastAPI, httpx, onnxruntime | MIT | no — pip |
| uvicorn, numpy | BSD | no — pip |
| `opencv-python` | Apache-2.0 | no — pip |
| Atkinson Hyperlegible, Barlow (web fonts) | SIL OFL 1.1 | **yes** — `demo/rps-amazing-hand/app/static/fonts/`, with the per-family licence text alongside, as the OFL requires |
| Qualcomm QNN SDK / `libqnn1` | Qualcomm proprietary | no — Ubuntu IoT PPA |

## Models and data

| Artifact | Licence | Notes |
|---|---|---|
| Llama 3.2 1B Instruct (GGUF) | **Llama 3.2 Community License** | Not an OSI open-source licence. Fetched by `scripts/fetch-llm-model.sh`; never redistributed here. See below. |
| Piper voice (`en_US-kristin-medium`) | public domain (LibriVox) | Fetched by `scripts/fetch-tts-voice.sh`. **The voice catalogue is mixed — see below.** |
| `models/model.pte` (gesture classifier) | — | Trained in this project. Fine-tuned from torchvision's ImageNet MobileNetV2 weights; see the open question below. |

## Worth reading before you build on this

### 1. Piper is GPL-3.0-or-later

It links espeak-ng for phonemization, which is why the licence is copyleft.

This repository keeps it at arm's length rather than linking it:

- Piper runs as **its own process**, started by `start.sh`, behind its HTTP API.
- `app/speech.py` only ever POSTs to `127.0.0.1:5000` and parses the WAV that
  comes back. **No file in this repository imports Piper.**
- Its dependency lives in a separate
  [`requirements-speech.txt`](demo/rps-amazing-hand/requirements-speech.txt),
  not in `requirements.txt`, and `SKIP_SPEECH=1 ./install.sh` omits it.
- The same arrangement is used for llama.cpp, which is MIT and needs no such
  care — the structure was chosen for both reasons.

Process isolation is also the faster option here — a per-clip CLI invocation
would reload the voice model every time — so nothing was traded away to get
this separation.

If you redistribute this project, satisfy yourself that the arrangement meets
your own requirements. `SKIP_SPEECH=1 ./install.sh` omits Piper entirely; the
demo then runs silently and nothing else changes.

### 2. Llama 3.2 

llama.cpp (the engine) is MIT. The **weights** are under Meta's Llama 3.2
Community License, which is a source-available community licence.

This repository never redistributes the weights; `scripts/fetch-llm-model.sh`
downloads them from Hugging Face at install time.


## Training data

The gesture model is trained on images collected for this project. Their
provenance, the consent position, and the rules for collecting more are in
[docs/dataset.md](docs/dataset.md).

`RPS_SAVE_FRAMES` makes the running demo write camera frames to disk. It is off
by default. If you turn it on in front of other people, read that page first.
