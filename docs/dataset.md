# The gesture dataset

Provenance and consent record for the images behind `models/model.pte` and the
NPU model, and the rules for anyone collecting more.


## What the dataset is

Four classes, one gesture per image, photographed against a plain background:

| Class | Contents |
|---|---|
| `Rock`, `Paper`, `Scissors` | A single hand making the gesture |
| `unknown` | Frames with **no hand** — including faces and upper bodies |

**`unknown` is the class that makes the demo work, and it is the one people get
wrong when they reproduce this.** A classifier has no "is there a hand?"
output, so without a fourth class softmax is forced to pick rock, paper or
scissors for every empty frame. `unknown` absorbs those.

What goes in it matters as much as that it exists. Most non-gesture frames contain a **face, a torso, or someone walking past**, not an empty background. So `unknown` is built from exactly that — people in frame with no hand raised. An `unknown` class of blank
walls trains a model that rejects empty rooms and confidently misreads the
first face it sees.

That is also what puts identifiable people in the training set, which is why
the provenance below matters.

Class names are stored in `labels.txt` in alphabetical order, which is the
order Edge Impulse builds its one-hot targets in.

## Provenance

The images were collected for this project; it is not a public dataset. The
record of who is depicted and what they agreed to is held with the project
owners, not in this repository.

If you are reproducing this with your own data, that record is the thing to
create before you publish a model — see [Collecting more data](#collecting-more-data).

## What is published today

| Artifact | Published | Derived from the images? |
|---|---|---|
| The image dataset itself | No | — |
| `demo/rps-amazing-hand/models/model.pte` | **Yes**, committed to this repository | Yes |
| `model_qnn.pte` | On the Releases page | Yes |
| `X_split_test.npy` / `Y_split_test.npy` | No — used locally by `board/verify.sh` | Yes, these are the images as arrays |

## Collecting more data

`RPS_SAVE_FRAMES=<dir>` makes the running demo write the frames the model saw
to disk — raw material for a better `unknown` class. It is off by default and
should stay off unless the capture is deliberate.


## Known weaknesses

Recorded here because they shape what the next collection run should cover:

- Some scissors poses read as paper. More varied scissors data is the fix.
- `unknown` is under-trained; empty frames at odd angles still occasionally
  score as a gesture.
- The split is not stratified; best-epoch selection uses the test split rather
  than a held-out validation split.
