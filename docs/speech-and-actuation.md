# Speech and actuation

The two parts of the demo that are neither the model nor the game: the voice
that comments on each round, and the robot hand that throws the machine's
gesture. Both are optional, both run locally, and both are the kind of thing
that silently does nothing when it is misconfigured — which is why most of this
page is about telling the failure modes apart.

- [Where the compute goes](#where-the-compute-goes)
- [Architecture](#architecture)
- [Speech: choosing a backend](#speech-choosing-a-backend)
- [Speech: the licence arrangement](#speech-the-licence-arrangement)
- [Speech: audio output on an unknown board](#speech-audio-output-on-an-unknown-board)
- [Speech: sample rate](#speech-sample-rate)
- [The Amazing Hand](#the-amazing-hand)
- [Diagnosing it](#diagnosing-it)


Reference material lives with the demo itself:
[demo/rps-amazing-hand/README.md](../demo/rps-amazing-hand/README.md) has every
setting and its default. This page is the reasoning.

---

## Where the compute goes

By the time speech is added, three subsystems want hardware on one board. The
split is deliberate:

| Subsystem | Runs on | Why there |
|---|---|---|
| Gesture inference (`model_qnn.pte`) | Hexagon NPU, via the `rps-npu` service | 0.23–0.26 ms/inference, and it must keep up with a 30 fps camera ([results](results.md)) |
| LLM commentary (Llama 3.2 1B Q4_0) | Hexagon NPU, via llama.cpp's Hexagon backend | Not for speed — token generation is actually *faster* on the CPU. It is there to keep the CPU free ([decision 15](design-decisions.md#15-the-llm-on-the-npu-through-llamacpps-hexagon-backend-q4_0)) |
| Text to speech (Piper) | **CPU** | The NPU is already carrying the two above. Piper is comfortably faster than real time on these cores, and the clips that matter most are pre-synthesized anyway |

That last row is the whole argument for the default speech backend. Speech is
the one piece here with no reason to be on the NPU, and a real reason not to be.

Vision and the LLM do share `HTP0`, which is why `app/game.py` pauses the
detector around generation and synthesis and resumes it while the queued audio
plays out — audio playback touches neither. **The size of that contention has
not been measured.** It is the main open item for this part of the demo; see
[What is verified](#what-is-verified-and-what-is-not).

---

## Architecture

Both subsystems are Python `Protocol`s with a null implementation, so the demo
degrades one feature at a time instead of failing:

```python
class Actuator(Protocol):              # app/actuation.py
    async def prepare(self) -> None: ...
    async def play(self, gesture: Gesture) -> None: ...
    async def wiggle(self) -> None: ...
    async def home(self) -> None: ...
    async def release(self) -> None: ...

class Speaker(Protocol):               # app/speech.py
    async def warm_up(self) -> tuple[int, int] | None: ...
    async def synthesize(self, text: str) -> bytes | None: ...
    async def play(self, audio: bytes | None) -> None: ...
    async def close(self) -> None: ...
    def filler(self) -> tuple[str, bytes] | None: ...
    def opening_filler(self) -> tuple[str, bytes] | None: ...
    def countdown_clip(self, phrase: str) -> bytes | None: ...
```

| Protocol | Implementations | Selected by |
|---|---|---|
| `Actuator` | `AmazingHandActuator`, `NullActuator` | `ENABLE_AMAZING_HAND` + the serial port being accessible |
| `Speaker` | `PiperSpeaker`, `MeloTtsSpeaker`, `NullSpeaker` | `TTS_BACKEND` |

The factories are `_make_actuator()` and `_make_speaker()` at the top of
`app/main.py`. Anything that satisfies the protocol can be dropped in — LEDs, a
different hand, a cloud TTS — without touching the game loop.

Two properties are load-bearing and easy to break:

- **All serial I/O is offloaded.** Every `AmazingHandActuator` method wraps its
  blocking `rustypot` calls in `asyncio.to_thread`. The same event loop times
  the countdown audio, so a blocking servo write would drag the voice out of
  sync with the on-screen animation.
- **`PiperSpeaker` subclasses `MeloTtsSpeaker`** and overrides exactly
  `__init__` and `synthesize()`. The disk cache, cache keying, pruning, canned
  lead-ins, countdown clips, the persistent player pipe and the playout timing
  are all inherited. Only the transport differs, so only the transport is
  reimplemented.

---

## Speech: choosing a backend

`TTS_BACKEND` picks the engine. When it is unset, `ENABLE_AUDIO_ANALYTICS=1`
implies `melo`, so a setup predating this setting keeps the backend it had.

| | **Piper** (default) | **MeloTTS** |
|---|---|---|
| How a third party gets it | `pip install piper-tts[http]` — prebuilt `manylinux_2_17_aarch64` abi3 wheel, no compiler, no apt packages | A Docker image that is not published and not in this repository |
| Voice | Two files fetched by name, ~60 MB for a medium voice | Baked into the container |
| Runs on | CPU (onnxruntime) | The container's own stack |
| Voice quality | Clear and intelligible; prosody is flatter — recognizably a 2022-era VITS | Noticeably more natural |
| Sample rate | 22.05 kHz (medium) / 16 kHz (low), read from the audio | 44.1 kHz |
| Licence | GPL-3.0-or-later | Unknown to this project |

**Piper is the default because it is the only one anyone can reproduce.** 

Perceived latency is handled by the architecture rather than the engine. A
canned lead-in (`TTS_FILLERS`, pre-synthesized at startup and cached on disk)
speaks the instant a round resolves, covering LLM generation and synthesis;
the real line overwrites it when ready. The countdown words and "Shoot!" are
cached too, so the clips that must land on a beat never wait on synthesis at
all.

---

## Speech: sample rate

The player's sample rate is baked into its arguments at spawn time, and
`Speaker.play()` derives each clip's duration from it:

```python
seconds = len(audio) / (self._sample_rate * 2)
```

That duration paces the typewriter reveal and, with `AUDIO_ONSET_SECONDS`,
decides when on-screen text lands relative to the voice. So a wrong sample rate
is not just wrong pitch — it is wrong timing throughout the match. 

It is therefore resolved from the audio itself, in two steps:

1. At construction, from the voice's sidecar `<name>.onnx.json`
   (`audio.sample_rate`). This is why a voice is **two files that must stay
   together** — the `.onnx` alone is not enough, and `start.sh` refuses to
   start speech if the JSON is missing.
2. On the first synthesized clip, from the WAV header. If it disagrees with
   step 1, the real rate wins: the speaker adopts it, re-resolves the player
   command, and drops the old player so it respawns correctly.

Step 2 runs during warm-up's priming call, before any player exists and before
any clip is cached, so the correction costs nothing and cache keys are computed
at the correct rate. Step 1 exists so the common case never needs step 2.

The cache key is `engine | voice | rate | speaking-rate | text`, so switching
voice or backend invalidates only the affected clips rather than requiring a
manual cache clear, and stale files are pruned after each successful warm-up.

---

## The Amazing Hand

Eight SCS0009 serial servos, two per finger, driven through
[`rustypot`](https://pypi.org/project/rustypot/)'s `Scs0009PyController` at
1 Mbaud. A pose is a pair of angles per finger, offset by that motor's neutral
calibration:

```python
def _rock(self) -> None:
    self._index(-90, 90, self._close_speed)
    self._middle(-90, 90, self._close_speed)
    self._ring(-90, 90, self._close_speed)
    self._thumb(-90, 90, self._close_speed + 1)   # the thumb needs a touch more
```

`AMAZING_HAND_MIDDLE_CALIBRATION` is eight per-motor neutral offsets in
degrees, in motor order 1–8: index, index, middle, middle, ring, ring, thumb,
thumb. The committed values are for the physical hand this demo was built
with. `AMAZING_HAND_SIDE` records left (2) or right (1) orientation; the motion
code does not use it yet, and right-hand support would need mirrored poses.

Calibration is the one thing here that is genuinely per-unit: servo horns are
installed by hand, so offsets differ between assemblies. `scripts/test-hand.py`
takes `--cal 15,-13,0,-12,0,3,10,-10` so a value can be tried before being
written into `app/config.py`.

If the serial port is missing or inaccessible, `start.sh` says so with the fix
and runs the game with `NullActuator` — the on-screen hand still plays.

---

## Diagnosing it

Both subsystems fail in ways that look identical from outside, so both have a
standalone check that runs without the web UI in the way. Both import the
app's own modules rather than reimplementing anything, so what they test cannot
drift from what `start.sh` does.

### Audio: `scripts/check-audio.sh`

"No sound" has three independent causes. The script tests them apart and names
the one that failed:

| Stage | Tests | Typical failure |
|---|---|---|
| 1 | A player is resolved and its binary is on `PATH` | Nothing installed; or a `TTS_PLAYER_CMD` typo |
| 2 | The player accepts audio and exits cleanly (a generated 440 Hz tone, no Piper needed) | No sound server running; wrong device name |
| 3 | Piper and its voice are present, and a synthesized line plays | Package or voice files missing |

```bash
./scripts/check-audio.sh                                   # all three stages
./scripts/check-audio.sh --list                            # what is installed, what
                                                           # would be chosen, and the
                                                           # valid device names
./scripts/check-audio.sh --player aplay --device hw:0,0     # try a combination
./scripts/check-audio.sh --no-synth                        # tone only
```

**Stage 2 has an honest limit.** It passes whenever the player accepts the
audio and exits cleanly — including when it is writing to an output nobody can
hear. There is no way to verify from software that sound left the speaker. A
stage that passes while the room stays quiet is the signal to run `--list` and
try a different `--device`.

### The hand: `scripts/test-hand.py`

Four candidate causes when the hand misbehaves — the port, group membership,
the calibration, the game logic — and the web UI cannot separate them. This
walks the first three and fails with the specific remedy, then drives the real
`AmazingHandActuator` through every pose:

```bash
.venv/bin/python scripts/test-hand.py                 # all poses, with pauses
.venv/bin/python scripts/test-hand.py --list          # settings only, no movement
.venv/bin/python scripts/test-hand.py --pose paper    # one pose and stop
.venv/bin/python scripts/test-hand.py --port /dev/ttyUSB0
```

It reads the owning group off the device rather than assuming `dialout`, and
prints the exact `usermod` line including the username. Like stage 2 above, a
clean exit only proves the servos accepted the commands — not that the poses
look right — so it ends by saying what to look at and which calibration entry
to change.

---

