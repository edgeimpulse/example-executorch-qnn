# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Speaker contract + implementations: NullSpeaker, Piper, audio-analytics TTS.

Both real backends synthesize over local HTTP and play raw 16-bit mono PCM
through a player subprocess. Only `synthesize()` differs between them, so
PiperSpeaker subclasses MeloTtsSpeaker and inherits the disk cache, the canned
lead-in/countdown clips, the persistent player pipe and the playout timing.

Any transport, HTTP, or playback failure is logged and swallowed so a dead TTS
service or a missing audio player never interrupts a match.
"""

import asyncio
import hashlib
import io
import json
import random
import shutil
import wave
from contextlib import suppress
from pathlib import Path
from typing import Protocol

import httpx

import config


class Speaker(Protocol):
    async def warm_up(self) -> tuple[int, int] | None: ...

    async def synthesize(self, text: str) -> bytes | None: ...

    async def play(self, audio: bytes | None) -> None: ...

    async def close(self) -> None: ...

    def filler(self) -> tuple[str, bytes] | None: ...

    def opening_filler(self) -> tuple[str, bytes] | None: ...

    def countdown_clip(self, phrase: str) -> bytes | None: ...


class NullSpeaker:
    async def warm_up(self) -> tuple[int, int] | None:
        return None

    async def synthesize(self, text: str) -> bytes | None:
        return None

    async def play(self, audio: bytes | None) -> None:
        return

    async def close(self) -> None:
        return

    def filler(self) -> tuple[str, bytes] | None:
        return None

    def opening_filler(self) -> tuple[str, bytes] | None:
        return None

    def countdown_clip(self, phrase: str) -> bytes | None:
        return None


# --- Audio player resolution -------------------------------------------------
# Three tiers, most specific first: an explicit TTS_PLAYER_CMD, then a named
# entry from config.TTS_PLAYERS, then the first entry on PATH. See the
# "Speech: audio output" section of config.py for the rationale.


def _fill_one(arg: str, rate: int, device: str) -> str:
    if "{" not in arg:
        return arg
    try:
        return arg.format(rate=rate, device=device)
    except (KeyError, IndexError, ValueError):
        # A user-supplied TTS_PLAYER_CMD can contain an unbalanced brace or an
        # unknown placeholder. Pass it through verbatim rather than taking the
        # whole speaker down over one argument.
        print(f"speech: leaving player argument {arg!r} as-is (unrecognized placeholder).")
        return arg


def _fill(args: list[str], rate: int, device: str) -> list[str]:
    return [_fill_one(a, rate, device) for a in args]


def build_player_cmd(entry: dict, rate: int, device: str) -> list[str]:
    """Assemble one player entry into a command line.

    Order is args + device + stdin, and the stdin marker genuinely has to come
    last: `aplay -t raw ... - -D hw:0,0` reads stdin and then treats the device
    flag as a second filename.
    """
    cmd = _fill(entry["args"], rate, device)
    if device:
        if entry.get("device"):
            cmd += _fill(entry["device"], rate, device)
        else:
            print(
                f"speech: '{entry['name']}' has no device option -- ignoring "
                f"TTS_AUDIO_DEVICE='{device}' and using the system default."
            )
    cmd += _fill(entry.get("stdin") or [], rate, device)
    return cmd


def player_names() -> list[str]:
    return [entry["name"] for entry in config.TTS_PLAYERS]


def installed_players() -> list[str]:
    return [entry["name"] for entry in config.TTS_PLAYERS if shutil.which(entry["name"])]


def resolve_player(rate: int, *, quiet: bool = False) -> tuple[list[str], str] | None:
    """The player command to use for `rate`, or None if nothing is available.

    Returns (argv, display name). `quiet` suppresses the normal one-line
    announcement, for the re-resolve after a sample-rate correction.
    """
    device = config.TTS_AUDIO_DEVICE

    if config.TTS_PLAYER_CMD:
        cmd = _fill(config.TTS_PLAYER_CMD, rate, device)
        if shutil.which(cmd[0]) is None:
            # Honoured anyway -- it was asked for explicitly, and warm_up()
            # disables audio cleanly -- but say so here, where the mistake is.
            print(
                f"speech: TTS_PLAYER_CMD names '{cmd[0]}', which is not on PATH. "
                f"Installed players: {', '.join(installed_players()) or 'none'}."
            )
        elif not quiet:
            print(f"speech: audio player '{cmd[0]}' (from TTS_PLAYER_CMD)")
        return cmd, cmd[0]

    requested = config.TTS_PLAYER
    if requested:
        entry = next((e for e in config.TTS_PLAYERS if e["name"] == requested), None)
        if entry is None:
            print(
                f"speech: TTS_PLAYER='{requested}' is not a player this app knows. "
                f"Known players: {', '.join(player_names())}. "
                "Set TTS_PLAYER_CMD to use something else. Auto-detecting instead."
            )
        elif shutil.which(entry["name"]) is None:
            print(
                f"speech: TTS_PLAYER='{requested}' is not installed on this board. "
                "Auto-detecting instead."
            )
        else:
            if not quiet:
                print(f"speech: audio player '{entry['name']}' (from TTS_PLAYER)")
            return build_player_cmd(entry, rate, device), entry["name"]

    for entry in config.TTS_PLAYERS:
        if shutil.which(entry["name"]):
            if not quiet:
                print(f"speech: audio player '{entry['name']}' (auto-detected)")
            return build_player_cmd(entry, rate, device), entry["name"]

    print(
        "speech: audio disabled -- none of these audio players are installed: "
        f"{', '.join(player_names())}."
    )
    print(
        "speech: install one (e.g. 'sudo apt install alsa-utils' for aplay), or set "
        "TTS_PLAYER_CMD to a player of your own. Run ./scripts/check-audio.sh to diagnose."
    )
    return None


class MeloTtsSpeaker:
    """Synthesizes a line of commentary on the audio-analytics TTS container and
    plays it on the device's audio sink.
    """

    # Shown in the startup line; overridden by subclasses.
    _label = "audio-analytics TTS"

    def __init__(
        self,
        base_url: str,
        *,
        model: str = config.TTS_MODEL,
        voice: str = config.TTS_VOICE,
        language: str = config.TTS_LANGUAGE,
        sample_rate: int = config.TTS_SAMPLE_RATE,
        speaking_rate: str = config.TTS_SPEAKING_RATE,
        timeout: float = config.TTS_TIMEOUT,
        player_cmd: list[str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._voice = voice
        self._language = language
        self._sample_rate = sample_rate
        self._speaking_rate = speaking_rate
        self._timeout = timeout
        # Resolve the player now so a missing one is reported at startup rather
        # than mid-match. An explicit argument (tests, callers) wins over the
        # three-tier resolution.
        if player_cmd is not None:
            self._player_cmd: list[str] | None = _fill(
                player_cmd, sample_rate, config.TTS_AUDIO_DEVICE
            )
            self._player_name = self._player_cmd[0]
        else:
            resolved = resolve_player(sample_rate)
            self._player_cmd = resolved[0] if resolved else None
            self._player_name = resolved[1] if resolved else ""
        self._cache_dir = Path(config.TTS_CACHE_DIR)
        # Disabled after warm-up detects an unreachable service or missing player.
        self._enabled = True
        # One long-lived player process fed raw PCM over a pipe, so cues don't
        # pay a per-clip subprocess spawn (and the sink/Bluetooth link stays warm
        # between clips). Kept low so a stopped match can flush it by killing it.
        self._player_process: asyncio.subprocess.Process | None = None
        self._player_lock = asyncio.Lock()
        # Pre-synthesized (text, pcm) lead-ins, populated during warm-up.
        self._fillers: list[tuple[str, bytes]] = []
        self._opening_fillers: list[tuple[str, bytes]] = []
        # Pre-synthesized countdown words, keyed by phrase.
        self._countdown: dict[str, bytes] = {}

    async def warm_up(self) -> tuple[int, int] | None:
        # Prime the model load and validate the pipeline (endpoint, voice, and
        # audio player) so a broken setup degrades to silence at startup rather
        # than mid-match. The primed audio is synthesized but never played.
        print(f"speech: using {self._label} at {self._base_url}")
        if self._player_cmd is None:
            # resolve_player() already explained why and how to fix it.
            self._enabled = False
            return None
        if shutil.which(self._player_cmd[0]) is None:
            print(f"speech: audio disabled -- player '{self._player_cmd[0]}' not found on PATH.")
            self._enabled = False
            return None
        print("speech: warming the TTS model ...")
        # This first call also settles the real sample rate (see
        # _adopt_sample_rate), which is why it runs before anything is cached.
        primed = await self.synthesize("Ready.")
        if primed is None:
            print("speech: audio disabled -- TTS service did not return audio.")
            self._enabled = False
            return None
        # Cache canned lead-ins in the active voice so they can play instantly.
        self._fillers, new_fillers = await self._cache_clips(config.TTS_FILLERS)
        self._opening_fillers, new_opening = await self._cache_clips(config.TTS_OPENING_FILLERS)
        new_countdown = 0
        for phrase in [config.TTS_COUNTDOWN_READY, *config.TTS_COUNTDOWN_WORDS, config.TTS_SHOOT_WORD]:
            pcm, is_new = await self._synthesize_cached(phrase)
            if pcm:
                self._countdown[phrase] = pcm
                if is_new:
                    new_countdown += 1
        cached = len(self._fillers) + len(self._opening_fillers) + len(self._countdown)
        new_total = new_fillers + new_opening + new_countdown
        self._prune_cache(
            [*config.TTS_FILLERS, *config.TTS_OPENING_FILLERS,
             config.TTS_COUNTDOWN_READY, *config.TTS_COUNTDOWN_WORDS, config.TTS_SHOOT_WORD]
        )
        # Open the persistent player and wake the sink so the first real cue
        # doesn't pay the stream/Bluetooth start-up latency.
        await self._prime_player()
        return cached, new_total

    def _adopt_sample_rate(self, rate: int) -> None:
        """Switch to the rate the backend actually produced.

        The player is started with a rate baked into its arguments and play()
        derives each clip's duration from it, so a mismatch means both wrong
        pitch and wrong timing. Rather than trusting a configured constant, the
        first clip's own header is taken as the truth.

        In practice this runs during warm-up's priming call, before any player
        exists and before anything is cached, so killing the player here is a
        no-op and cache keys are computed at the corrected rate.
        """
        if rate == self._sample_rate:
            return
        print(
            f"speech: voice is {rate} Hz, not the configured {self._sample_rate} Hz "
            "-- using the voice's rate."
        )
        self._sample_rate = rate
        resolved = resolve_player(rate, quiet=True)
        if resolved is not None:
            self._player_cmd, self._player_name = resolved
        # Respawns at the new rate on the next play().
        self._kill_player()

    def _pcm_from_wav(self, data: bytes) -> bytes | None:
        """Strip a WAV container down to the raw PCM the player expects."""
        try:
            with wave.open(io.BytesIO(data), "rb") as wav:
                channels = wav.getnchannels()
                width = wav.getsampwidth()
                rate = wav.getframerate()
                frames = wav.readframes(wav.getnframes())
        except (wave.Error, EOFError) as exc:
            print(f"speech: TTS response was not a readable WAV file ({exc})")
            return None
        if width != 2 or channels != 1:
            print(
                f"speech: unexpected audio format ({channels} channel(s), {width * 8}-bit) "
                "-- the player is configured for 16-bit mono."
            )
            return None
        self._adopt_sample_rate(rate)
        return frames

    async def _cache_clips(self, phrases: list[str]) -> tuple[list[tuple[str, bytes]], int]:
        clips: list[tuple[str, bytes]] = []
        new_count = 0
        for phrase in phrases:
            pcm, is_new = await self._synthesize_cached(phrase)
            if pcm:
                clips.append((phrase, pcm))
                if is_new:
                    new_count += 1
        return clips, new_count

    def _cache_path(self, text: str) -> Path:
        # Key on everything that changes the resulting audio, so switching
        # model/voice/rate/speaking-rate or editing a phrase invalidates only
        # the affected clip rather than requiring a manual cache clear.
        key = "|".join(
            [self._model, self._voice, str(self._sample_rate), self._speaking_rate, text]
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return self._cache_dir / f"{digest}.pcm"

    async def _synthesize_cached(self, text: str) -> tuple[bytes | None, bool]:
        path = self._cache_path(text)
        try:
            return path.read_bytes(), False
        except OSError:
            pass
        pcm = await self.synthesize(text)
        if pcm:
            try:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                path.write_bytes(pcm)
            except OSError as exc:
                print(f"speech: could not write TTS cache file ({exc})")
        return pcm, True

    def _prune_cache(self, active_phrases: list[str]) -> None:
        # Delete cache files left over from a prior model/voice/rate/phrase
        # combination, so repeated config changes don't grow the cache forever.
        if not self._cache_dir.is_dir():
            return
        active_names = {self._cache_path(phrase).name for phrase in active_phrases}
        removed = 0
        for path in self._cache_dir.glob("*.pcm"):
            if path.name not in active_names:
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    pass
        if removed:
            print(f"speech: pruned {removed} stale TTS cache clip(s).")

    def filler(self) -> tuple[str, bytes] | None:
        return random.choice(self._fillers) if self._fillers else None

    def opening_filler(self) -> tuple[str, bytes] | None:
        return random.choice(self._opening_fillers) if self._opening_fillers else None

    def countdown_clip(self, phrase: str) -> bytes | None:
        return self._countdown.get(phrase)

    async def synthesize(self, text: str) -> bytes | None:
        if not self._enabled:
            return None
        text = text.strip()
        if not text:
            return None
        payload = {
            "text": text,
            "model": self._model,
            "language": self._language,
            "voice": self._voice,
            "sample_rate": self._sample_rate,
            "parameters": {"speaking_rate": self._speaking_rate},
        }
        url = f"{self._base_url}/audio-analytics/v1/api/tts/synthesize"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    print(f"speech: TTS server returned {response.status_code}")
                    return None
                return response.content
        except httpx.HTTPError as exc:
            print(f"speech: TTS server unreachable ({exc})")
            return None

    async def play(self, audio: bytes | None) -> None:
        if not audio or not self._enabled:
            return
        async with self._player_lock:
            proc = await self._ensure_player_locked()
        if proc is None or proc.stdin is None:
            return
        try:
            # write() buffers the whole clip; the player drains it in real time,
            # so we pace with the clip's playout length rather than awaiting drain
            # (which would block for the audio's full duration on long clips).
            proc.stdin.write(audio)
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            print(f"speech: audio player write failed ({exc})")
            self._kill_player()
            return
        seconds = len(audio) / (self._sample_rate * 2)
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            # Flush queued audio so a stopped match goes quiet promptly.
            self._kill_player()
            raise

    async def _ensure_player_locked(self) -> asyncio.subprocess.Process | None:
        proc = self._player_process
        if proc is not None and proc.returncode is None:
            return proc
        if self._player_cmd is None:
            return None
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._player_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as exc:
            print(f"speech: cannot start audio player ({exc})")
            self._player_process = None
            return None
        self._player_process = proc
        return proc

    def _kill_player(self) -> None:
        proc = self._player_process
        self._player_process = None
        if proc is not None and proc.returncode is None:
            proc.kill()

    async def _prime_player(self) -> None:
        if not self._enabled:
            return
        async with self._player_lock:
            proc = await self._ensure_player_locked()
        if proc is None or proc.stdin is None:
            return
        # ~50ms of silence to establish the stream and wake the sink.
        silence = b"\x00" * (int(self._sample_rate * 0.05) * 2)
        try:
            proc.stdin.write(silence)
        except (BrokenPipeError, ConnectionResetError, OSError):
            self._kill_player()

    async def close(self) -> None:
        async with self._player_lock:
            proc = self._player_process
            self._player_process = None
            if proc is not None and proc.returncode is None:
                proc.kill()
                with suppress(ProcessLookupError):
                    await proc.wait()


def voice_sample_rate(voices_dir: str, voice: str) -> int | None:
    """Sample rate from a Piper voice's sidecar JSON, or None if unreadable.

    Every Piper voice ships <name>.onnx alongside <name>.onnx.json, which
    carries audio.sample_rate. Reading it means the player starts at the right
    rate on the very first clip instead of being corrected afterwards.
    """
    path = Path(voices_dir) / f"{voice}.onnx.json"
    try:
        with path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    rate = meta.get("audio", {}).get("sample_rate")
    return int(rate) if isinstance(rate, (int, float)) and rate > 0 else None


class PiperSpeaker(MeloTtsSpeaker):
    """Synthesizes on a local Piper HTTP server (started by start.sh).

    Differs from the base class in exactly two places: where it posts, and the
    fact that Piper answers with a WAV file rather than raw PCM. Everything
    else -- caching, lead-in clips, the player pipe, playout timing -- is
    inherited unchanged.

    Piper runs as its own process rather than being imported. That keeps the
    model loaded across clips (a per-call CLI invocation would re-load it every
    time), and it keeps this GPL-licensed engine at arm's length from the demo,
    the same arrangement the MeloTTS container had.
    """

    _label = "Piper TTS"

    def __init__(
        self,
        base_url: str,
        *,
        voice: str = config.PIPER_VOICE,
        voices_dir: str = config.PIPER_VOICES_DIR,
        length_scale: float = config.PIPER_LENGTH_SCALE,
        sample_rate: int | None = None,
        timeout: float = config.TTS_TIMEOUT,
        player_cmd: list[str] | None = None,
    ) -> None:
        rate = sample_rate or voice_sample_rate(voices_dir, voice)
        if rate is None:
            rate = config.PIPER_FALLBACK_SAMPLE_RATE
            print(
                f"speech: could not read {voice}.onnx.json -- assuming {rate} Hz for now; "
                "the first clip's own header will correct it."
            )
        # model/voice/speaking_rate feed the inherited cache key, so recording
        # the voice and length scale there makes switching either invalidate
        # only the affected clips.
        super().__init__(
            base_url,
            model="piper",
            voice=voice,
            language="en",
            sample_rate=rate,
            speaking_rate=str(length_scale),
            timeout=timeout,
            player_cmd=player_cmd,
        )
        self._voices_dir = voices_dir
        self._length_scale = length_scale

    async def synthesize(self, text: str) -> bytes | None:
        if not self._enabled:
            return None
        text = text.strip()
        if not text:
            return None
        payload = {
            "text": text,
            "voice": self._voice,
            "length_scale": self._length_scale,
        }
        url = f"{self._base_url}/synthesize"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            print(f"speech: Piper server unreachable at {self._base_url} ({exc})")
            return None
        if response.status_code != 200:
            print(f"speech: Piper server returned {response.status_code}")
            return None
        return self._pcm_from_wav(response.content)
