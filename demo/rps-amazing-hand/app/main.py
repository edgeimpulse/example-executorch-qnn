"""FastAPI app: camera feed, live detection overlay during every live-feed phase,
the game itself, and the freeze-frame capture.

Routes:
  GET  /stream.mjpg   continuous MJPEG stream: the live camera square with
                      detection boxes composited server-side onto each frame
  GET  /captured.jpg  the exact frame that decided the current/last round,
                      with its winning detection box baked in
  WS   /ws            two kinds of message, distinguished by "kind":
                         {"kind": "match", ...Match fields}
                         {"kind": "commentary", ...token stream}
  POST /start         begin a match; body: {"strategy": "...", "rounds": N}
  POST /stop          end an active match and keep its completed results
  POST /reset         clear a finished match while preserving its configuration

Boxes are drawn into the stream while the browser shows the live feed; the
freeze-frame (/captured.jpg) owns the visual during resolve and result display.
"""

import asyncio
import functools
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

import artifact
import config
import overlay
from actuation import AmazingHandActuator, NullActuator
from commentary import LlamaCppCommentator, NullCommentator
from data_model import Detection, MatchConfig, Phase, Strategy
from game import GameEngine
from perception import CameraSource
from pte_perception import PteGestureDetector
from speech import MeloTtsSpeaker, NullSpeaker, PiperSpeaker

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Confidence gate for the idle live-box overlay when no match is active.
IDLE_MIN_CONFIDENCE = config.MIN_CONFIDENCE

camera = CameraSource()
# Swap for EIGestureDetector(camera) (and re-add the perception import) to run
# the Edge Impulse .eim path instead -- useful for an A/B on the same hardware.
detector = PteGestureDetector(camera)


def _make_actuator():
    # start.sh exports the enable flag and serial port; config.py reads them.
    if config.ENABLE_AMAZING_HAND:
        return AmazingHandActuator(
            serial_port=config.AMAZING_HAND_SERIAL_PORT,
            baudrate=config.AMAZING_HAND_BAUDRATE,
            timeout=config.AMAZING_HAND_TIMEOUT,
            calibration=config.AMAZING_HAND_MIDDLE_CALIBRATION,
            max_speed=config.AMAZING_HAND_MAX_SPEED,
            close_speed=config.AMAZING_HAND_CLOSE_SPEED,
            side=config.AMAZING_HAND_SIDE,
        )
    return NullActuator()


def _make_commentator():
    # start.sh exports these; config.py reads them so the app stays env-free.
    if config.ENABLE_LLM_COMMENTARY:
        return LlamaCppCommentator(config.LLAMA_SERVER_URL)
    return NullCommentator()


def _make_speaker():
    # TTS_BACKEND picks the engine; both talk local HTTP and are started by
    # start.sh. An unreachable server degrades to silence inside warm_up(),
    # so a wrong choice here costs the voice, not the match.
    backend = config.TTS_BACKEND
    if backend == "piper":
        return PiperSpeaker(config.PIPER_SERVER_URL)
    if backend == "melo":
        if config.ENABLE_AUDIO_ANALYTICS:
            return MeloTtsSpeaker(config.AUDIO_ANALYTICS_SERVER_URL)
        print("speech: TTS_BACKEND=melo needs ENABLE_AUDIO_ANALYTICS=1 -- running silent.")
        return NullSpeaker()
    if backend not in ("none", "off"):
        print(f"speech: unknown TTS_BACKEND='{backend}' -- expected piper, melo or none.")
    return NullSpeaker()


actuator = _make_actuator()
commentator = _make_commentator()
speaker = _make_speaker()


@dataclass
class AppState:
    """Mutable application runtime state."""
    engine: GameEngine | None = None
    match_task: asyncio.Task | None = None
    match_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Newest composited JPEG (live square + boxes) that MJPEG clients stream.
    stream_jpeg: bytes | None = None


state = AppState()
active_websockets: set[WebSocket] = set()


@asynccontextmanager
async def _startup_spinner() -> None:
    if not sys.stdout.isatty():
        yield
        return

    frames = "|/-\\"
    index = 0

    async def spin() -> None:
        nonlocal index
        while True:
            print(f"\r{frames[index % len(frames)]}", end="", flush=True)
            index += 1
            await asyncio.sleep(0.15)

    task = asyncio.create_task(spin())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        print("\r\033[K", end="", flush=True)


class _QuietArtifactPolling(logging.Filter):
    """Drop uvicorn access lines for the artifact panel's background poll.

    The panel polls /artifact every 2 s so its NPU/CPU badge stays live. That
    is machinery, not a user action, and over a booth session it is thousands
    of lines burying the ones worth reading -- NPU ACTIVE / CPU FALLBACK,
    speech failures, tracebacks. Set RPS_LOG_ARTIFACT=1 to see them again when
    debugging the route itself.

    uvicorn.access passes its fields as record.args:
    (client_addr, method, full_path, http_version, status_code).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3:
            return not str(args[2]).startswith("/artifact")
        return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Added here rather than at import time so it lands after uvicorn has
    # configured its own logging.
    if os.environ.get("RPS_LOG_ARTIFACT", "").lower() not in ("1", "true", "yes"):
        logging.getLogger("uvicorn.access").addFilter(_QuietArtifactPolling())

    camera.start()
    await detector.start()  # runs continuously regardless of match state
    await actuator.home()  # rest the hand in its default rock pose
    async with _startup_spinner():
        await commentator.warm_up()
    if config.ENABLE_LLM_COMMENTARY:
        print("commentary: warm-up request finished.")
    async with _startup_spinner():
        result = await speaker.warm_up()
    if result is not None:
        total, new = result
        loaded = total - new
        if loaded:
            print(f"speech: loaded {loaded} filler clip(s) from disk cache.")
        if new:
            print(f"speech: synthesized {new} filler clip(s).")
        print("speech: warm-up request finished.")

    # Hash and parse both .pte files once, so GET /artifact is cheap and any
    # parsing problem is visible in the startup log rather than in the UI.
    await asyncio.to_thread(artifact.warm)

    compositor_task = asyncio.create_task(_composite_stream())
    yield
    compositor_task.cancel()

    if state.match_task is not None:
        state.match_task.cancel()
    # Await cancellation so an in-flight match runs its finally block
    # (actuator.release()) before we tear down the detector and camera.
    for task in (compositor_task, state.match_task):
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass
    await speaker.close()  # stop the persistent audio player
    await detector.stop()
    camera.stop()


app = FastAPI(lifespan=lifespan)


# --- Video ---

def _overlay_detections() -> list[Detection]:
    """Boxes to composite onto the live stream now, or [] during frozen phases."""
    engine = state.engine
    show_boxes = engine is None or engine.match.phase in (
        Phase.IDLE,
        Phase.OPENING,
        Phase.COUNTDOWN,
        Phase.CAPTURE,
    )
    if not show_boxes:
        return []
    result = detector.latest_result
    if result is None:
        return []
    min_conf = engine.match.config.min_confidence if engine is not None else IDLE_MIN_CONFIDENCE
    return [d for d in result.detections if d.confidence >= min_conf]


async def _composite_stream() -> None:
    """Single shared loop: overlays the latest boxes onto the newest camera
    square and publishes one JPEG for every MJPEG client to stream."""
    interval = 1.0 / config.STREAM_FPS
    loop = asyncio.get_running_loop()
    while True:
        display = camera.latest_display
        if display is not None and camera.square_size:
            scale = display.shape[0] / camera.square_size
            render = functools.partial(
                overlay.render_jpeg, display, _overlay_detections(), scale=scale
            )
            jpeg = await loop.run_in_executor(None, render)
            if jpeg is not None:
                state.stream_jpeg = jpeg
        await asyncio.sleep(interval)


@app.get("/stream.mjpg")
async def stream() -> StreamingResponse:
    async def frames():
        interval = 1.0 / config.STREAM_FPS
        while True:
            jpeg = state.stream_jpeg
            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
                )
            await asyncio.sleep(interval)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, max-age=0"},
    )


@app.get("/captured.jpg")
async def captured() -> Response:
    if state.engine is None or state.engine.last_capture_jpeg is None:
        return Response(status_code=404)
    return Response(content=state.engine.last_capture_jpeg, media_type="image/jpeg")


# --- Broadcasting (two independent senders sharing the same websocket set) ---

async def _send_all(payload: dict) -> None:
    if not active_websockets:
        return
    dead = set()
    # Iterate a snapshot: a socket can connect or disconnect (mutating the set)
    # while we await a send below, which would otherwise error mid-iteration.
    for ws in tuple(active_websockets):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.add(ws)
    active_websockets.difference_update(dead)


async def _broadcast(match) -> None:
    """Passed into GameEngine as its BroadcastFn. Fires on every phase change."""
    await _send_all({"kind": "match", **asdict(match)})


async def _broadcast_commentary(event: dict) -> None:
    """Passed into GameEngine: streams one round's commentary token-by-token."""
    await _send_all({"kind": "commentary", **event})


def _snapshot(detection: Detection | None = None) -> bytes | None:
    # Freeze the frame the deciding inference ran on and bake its winning box in
    # so /captured.jpg needs no client-side overlay; fall back to the live square.
    square = detector.result_square()
    scale = 1.0
    if square is None:
        square = camera.latest_display
        if square is not None and camera.square_size:
            scale = square.shape[0] / camera.square_size
    if square is None:
        return None
    detections = [detection] if detection is not None else []
    return overlay.render_jpeg(square, detections, scale=scale)


@app.post("/start")
async def start_match(request: Request) -> JSONResponse:
    body = await request.json() if await request.body() else {}

    try:
        strategy = Strategy(body.get("strategy", Strategy.RANDOM.value))
    except ValueError:
        return JSONResponse({"error": "invalid strategy"}, status_code=400)

    try:
        rounds = int(body.get("rounds", MatchConfig().rounds))
        if rounds not in (1, 3, 7):
            raise ValueError
    except (TypeError, ValueError):
        return JSONResponse({"error": "invalid rounds"}, status_code=400)

    async with state.match_lock:
        if state.match_task is not None and not state.match_task.done():
            return JSONResponse({"error": "a match is already running"}, status_code=409)

        config = MatchConfig(rounds=rounds, strategy=strategy)
        state.engine = GameEngine(
            detector,
            actuator,
            commentator,
            _broadcast,
            config,
            snapshot=_snapshot,
            commentary=_broadcast_commentary,
            speaker=speaker,
        )
        state.match_task = asyncio.create_task(state.engine.run_match())
    return JSONResponse({"status": "started"})


@app.post("/reset")
async def reset_match() -> JSONResponse:
    async with state.match_lock:
        if state.match_task is not None and not state.match_task.done():
            return JSONResponse({"error": "a match is already running"}, status_code=409)
        if state.engine is not None:
            await state.engine.reset()
    return JSONResponse({"status": "reset"})


@app.post("/stop")
async def stop_match() -> JSONResponse:
    async with state.match_lock:
        task = state.match_task
        if task is None or task.done():
            return JSONResponse({"error": "no match is running"}, status_code=409)
        task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        pass
    return JSONResponse({"status": "stopped"})


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    active_websockets.add(websocket)
    if state.engine is not None:
        await websocket.send_json({"kind": "match", **asdict(state.engine.match)})
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        active_websockets.discard(websocket)


@app.get("/config")
async def client_config() -> JSONResponse:
    # Client-relevant values that must match app/config.py; served so the
    # browser never hardcodes a copy that could drift.
    return JSONResponse({"countdownReadySeconds": config.COUNTDOWN_READY_SECONDS})


@app.get("/artifact")
async def artifact_info() -> JSONResponse:
    # Static facts about both .pte files (cached at startup) plus which one is
    # answering right now. This is the demo's thesis made visible: stop the
    # rps-npu service and `live.backend` flips to "cpu" while the game carries
    # on. Supersedes the planned /backend badge route -- same data, more of it.
    return JSONResponse(
        {
            **artifact.facts(),
            "live": detector.status(),
            "speech": {
                "backend": config.TTS_BACKEND,
                "voice": config.PIPER_VOICE if config.TTS_BACKEND == "piper" else config.TTS_VOICE,
            },
            "hand": {"enabled": config.ENABLE_AMAZING_HAND},
            "commentary": {"enabled": config.ENABLE_LLM_COMMENTARY},
        },
        headers={"Cache-Control": "no-store"},
    )


# Safari (and others) can otherwise cache a stale app.js/index.html/style.css
# across deploys, silently running old client code against the current server.
_NO_CACHE_PATHS = {"/", "/index.html", "/app.js", "/style.css", "/config"}


@app.middleware("http")
async def _no_cache_static(request: Request, call_next):
    response = await call_next(request)
    if request.url.path in _NO_CACHE_PATHS:
        response.headers["Cache-Control"] = "no-store"
    return response


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
