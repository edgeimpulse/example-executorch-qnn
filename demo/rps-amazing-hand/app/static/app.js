// Copyright (c) 2026 EdgeImpulse Inc.
// SPDX-License-Identifier: BSD-3-Clause-Clear

// Single /ws connection carries two kinds of message:
//   {kind: "match", ...}       -- full Match snapshot, drives scoreboard/history/phase
//   {kind: "commentary", ...}  -- one round's remark, streamed token-by-token
// The camera square is a server-composited MJPEG stream (/stream.mjpg) with
// detection boxes already drawn in. During resolve, result, and match-over the
// freeze-frame (/captured.jpg, box baked in) replaces the stream in the <img>.

const GESTURE_EMOJI = { rock: "\u270A", paper: "\u270B", scissors: "\u270C\uFE0F" };
const GESTURE_LABEL = { rock: "Rock", paper: "Paper", scissors: "Scissors" };
// Countdown words are pushed by the server (match.countdown_word) so the display
// shares its clock; the browser just renders them. "Shoot!" is shown for the
// whole capture phase.
const SHOOT_WORD = "Shoot!";
const CONFIG_STORAGE_KEY = "rps-match-config";

const el = {
  roundCurrent: document.getElementById("round-current"),
  roundSeparator: document.getElementById("round-separator"),
  roundTotal: document.getElementById("round-total"),
  scoreMachine: document.getElementById("score-machine"),
  scoreHuman: document.getElementById("score-human"),
  machineSquare: document.querySelector(".machine-square"),
  cameraSquare: document.querySelector(".camera-square"),
  machineGesture: document.getElementById("machine-gesture"),
  feed: document.getElementById("feed"),
  shootOverlay: document.getElementById("shoot-overlay"),
  shootText: document.getElementById("shoot-text"),
  historyBody: document.getElementById("history-body"),
  matchStatusText: document.getElementById("match-status-text"),
  commentaryText: document.getElementById("commentary-text"),
  startButton: document.getElementById("start-button"),
  resetButton: document.getElementById("reset-button"),
  strategySelect: document.getElementById("strategy-select"),
  roundsSelect: document.getElementById("rounds-select"),
};

const COMMENTARY_PLACEHOLDER = el.commentaryText.textContent.trim();

const STREAM_URL = "/stream.mjpg";
// Tracks which source the <img> currently shows so we don't reload the MJPEG
// stream on every match message (resetting src would restart the connection).
let feedMode = null; // "live" | "frozen"
// Defaults to "idle" (not null) to match the server's true default state
// before any match has ever been started -- main.py only sends an initial
// "match" message on connect if a match already exists, so on a fresh
// page load this stays "idle" (correctly) until a real message arrives.
let lastPhase = "idle";

// Commentary for the round currently being narrated. The server sends one
// "say" event per synthesized chunk with the chunk's playback duration; the
// browser types each chunk over that duration so the text tracks the voice.
// The final "match" message carries the same text in round.commentary, which
// reconciles this (and covers reconnects).
let commentaryRound = null;
let commentaryBuffer = "";
// Bumped on every new round/reset to invalidate any in-flight typing.
let commentaryEpoch = 0;
let sayQueue = [];
let sayRunning = false;

function restoreConfig() {
  const saved = sessionStorage.getItem(CONFIG_STORAGE_KEY);
  if (!saved) return;

  try {
    const config = JSON.parse(saved);
    if (["random", "always_win", "always_lose"].includes(config.strategy)) {
      el.strategySelect.value = config.strategy;
    }
    if (["1", "3", "7"].includes(String(config.rounds))) {
      el.roundsSelect.value = String(config.rounds);
    }
  } catch {
    sessionStorage.removeItem(CONFIG_STORAGE_KEY);
  }
}

function saveConfig() {
  sessionStorage.setItem(
    CONFIG_STORAGE_KEY,
    JSON.stringify({ strategy: el.strategySelect.value, rounds: el.roundsSelect.value }),
  );
}

// --- Feed source ---

function setFeedLive() {
  if (feedMode === "live") return;
  feedMode = "live";
  el.feed.onload = () => {
    el.feed.hidden = false;
  };
  el.feed.onerror = null;
  el.feed.src = STREAM_URL;
}

function setFeedFrozen() {
  feedMode = "frozen";
  el.feed.onload = () => {
    el.feed.onerror = null;
    el.feed.hidden = false;
  };
  el.feed.onerror = () => {
    el.feed.onload = null;
    el.feed.onerror = null;
    setFeedLive();
  };
  el.feed.src = `/captured.jpg?t=${Date.now()}`;
}

// --- History table ---

function winnerLabel(outcome) {
  if (outcome === "human") return { text: "You", cls: "winner-human" };
  if (outcome === "machine") return { text: "Amazing Hand", cls: "winner-machine" };
  return { text: "Draw", cls: "winner-draw" };
}

function renderHistory(rounds) {
  if (rounds.length === 0) {
    el.historyBody.innerHTML =
      '<tr class="placeholder-row"><td colspan="4">No rounds played yet.</td></tr>';
    return;
  }
  el.historyBody.innerHTML = rounds
    .map((r) => {
      const machine = `${GESTURE_EMOJI[r.machine_gesture]} ${GESTURE_LABEL[r.machine_gesture]}`;
      const human = r.human_gesture
        ? `${GESTURE_EMOJI[r.human_gesture]} ${GESTURE_LABEL[r.human_gesture]}`
        : "\u2014";
      const w = winnerLabel(r.outcome);
      return (
        `<tr><td>${r.index}</td><td>${machine}</td><td>${human}</td>` +
        `<td class="${w.cls}">${w.text}</td></tr>`
      );
    })
    .join("");
}

function matchStatus(match) {
  if (match.phase !== "match_over") return "Let's see what you've got!";
  if (match.machine_score > match.human_score) return "The Amazing Hand wins again!";
  if (match.human_score > match.machine_score) return "You defeated The Amazing Hand!";
  return "It's a draw!";
}

function setRoundWinnerFlash(outcome) {
  const winnerSquare = outcome === "machine"
    ? el.machineSquare
    : outcome === "human"
      ? el.cameraSquare
      : null;

  el.machineSquare.classList.toggle("round-winner", winnerSquare === el.machineSquare);
  el.cameraSquare.classList.toggle("round-winner", winnerSquare === el.cameraSquare);
}

// --- Match rendering ---

function renderMatch(match) {
  const matchNotStarted = match.phase === "idle" && match.current_round === 0;
  el.roundCurrent.textContent = matchNotStarted ? "-" : Math.max(1, match.current_round);
  el.roundSeparator.hidden = matchNotStarted;
  el.roundTotal.textContent = matchNotStarted ? "" : match.config.rounds;
  el.scoreMachine.textContent = match.machine_score;
  el.scoreHuman.textContent = match.human_score;
  el.matchStatusText.textContent = matchStatus(match);
  // Only mirror the server config while a match is running (controls are
  // disabled then). When idle/over, the dropdowns keep the browser's own
  // persisted choice so a refresh doesn't revert it to the last match.
  if (match.phase !== "idle" && match.phase !== "match_over") {
    el.strategySelect.value = match.config.strategy;
    el.roundsSelect.value = String(match.config.rounds);
  }

  renderHistory(match.rounds);

  const lastRound = match.rounds.length ? match.rounds[match.rounds.length - 1] : null;
  if (lastRound) {
    el.machineGesture.textContent = GESTURE_EMOJI[lastRound.machine_gesture];
    if (lastRound.commentary) {
      el.commentaryText.textContent = lastRound.commentary;
      el.commentaryText.classList.remove("placeholder");
    }
  } else if (matchNotStarted || commentaryRound === null) {
    el.machineGesture.textContent = "";
    commentaryEpoch += 1; // stop any lingering typing from a stopped match
    sayQueue = [];
    commentaryRound = null;
    commentaryBuffer = "";
    el.commentaryText.textContent = COMMENTARY_PLACEHOLDER;
    el.commentaryText.classList.add("placeholder");
  }

  if (match.phase === "idle" || match.phase === "opening" || match.phase === "countdown" || match.phase === "capture") {
    el.machineGesture.textContent = "";
  }

  const phaseChanged = match.phase !== lastPhase;

  const displayRoundWinner =
    match.phase === "round_result" || match.phase === "match_over";
  setRoundWinnerFlash(displayRoundWinner ? lastRound?.outcome : null);

  if (match.phase === "countdown") {
    el.shootOverlay.classList.remove("hidden");
    el.shootText.textContent = match.countdown_word || "Ready?";
    setFeedLive();
  } else if (match.phase === "capture") {
    // "Shoot!" holds over the live feed until the gesture resolves (RESOLVE).
    el.shootOverlay.classList.remove("hidden");
    if (phaseChanged) el.shootText.textContent = SHOOT_WORD;
    setFeedLive();
  } else {
    el.shootOverlay.classList.add("hidden");
  }

  if (match.phase === "capture" || match.phase === "idle" || match.phase === "opening") {
    setFeedLive();
  } else if (
    match.phase === "resolve" ||
    match.phase === "round_result" ||
    match.phase === "match_over"
  ) {
    if (phaseChanged && (match.phase === "resolve" || match.phase === "match_over")) {
      setFeedFrozen();
    }
  }

  const matchRunning = match.phase !== "idle" && match.phase !== "match_over";
  el.startButton.disabled = false;
  el.resetButton.disabled = matchRunning;
  el.strategySelect.disabled = matchRunning;
  el.roundsSelect.disabled = matchRunning;
  el.startButton.textContent = matchRunning ? "Stop Match" : "Start Match";

  lastPhase = match.phase;
}

// --- Commentary typewriter (paced to spoken audio) ---

function resetCommentary(roundIndex) {
  commentaryEpoch += 1; // invalidate any in-flight typing from a prior round
  commentaryRound = roundIndex;
  commentaryBuffer = "";
  sayQueue = [];
  el.commentaryText.textContent = "";
  el.commentaryText.classList.remove("placeholder");
}

function enqueueSay(epoch, text, durationMs) {
  sayQueue.push({ epoch, text, durationMs });
  if (!sayRunning) runSayQueue();
}

async function runSayQueue() {
  sayRunning = true;
  while (sayQueue.length) {
    const item = sayQueue.shift();
    if (item.epoch !== commentaryEpoch) continue; // dropped by a newer round
    await typeText(item.epoch, item.text, item.durationMs);
  }
  sayRunning = false;
}

function typeText(epoch, text, durationMs) {
  return new Promise((resolve) => {
    const chars = [...text];
    if (!chars.length) {
      resolve();
      return;
    }
    const perChar = Math.max(8, durationMs / chars.length);
    let i = 0;
    const step = () => {
      if (epoch !== commentaryEpoch) {
        resolve();
        return;
      }
      commentaryBuffer += chars[i];
      el.commentaryText.textContent = commentaryBuffer;
      i += 1;
      if (i < chars.length) setTimeout(step, perChar);
      else resolve();
    };
    step();
  });
}

// --- WebSocket ---

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws`);

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.kind === "match") {
      renderMatch(data);
    } else if (data.kind === "commentary") {
      if (data.operation === "start") {
        resetCommentary(data.round_index);
      } else if (data.operation === "say" && data.round_index === commentaryRound) {
        enqueueSay(commentaryEpoch, data.text, data.duration_ms);
      }
    }
  };

  ws.onclose = () => setTimeout(connect, 1000);
  ws.onerror = () => ws.close();
}

el.startButton.addEventListener("click", async () => {
  if (lastPhase !== "idle" && lastPhase !== "match_over") {
    const res = await fetch("/stop", { method: "POST" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      console.warn("stop failed:", res.status, err);
    }
    return;
  }

  const body = JSON.stringify({
    strategy: el.strategySelect.value,
    rounds: parseInt(el.roundsSelect.value, 10),
  });
  const res = await fetch("/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    console.warn("start failed:", res.status, err);
  }
});

el.resetButton.addEventListener("click", async () => {
  const res = await fetch("/reset", { method: "POST" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    console.warn("reset failed:", res.status, err);
  }
});

el.strategySelect.addEventListener("change", saveConfig);
el.roundsSelect.addEventListener("change", saveConfig);

restoreConfig();
setFeedLive();
connect();

// --- Artifact panel -------------------------------------------------------
// The demo's claim is "one training run, two artifacts, and you can see which
// one is answering". GET /artifact serves static facts (hashed and parsed once
// at startup) plus the live backend. The badge in the <summary> stays visible
// while the panel is collapsed, so stopping the rps-npu service mid-match is
// visible from across a booth.

const ARTIFACT_POLL_MS = 2000;

const artifactEl = {
  panel: document.getElementById("artifact-panel"),
  badge: document.getElementById("artifact-badge"),
  body: document.getElementById("artifact-body"),
};

function esc(value) {
  // Values come from our own server, but paths and op names are free-form, so
  // they go through the DOM's own escaping rather than into innerHTML raw.
  const div = document.createElement("div");
  div.textContent = value === null || value === undefined ? "" : String(value);
  return div.innerHTML;
}

function fmtBytes(n) {
  if (typeof n !== "number") return "—";
  if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${n} B`;
}

function renderArtifactBadge(live) {
  const el = artifactEl.badge;
  el.classList.remove("artifact-badge--npu", "artifact-badge--cpu");
  const backend = live && live.backend;
  if (backend === "npu") {
    const us = live.last_npu_us;
    const ms = typeof us === "number" && us > 0 ? ` ${(us / 1000).toFixed(2)} ms` : "";
    el.classList.add("artifact-badge--npu");
    el.textContent = `NPU${ms}`;
  } else if (backend === "cpu") {
    el.classList.add("artifact-badge--cpu");
    el.textContent = "CPU fallback";
  } else {
    el.textContent = backend ? String(backend) : "—";
  }
}

function artifactRows(pairs) {
  return pairs
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .map(([label, value, mono]) =>
      `<span class="artifact-label">${esc(label)}</span>` +
      `<span class="artifact-value${mono ? " artifact-mono" : ""}">${value}</span>`)
    .join("");
}

function renderOneArtifact(art, isLive) {
  if (!art) return "";
  const title = esc(art.label) + (isLive ? " &middot; in use" : "");
  if (!art.present) {
    return `<div class="artifact-art"><h3 class="artifact-art-title">${title}</h3>` +
      `<p class="artifact-note">Not on this host (${esc(art.path)}).</p></div>`;
  }

  const method = (art.methods || [])[0] || {};
  const delegates = (method.delegates || [])
    .map((d) => `${esc(d.backend)} &times;${d.count}`).join(", ");
  const fallback = (method.portable_ops || []);
  const report = art.report || {};
  const delegation = report.delegation || {};

  let delegated = null;
  if (typeof delegation.delegated_percent === "number") {
    delegated = `${delegation.delegated_percent}% ` +
      `<span class="artifact-note">(${delegation.delegated_nodes} of ` +
      `${delegation.delegated_nodes + delegation.non_delegated_nodes} nodes)</span>`;
  }

  const chipset = [report.soc_model, report.executorch_qnn_sdk, report.quant_scheme]
    .filter(Boolean).map(esc).join(" &middot; ");

  const rows = artifactRows([
    ["Size", esc(fmtBytes(art.size_bytes))],
    ["SHA-256", esc(art.sha256 ? art.sha256.slice(0, 16) : ""), true],
    ["Input", esc((method.inputs || []).join(", ")), true],
    ["Output", esc((method.outputs || []).join(", ")), true],
    ["Delegate", delegates],
    ["Delegated", delegated],
    ["On CPU", fallback.length
      ? fallback.map((o) => esc(o.op)).join(", ")
      : (delegates ? "nothing — the whole graph is in the delegate" : null)],
    ["Target", chipset],
  ]);

  const note = art.error ? `<p class="artifact-note">${esc(art.error)}</p>` : "";
  return `<div class="artifact-art">` +
    `<h3 class="artifact-art-title">${title}</h3>` +
    `<p class="artifact-art-file artifact-mono">${esc(art.path)}</p>` +
    `<div class="artifact-rows">${rows}</div>${note}</div>`;
}

function renderArtifactBody(data) {
  const live = data.live || {};
  const counts = [];
  if (typeof live.npu_frames === "number") counts.push(`${live.npu_frames} NPU`);
  if (typeof live.cpu_frames === "number") counts.push(`${live.cpu_frames} CPU`);

  const extras = [
    data.commentary && data.commentary.enabled ? "commentary on" : "commentary off",
    data.speech && data.speech.backend && data.speech.backend !== "none"
      ? `speech ${data.speech.backend}` : "speech off",
    data.hand && data.hand.enabled ? "hand on" : "hand off",
  ];

  artifactEl.body.innerHTML =
    `<p class="artifact-live">Frames so far: ${esc(counts.join(" / ") || "—")}` +
    (live.npu_enabled === false ? " &middot; NPU disabled by configuration" : "") +
    `</p>` +
    renderOneArtifact(data.npu, live.backend === "npu") +
    renderOneArtifact(data.cpu, live.backend === "cpu") +
    `<p class="artifact-host">${esc((data.host && data.host.machine) || "")}` +
    (data.host && data.host.executorch ? ` &middot; ExecuTorch ${esc(data.host.executorch)}` : "") +
    ` &middot; ${extras.map(esc).join(" &middot; ")}</p>`;
}

async function pollArtifact() {
  try {
    const res = await fetch("/artifact", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderArtifactBadge(data.live);
    // Only rebuild the body when someone is looking at it.
    if (artifactEl.panel.open) renderArtifactBody(data);
  } catch (err) {
    artifactEl.badge.textContent = "—";
    if (artifactEl.panel.open) {
      artifactEl.body.innerHTML =
        `<p class="artifact-note">Could not read /artifact (${esc(err.message)}).</p>`;
    }
  }
}

// Repaint immediately on open rather than waiting for the next tick.
artifactEl.panel.addEventListener("toggle", () => {
  if (artifactEl.panel.open) pollArtifact();
});

pollArtifact();
setInterval(pollArtifact, ARTIFACT_POLL_MS);
