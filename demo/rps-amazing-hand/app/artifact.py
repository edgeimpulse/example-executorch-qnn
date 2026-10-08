"""Facts about the ExecuTorch artifacts behind this demo, served at GET /artifact.

The project's claim is "one training run, two artifacts": a portable CPU `.pte`
that runs anywhere ExecuTorch runs, and a QNN `.pte` holding a context binary
for one chipset and SDK range. On screen that claim is invisible -- a visitor
sees a game that could be running anything, on anything. This module reads the
facts back out of the files themselves, so the UI can show them and so "which
artifact is answering right now" is checkable rather than asserted.

Only fields that `tools/inspect_pte.py` already exercises on this project's
real artifacts are read here, so the parsing is not new ground.

Everything is best-effort. A missing file, an unusable executorch install, or
an unreadable `.pte` yields fewer facts, never an exception: the panel is a
nice-to-have and the game is not.
"""

import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import config

# Same table as tools/inspect_pte.py -- ExecuTorch's ScalarType enum.
_DTYPES = {0: "uint8", 1: "int8", 2: "int16", 3: "int32", 4: "int64", 5: "float16",
           6: "float32", 7: "float64", 11: "bool"}

# facts() is pure for the life of the process (the files do not change under
# us), and hashing plus flatbuffer parsing is not free, so it is computed once.
_cache: dict | None = None


def _tensor_desc(value) -> str:
    t = getattr(value, "val", None)
    sizes = getattr(t, "sizes", None)
    if sizes is None:
        return type(t).__name__
    dtype = _DTYPES.get(int(getattr(t, "scalar_type", -1)), str(getattr(t, "scalar_type", "?")))
    return f"{dtype}{list(sizes)}"


def _program_facts(data: bytes) -> dict:
    """Backends, fallback ops, tensors and segments, read out of the .pte.

    Returns {"error": "..."} rather than raising. Note the bare `except
    Exception`: a broken executorch install does not necessarily raise
    ImportError -- a CUDA-flavoured torch wheel that cannot find its libraries
    raises ValueError from torch's own import, and that must not take the
    route down with it.
    """
    try:
        from executorch.exir._serialize._program import deserialize_pte_binary
    except Exception as exc:
        return {"error": f"executorch is not usable here ({type(exc).__name__})"}

    try:
        program = deserialize_pte_binary(data).program
    except Exception as exc:
        return {"error": f"could not parse the program ({type(exc).__name__}: {exc})"}

    facts: dict = {"methods": []}
    try:
        for plan in program.execution_plan:
            backends = Counter(d.id for d in plan.delegates)
            ops = Counter(
                op.name + (f".{op.overload}" if op.overload else "")
                for op in plan.operators
            )
            facts["methods"].append({
                "name": plan.name,
                "inputs": [_tensor_desc(plan.values[i]) for i in plan.inputs],
                "outputs": [_tensor_desc(plan.values[i]) for i in plan.outputs],
                "delegates": [{"backend": b, "count": n} for b, n in backends.items()],
                # Whatever the delegate did not take runs as a portable CPU
                # kernel. On a fully delegated model this list is empty.
                "portable_ops": [{"op": name, "count": n} for name, n in sorted(ops.items())],
            })
        sizes = [s.size for s in program.segments]
        if sizes:
            # A delegate payload -- a QNN context binary, say -- lives here.
            facts["segments"] = {"count": len(sizes), "largest_bytes": max(sizes)}
    except Exception as exc:
        facts["error"] = f"parsed, but reading it failed ({type(exc).__name__}: {exc})"
    return facts


def _report_facts(path: Path) -> dict:
    """The authoritative numbers from export_report.json, when it is present.

    Delegation percentage comes from here rather than being derived from the
    `.pte`: the report records `get_delegation_info()` from export time, which
    counts graph nodes. Nothing in the file itself gives that, so when the
    report is absent the honest answer is the fallback-op list and no
    percentage.
    """
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    if not isinstance(report, dict):
        return {}

    out: dict = {"path": str(path)}
    for key in ("status", "backend", "soc_model", "quant_scheme",
                "executorch_qnn_sdk", "host", "input_shape", "reason"):
        if report.get(key) is not None:
            out[key] = report[key]

    delegation = report.get("delegation")
    if isinstance(delegation, dict):
        out["delegation"] = dict(delegation)
        delegated = delegation.get("delegated_nodes")
        non = delegation.get("non_delegated_nodes")
        if isinstance(delegated, int) and isinstance(non, int) and (delegated + non) > 0:
            out["delegation"]["delegated_percent"] = round(
                100.0 * delegated / (delegated + non), 1
            )

    accuracy = report.get("accuracy")
    if isinstance(accuracy, dict):
        out["accuracy"] = dict(accuracy)
    return out


def describe(path_str: str, role: str, label: str, report_name: str = "export_report.json") -> dict:
    """One artifact: where it is, what it hashes to, and what is inside it."""
    path = Path(path_str)
    out: dict = {"role": role, "label": label, "path": str(path), "present": False}

    try:
        data = path.read_bytes()
    except OSError as exc:
        out["error"] = f"{type(exc).__name__}: {exc.strerror or exc}"
        return out

    out["present"] = True
    out["size_bytes"] = len(data)
    out["sha256"] = hashlib.sha256(data).hexdigest()
    out.update(_program_facts(data))

    report = _report_facts(path.parent / report_name)
    if report:
        out["report"] = report
    return out


def _host_facts() -> dict:
    out = {
        "machine": platform.machine(),
        "system": platform.system(),
        "python": platform.python_version(),
    }
    # executorch deliberately exposes no __version__, so read distribution
    # metadata; the same applies to anything else reported here.
    import importlib.metadata as md
    for dist in ("executorch", "torch"):
        try:
            out[dist] = md.version(dist)
        except Exception:
            out[dist] = None
    return out


def facts() -> dict:
    """Static facts for both artifacts plus the host. Computed once, cached."""
    global _cache
    if _cache is None:
        _cache = {
            "cpu": describe(
                config.MODEL_PATH,
                role="cpu",
                label="Portable CPU program",
            ),
            "npu": describe(
                config.NPU_MODEL_PATH,
                role="npu",
                label="Hexagon NPU program",
            ),
            "host": _host_facts(),
        }
    return _cache


def warm() -> None:
    """Build the cache at startup, so the first request is not the slow one and
    any parsing problem shows up in the startup log rather than in the UI."""
    data = facts()
    for key in ("cpu", "npu"):
        art = data[key]
        if not art.get("present"):
            print(f"artifact: {key} model not found at {art['path']}")
            continue
        backends = [
            d["backend"]
            for method in art.get("methods", [])
            for d in method.get("delegates", [])
        ]
        detail = ", ".join(sorted(set(backends))) or art.get("error", "no delegate")
        size_mb = art["size_bytes"] / (1024 * 1024)
        print(f"artifact: {key} {Path(art['path']).name} {size_mb:.1f} MB "
              f"sha256 {art['sha256'][:12]} [{detail}]")
    sys.stdout.flush()
