#!/usr/bin/env python3
"""Exercise the Amazing Hand on its own, outside the web app.

    .venv/bin/python scripts/test-hand.py                 all poses, with pauses
    .venv/bin/python scripts/test-hand.py --pose paper     one pose and stop
    .venv/bin/python scripts/test-hand.py --list           what the app will use
    .venv/bin/python scripts/test-hand.py --port /dev/ttyUSB0
    .venv/bin/python scripts/test-hand.py --cal 15,-13,0,-12,0,3,10,-10

Why this exists: when the hand misbehaves inside the demo there are four
candidate causes -- the serial port, the user's group membership, the servo
calibration, and the game logic -- and the web UI cannot tell them apart. This
walks the first three in order and fails with the specific fix.

It drives the real AmazingHandActuator from app/actuation.py rather than
reimplementing the poses, so "the script works" and "the app works" cannot
diverge. Nothing here is imported by the app.
"""

from __future__ import annotations

import argparse
import asyncio
import grp
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "app"))

POSES = ("rock", "paper", "scissors")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the Amazing Hand's serial link, permissions and poses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", help="serial port (default: the app's AMAZING_HAND_SERIAL_PORT)")
    parser.add_argument(
        "--pose",
        choices=(*POSES, "wiggle", "prepare", "all"),
        default="all",
        help="run a single movement instead of the whole sequence",
    )
    parser.add_argument(
        "--cal",
        help="override the 8 per-motor neutral offsets, comma-separated, "
             "e.g. 15,-13,0,-12,0,3,10,-10",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=2.0,
        help="seconds to hold each pose so you can look at it (default: 2)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the settings the app would use and exit without moving anything",
    )
    return parser.parse_args()


def parse_calibration(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 8:
        raise SystemExit(
            f"--cal needs exactly 8 values (one per servo), got {len(parts)}.\n"
            "Order is motors 1..8: index, index, middle, middle, ring, ring, thumb, thumb."
        )
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise SystemExit(f"--cal values must be whole degrees: {exc}")


def check_serial_port(port: str) -> None:
    """Fail with the actual remedy, not a generic permissions message."""
    if not os.path.exists(port):
        print(f"FAIL: no serial device at {port}", file=sys.stderr)
        print("", file=sys.stderr)
        candidates = sorted(
            str(p) for pattern in ("ttyACM*", "ttyUSB*") for p in Path("/dev").glob(pattern)
        )
        if candidates:
            print("Serial devices that do exist:", file=sys.stderr)
            for candidate in candidates:
                print(f"  {candidate}", file=sys.stderr)
            print("", file=sys.stderr)
            print(f"Try: {sys.argv[0]} --port {candidates[0]}", file=sys.stderr)
        else:
            print(
                "No ttyACM* or ttyUSB* devices at all. Check the USB cable and that\n"
                "the hand's controller is powered, then re-run.",
                file=sys.stderr,
            )
        raise SystemExit(1)

    readable = os.access(port, os.R_OK)
    writable = os.access(port, os.W_OK)
    if readable and writable:
        print(f"PASS: {port} exists and is read/write for this user.")
        return

    print(f"FAIL: {port} exists but is not accessible by this user.", file=sys.stderr)
    print("", file=sys.stderr)
    try:
        owning_group = grp.getgrgid(os.stat(port).st_gid).gr_name
    except (KeyError, OSError):
        owning_group = "dialout"
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "$USER"
    print(f"The device belongs to the '{owning_group}' group. Add yourself:", file=sys.stderr)
    print(f"  sudo usermod -aG {owning_group} {user}", file=sys.stderr)
    print("", file=sys.stderr)
    print(
        "Group changes only apply to new logins, so then either log out and back\n"
        f"in, or start a shell with the new group: newgrp {owning_group}",
        file=sys.stderr,
    )
    raise SystemExit(1)


async def run(args: argparse.Namespace, config, actuation) -> int:
    port = args.port or config.AMAZING_HAND_SERIAL_PORT
    calibration = parse_calibration(args.cal) if args.cal else config.AMAZING_HAND_MIDDLE_CALIBRATION

    print(f"Serial port  : {port}")
    print(f"Baud rate    : {config.AMAZING_HAND_BAUDRATE}")
    print(f"Side         : {config.AMAZING_HAND_SIDE} "
          f"({'right' if config.AMAZING_HAND_SIDE == 1 else 'left'} hand)")
    print(f"Calibration  : {calibration}"
          f"{'  (from --cal)' if args.cal else '  (from app/config.py)'}")
    print(f"Speeds       : max={config.AMAZING_HAND_MAX_SPEED} "
          f"close={config.AMAZING_HAND_CLOSE_SPEED}")
    print("")

    if args.list:
        print("--list given, so nothing was moved.")
        return 0

    check_serial_port(port)

    try:
        import rustypot  # noqa: F401
    except ImportError:
        print("FAIL: the 'rustypot' package is not installed in this environment.",
              file=sys.stderr)
        print("", file=sys.stderr)
        print("  ./install.sh      (it is in requirements.txt)", file=sys.stderr)
        print(f"Make sure you are running this with .venv/bin/python, not {sys.executable}.",
              file=sys.stderr)
        return 1

    print("")
    print("Opening the serial link and enabling torque on all 8 servos ...")
    print("The hand will move. Keep fingers clear of the linkages.")
    try:
        hand = actuation.AmazingHandActuator(
            serial_port=port,
            baudrate=config.AMAZING_HAND_BAUDRATE,
            timeout=config.AMAZING_HAND_TIMEOUT,
            calibration=calibration,
            max_speed=config.AMAZING_HAND_MAX_SPEED,
            close_speed=config.AMAZING_HAND_CLOSE_SPEED,
            side=config.AMAZING_HAND_SIDE,
        )
    except Exception as exc:  # rustypot raises its own exception types
        print(f"FAIL: could not open the hand on {port}: {exc}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Things worth checking, in order:", file=sys.stderr)
        print("  - the controller is powered (torque needs the external supply,", file=sys.stderr)
        print("    not just USB)", file=sys.stderr)
        print(f"  - nothing else holds the port open: sudo lsof {port}", file=sys.stderr)
        print(f"  - the baud rate matches the controller "
              f"({config.AMAZING_HAND_BAUDRATE})", file=sys.stderr)
        return 1
    print("PASS: serial link open, torque enabled.")

    from data_model import Gesture

    if args.pose == "all":
        steps: list[tuple[str, str]] = [
            ("prepare", "open-and-close flourish (what each match starts with)"),
            ("rock", "all four fingers closed"),
            ("paper", "all four fingers open"),
            ("scissors", "index and middle open, ring and thumb closed"),
            ("wiggle", "the countdown shake, ending back on rock"),
        ]
    else:
        steps = [(args.pose, "")]

    try:
        for name, description in steps:
            label = f"{name}" + (f" -- {description}" if description else "")
            print("")
            print(f"  {label}")
            if name == "prepare":
                await hand.prepare()
            elif name == "wiggle":
                await hand.wiggle()
            else:
                await hand.play(Gesture(name))
            await asyncio.sleep(args.pause)
    except KeyboardInterrupt:
        print("")
        print("Interrupted. Returning to the rest pose ...")
        await hand.home()
        return 130
    except Exception as exc:
        print("")
        print(f"FAIL: a movement raised {type(exc).__name__}: {exc}", file=sys.stderr)
        print("", file=sys.stderr)
        print("The link opened, so this is the servos or the calibration rather", file=sys.stderr)
        print("than the port. If one finger is clearly out of position, adjust", file=sys.stderr)
        print("that motor's offset in AMAZING_HAND_MIDDLE_CALIBRATION and retry", file=sys.stderr)
        print("with --cal before editing app/config.py.", file=sys.stderr)
        return 1

    print("")
    print("Returning to the rest pose (rock) ...")
    await hand.home()

    print("")
    print("All movements completed without error.")
    print("")
    print("What to look for, since a clean exit only proves the servos accepted")
    print("the commands:")
    print("  - paper should be flat, with all four fingers fully extended")
    print("  - scissors should show exactly two fingers")
    print("  - no servo should be buzzing or straining at the end of a pose")
    print("")
    print("If a finger sits visibly off, that motor's entry in")
    print("AMAZING_HAND_MIDDLE_CALIBRATION (app/config.py) is the thing to change;")
    print("the order is motors 1..8 = index, index, middle, middle, ring, ring,")
    print("thumb, thumb. Try a value with --cal first, then make it permanent.")
    print("")
    print("Then run the demo with the hand enabled:")
    print(f"  ENABLE_AMAZING_HAND=1 AMAZING_HAND_SERIAL_PORT={port} ./start.sh")
    return 0


def main() -> int:
    # Progress goes to stdout and failures to stderr; without line buffering
    # the two interleave wrongly as soon as the output is piped to a file.
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    try:
        import actuation
        import config
    except ImportError as exc:
        print(f"FAIL: could not import the app's modules: {exc}", file=sys.stderr)
        print("Run this from the project root, with the project's interpreter:",
              file=sys.stderr)
        print("  .venv/bin/python scripts/test-hand.py", file=sys.stderr)
        return 1
    return asyncio.run(run(args, config, actuation))


if __name__ == "__main__":
    sys.exit(main())
