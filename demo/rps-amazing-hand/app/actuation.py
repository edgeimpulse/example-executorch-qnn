# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Actuator contract + implementations: NullActuator, or the Amazing Hand."""

import asyncio
import time
from typing import Protocol

import numpy as np

from data_model import Gesture


class Actuator(Protocol):
    async def prepare(self) -> None: ...
    async def play(self, gesture: Gesture) -> None: ...
    async def wiggle(self) -> None: ...
    async def home(self) -> None: ...
    async def release(self) -> None: ...


class NullActuator:
    async def prepare(self) -> None:
        pass

    async def play(self, gesture: Gesture) -> None:
        pass

    async def wiggle(self) -> None:
        pass

    async def home(self) -> None:
        pass

    async def release(self) -> None:
        pass


class AmazingHandActuator:
    """Drives the Amazing Hand's 8 serial servos (two per finger).

    Settings/calibration were designed for a LEFT hand (side=2); the motion code
    does not use `side` yet, but it is kept as an orientation indicator and for
    future right-hand support. All serial I/O is blocking and includes short
    sleeps, so every public method offloads it to a thread to keep the event
    loop (which also times the countdown audio) responsive.
    """

    def __init__(
        self,
        serial_port: str,
        baudrate: int,
        timeout: float,
        calibration: list[int],
        max_speed: int,
        close_speed: int,
        side: int,
    ) -> None:
        # Imported lazily so the NullActuator path never requires rustypot.
        from rustypot import Scs0009PyController

        self._cal = calibration
        self._max_speed = max_speed
        self._close_speed = close_speed
        self._side = side
        self._c = Scs0009PyController(
            serial_port=serial_port,
            baudrate=baudrate,
            timeout=timeout,
        )
        for motor_id in range(1, 9):
            self._c.write_torque_enable(motor_id, 1)  # 1 = On / 2 = Off / 3 = Free

    # --- Public async contract (each offloads blocking serial to a thread) ---

    async def prepare(self) -> None:
        await asyncio.to_thread(self._prepare)

    async def play(self, gesture: Gesture) -> None:
        gestures = {
            Gesture.ROCK: self._rock,
            Gesture.PAPER: self._paper,
            Gesture.SCISSORS: self._scissors,
        }
        await asyncio.to_thread(gestures[gesture])

    async def wiggle(self) -> None:
        await asyncio.to_thread(self._wiggle)

    async def home(self) -> None:
        await asyncio.to_thread(self._rock)

    async def release(self) -> None:
        # Intentionally leaves the hand in its last gesture.
        pass

    # --- Blocking motion primitives (ported from the demo script) ----------

    def _move_finger(self, m1, m2, c1, c2, angle_1, angle_2, speed) -> None:
        self._c.write_goal_speed(m1, speed)
        time.sleep(0.0002)
        self._c.write_goal_speed(m2, speed)
        time.sleep(0.0002)
        self._c.write_goal_position(m1, np.deg2rad(self._cal[c1] + angle_1))
        self._c.write_goal_position(m2, np.deg2rad(self._cal[c2] + angle_2))
        time.sleep(0.005)

    def _index(self, angle_1, angle_2, speed) -> None:
        self._move_finger(1, 2, 0, 1, angle_1, angle_2, speed)

    def _middle(self, angle_1, angle_2, speed) -> None:
        self._move_finger(3, 4, 2, 3, angle_1, angle_2, speed)

    def _ring(self, angle_1, angle_2, speed) -> None:
        self._move_finger(5, 6, 4, 5, angle_1, angle_2, speed)

    def _thumb(self, angle_1, angle_2, speed) -> None:
        self._move_finger(7, 8, 6, 7, angle_1, angle_2, speed)

    def _rock(self) -> None:
        self._index(-90, 90, self._close_speed)
        self._middle(-90, 90, self._close_speed)
        self._ring(-90, 90, self._close_speed)
        self._thumb(-90, 90, self._close_speed + 1)

    def _paper(self) -> None:
        self._index(30, -30, self._max_speed)
        self._middle(30, -30, self._max_speed)
        self._ring(30, -30, self._max_speed)
        self._thumb(30, -30, self._max_speed)

    def _scissors(self) -> None:
        self._index(30, -30, self._max_speed)
        self._middle(30, -30, self._max_speed)
        self._ring(-90, 90, self._close_speed)
        self._thumb(-90, 90, self._close_speed)

    def _wiggle(self) -> None:
        self._index(-45, 120, self._max_speed)
        self._middle(-45, 120, self._max_speed)
        self._ring(-45, 120, self._max_speed)
        self._thumb(-135, 60, self._max_speed)
        time.sleep(0.2)
        self._rock()

    def _progressive_close(self) -> None:
        self._thumb(-90, 90, self._close_speed - 3)
        time.sleep(0.3)
        self._index(-90, 90, self._close_speed - 3)
        time.sleep(0.3)
        self._middle(-90, 90, self._close_speed - 3)
        time.sleep(0.3)
        self._ring(-90, 90, self._close_speed - 3)

    def _progressive_open(self) -> None:
        self._ring(30, -30, self._max_speed - 3)
        time.sleep(0.3)
        self._middle(30, -30, self._max_speed - 3)
        time.sleep(0.3)
        self._index(30, -30, self._max_speed - 3)
        time.sleep(0.3)
        self._thumb(30, -30, self._max_speed - 3)

    def _prepare(self) -> None:
        # Start from rock so the open flourish is visible even if the last match ended on paper/scissors.
        self._rock()
        time.sleep(0.5)
        self._progressive_open()
        time.sleep(0.5)
        self._progressive_close()
