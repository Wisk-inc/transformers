# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
How long a notebook session may run, and the run's plan to be done before it ends.

molab stops a notebook 12 hours after it starts ("Notebooks can run for as long as 12 hours before molab shuts
them down", docs.marimo.io/guides/molab). The TriStream program ran there on 24 October 2026: the data stage took
six hours, and the session was stopped in the middle of the prepare stage, which then had to start again from its
first file. A session that knows its end can save, push to the Hub and stop a little before it, so that the next
session goes on from there.

The end is ``HELIX_SESSION_HOURS`` when it is set (0 for none), 12 hours on molab, and no end elsewhere. The start
is ``HELIX_SESSION_START`` (seconds since 1970) when given, or the earliest start among this process and the
processes above it (the notebook kernel, its server, the container) that is less than the limit ago.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

MOLAB_HOURS = 12.0
#: minutes before a session's end at which a stage saves, pushes and stops
WRAP_UP_MINUTES = 30.0
#: where molab keeps a notebook's packages
MOLAB_PACKAGES = "/tmp/uv-venv"


class SessionEnding(Exception):
    """Raised by a stage that saved its work because the session is about to be stopped."""


def _proc_stat(pid: int) -> list:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()


def ancestors(most: int = 16) -> list:
    """This process and the ones above it, nearest first."""
    out, pid = [], os.getpid()
    while pid > 0 and len(out) < most:
        out.append(pid)
        try:
            pid = int(_proc_stat(pid)[1])
        except (OSError, ValueError, IndexError):
            break
    return out


def process_start(pid: int) -> float | None:
    """When a process started, in seconds since 1970, or None where there is no ``/proc``."""
    try:
        boot = next(float(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                    if line.startswith("btime"))
        return boot + int(_proc_stat(pid)[19]) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, StopIteration):
        return None


def command_line(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def on_molab(env=None, places=None, commands=None) -> bool:
    """Whether this runs in a molab notebook: the packages are where molab keeps them, a process above this one is
    marimo, or marimo's settings are in the environment."""
    env = os.environ if env is None else env
    places = [sys.prefix, __file__] if places is None else places
    commands = [command_line(pid) for pid in ancestors()[1:]] if commands is None else commands
    return (any(str(p).startswith(MOLAB_PACKAGES) for p in places) or any("marimo" in c for c in commands)
            or any(k.startswith("MARIMO") or "MOLAB" in k for k in env))


def session_limit(env=None) -> tuple:
    """``(hours, why)``, or ``(None, why)`` when nothing says the session ends."""
    env = os.environ if env is None else env
    given = str(env.get("HELIX_SESSION_HOURS", "")).strip()
    if given:
        hours = float(given)
        return (hours, "HELIX_SESSION_HOURS") if hours > 0 else (None, "HELIX_SESSION_HOURS=0")
    if on_molab(env):
        return MOLAB_HOURS, "molab stops a notebook after 12 hours"
    return None, "nothing says this machine stops its sessions"


def session_start(hours: float, env=None, now: float | None = None) -> float:
    """When this session began (see the module's notes)."""
    env = os.environ if env is None else env
    now = time.time() if now is None else now
    if env.get("HELIX_SESSION_START"):
        return float(env["HELIX_SESSION_START"])
    times = [t for t in map(process_start, ancestors()) if t is not None and now - hours * 3600.0 < t <= now]
    return min(times, default=now)


class Session:
    """The session this process runs in. ``wrap_up()`` is True from ``WRAP_UP_MINUTES`` before its end."""

    def __init__(self, hours: float | None = None, began: float | None = None, why: str = "",
                 wrap_up_minutes: float = WRAP_UP_MINUTES) -> None:
        self.hours, self.why, self.wrap_minutes = hours, why, wrap_up_minutes
        self.began = began if began is not None else time.time()
        self.end = None if not hours else self.began + hours * 3600.0

    @classmethod
    def find(cls, env=None) -> Session:
        hours, why = session_limit(env)
        return cls(hours, session_start(hours, env) if hours else None, why)

    def left_hours(self, now: float | None = None) -> float:
        if self.end is None:
            return float("inf")
        return (self.end - (time.time() if now is None else now)) / 3600.0

    def wrap_up(self, now: float | None = None) -> bool:
        return self.end is not None and self.left_hours(now) * 60.0 <= self.wrap_minutes

    def describe(self) -> str:
        if self.end is None:
            return f"no session end ({self.why})"
        clock = lambda t: time.strftime("%H:%M", time.gmtime(t))          # noqa: E731
        return (f"this session began at {clock(self.began)} UTC and ends {self.hours:g} hours later ({self.why}); "
                f"the run saves, pushes to the Hub and stops by {clock(self.end - self.wrap_minutes * 60)} UTC")

    def environment(self) -> dict:
        """What a child process needs to plan the same end."""
        return {"HELIX_SESSION_HOURS": f"{self.hours or 0:g}", "HELIX_SESSION_START": f"{self.began:.0f}"}


__all__ = ["MOLAB_HOURS", "WRAP_UP_MINUTES", "Session", "SessionEnding", "on_molab", "session_limit",
           "session_start"]
