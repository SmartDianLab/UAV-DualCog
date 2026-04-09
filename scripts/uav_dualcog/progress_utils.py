from __future__ import annotations

import contextlib
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


def _format_seconds(seconds: float) -> str:
    sec = max(0, int(seconds))
    minute, sec = divmod(sec, 60)
    hour, minute = divmod(minute, 60)
    if hour > 0:
        return f"{hour:02d}:{minute:02d}:{sec:02d}"
    return f"{minute:02d}:{sec:02d}"


class _TTYProgressManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._order: list[str] = []
        self._lines: dict[str, str] = {}
        self._rendered_count = 0

    def render(self, label: str, line: str) -> None:
        with self._lock:
            if label not in self._order:
                self._order.append(label)
            self._lines[label] = line
            target_count = len(self._order)
            if self._rendered_count == 0:
                self._rendered_count = target_count
            elif target_count > self._rendered_count:
                print("\n" * (target_count - self._rendered_count), end="", flush=True)
                self._rendered_count = target_count

            move_up = max(0, self._rendered_count - 1)
            if move_up > 0:
                print(f"\x1b[{move_up}A", end="", flush=False)

            for idx, item_label in enumerate(self._order):
                text = self._lines.get(item_label, "")
                end = "\n" if idx < len(self._order) - 1 else ""
                print(f"\r\x1b[2K{text}", end=end, flush=False)
            sys.stdout.flush()


_TTY_PROGRESS_MANAGER = _TTYProgressManager()


@dataclass
class StageLogger:
    name: str

    def _emit(self, level: str, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}][{self.name}][{level}] {message}", flush=True)

    def info(self, message: str) -> None:
        self._emit("INFO", message)

    def warn(self, message: str) -> None:
        self._emit("WARN", message)

    def error(self, message: str) -> None:
        self._emit("ERROR", message)

    @contextlib.contextmanager
    def timed(self, step_name: str):
        t0 = time.time()
        self.info(f"{step_name} - start")
        try:
            yield
        finally:
            dt = time.time() - t0
            self.info(f"{step_name} - done in {_format_seconds(dt)}")


class ProgressBar:
    def __init__(
        self,
        total: int,
        label: str = "progress",
        width: int = 28,
        min_interval_sec: float = 0.5,
        emit_fn: Callable[[str], Any] | None = None,
        label_width: int | None = None,
    ):
        self.total = max(0, int(total))
        self.label = label
        self.label_width = int(label_width) if label_width is not None else None
        self.width = max(10, int(width))
        self.min_interval_sec = float(min_interval_sec)
        self.emit_fn = emit_fn
        self.start_ts = time.time()
        self.last_emit_ts = 0.0
        self.current = 0
        self.is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())

    def _render(self, detail: str = "") -> str:
        elapsed = time.time() - self.start_ts
        if self.total <= 0:
            pct = 100.0
            filled = self.width
            eta_text = "00:00"
        else:
            pct = min(100.0, max(0.0, (self.current / max(1, self.total)) * 100.0))
            filled = int(round((pct / 100.0) * self.width))
            if self.current > 0:
                eta_sec = elapsed * (self.total - self.current) / max(1, self.current)
                eta_text = _format_seconds(eta_sec)
            else:
                eta_text = "--:--"

        bar = "#" * filled + "-" * (self.width - filled)
        label_text = self.label
        if self.label_width is not None and self.label_width > 0:
            label_text = f"{label_text:<{self.label_width}}"
        base = f"[{label_text}] [{bar}] {pct:6.2f}% ({self.current}/{self.total}) elapsed={_format_seconds(elapsed)} eta={eta_text}"
        if detail:
            return f"{base} | {detail}"
        return base

    def update(self, current: int, detail: str = "") -> None:
        self.current = min(max(0, int(current)), self.total if self.total > 0 else int(current))
        now = time.time()
        should_emit = False
        if self.current == 0:
            should_emit = True
        elif self.current >= self.total:
            should_emit = True
        elif (now - self.last_emit_ts) >= self.min_interval_sec:
            should_emit = True

        if not should_emit:
            return

        line = self._render(detail=detail)
        if self.is_tty:
            _TTY_PROGRESS_MANAGER.render(self.label, line)
        else:
            print(line, flush=True)
        if callable(self.emit_fn):
            try:
                self.emit_fn(line)
            except Exception:
                pass
        self.last_emit_ts = now

    def advance(self, delta: int = 1, detail: str = "") -> None:
        self.update(self.current + int(delta), detail=detail)

    def finish(self, detail: str = "") -> None:
        target = self.total if self.total > 0 else self.current
        self.update(target, detail=detail)
