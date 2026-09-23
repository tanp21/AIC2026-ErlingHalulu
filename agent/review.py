"""A human checking the agent's candidates, alongside Gemini or instead of it.

Two ways to use it, chosen by `judge`:

- `"model"` (the default): Gemini judges every candidate the agent submits, but at a
  competition venue its replies can take far longer than the ~4 s they take on a quiet
  network. A teammate looking at the same keyframes with the same speech, OCR and caption
  context can often decide sooner, so the UI records decisions here (accept or reject a
  candidate, type a Q&A answer, stop the run) and the orchestrator applies them as verdicts
  Gemini can never overwrite, and stops waiting on Gemini as soon as a decision makes the
  wait pointless. An acceptance ends the run.
- `"human"`: no model ever sees a frame. The person is the judge, so an acceptance only
  ranks a candidate first -- they may accept several before submitting -- and the run
  ends when they say so (`finish`) or stop it. They also steer the search: `commands`
  carries "search more", "search inside this video", "add this neighbour" and "add this
  exact frame", which the orchestrator runs on its own thread, and `sequence` carries a
  TRAKE path they assembled.

`decided` is the moment a human decision ends the run. A rejection or a score alone never
does: it only moves a candidate.
"""
from __future__ import annotations

from collections import deque
import threading
import time

HUMAN_ACCEPT = 10.0
HUMAN_REJECT = 0.0
MAX_ANSWER = 100
MAX_HINT = 300
MAX_LOG = 500
MAX_ORDER = 300
COMMAND_TYPES = ("more", "scope", "neighbours", "add_frame")


class HumanReview:
    def __init__(self, judge: str = "model", clock=time.monotonic):
        if judge not in ("model", "human"):
            raise ValueError(f"unknown judge {judge!r}")
        self.judge = judge
        self._clock = clock
        self._started = clock()
        self._lock = threading.Lock()
        self._decisions: dict[str, str] = {}
        self._scores: dict[str, float] = {}
        self._answer: str | None = None
        self._stopped = False
        self._finished = False
        self._sequence: dict | None = None
        self._order: list[str] = []
        self._commands: deque[dict] = deque()
        self._log: list[dict] = []
        #: Set when a decision ends the run; the orchestrator waits on it beside Gemini.
        self.event = threading.Event()
        #: Set on every change, so a waiting orchestrator can re-apply decisions.
        self.changed = threading.Event()

    # -- what the reviewer can do ---------------------------------------------------
    def decide(self, candidate_id: str, decision: str) -> None:
        """`accept`, `reject`, or `clear` (withdraw an earlier decision or score)."""
        if decision not in ("accept", "reject", "clear"):
            raise ValueError(f"unknown decision {decision!r}")
        with self._lock:
            self._scores.pop(candidate_id, None)
            if decision == "clear":
                self._decisions.pop(candidate_id, None)
            else:
                self._decisions[candidate_id] = decision
            self._record("decide", candidate_id=candidate_id, decision=decision)
            self._signal()

    def score(self, candidate_id: str, value: float) -> None:
        """A 0-10 verdict between accept (10) and reject (0): 5 is "right video, wrong moment"."""
        value = float(value)
        if not 0.0 <= value <= 10.0:
            raise ValueError(f"score {value} is outside 0-10")
        with self._lock:
            self._decisions.pop(candidate_id, None)
            self._scores[candidate_id] = value
            self._record("score", candidate_id=candidate_id, score=value)
            self._signal()

    def set_answer(self, answer: str | None) -> None:
        with self._lock:
            text = " ".join((answer or "").split())[:MAX_ANSWER]
            self._answer = text or None
            self._record("answer", answer=self._answer)
            self._signal()

    def set_sequence(self, video_id: str, frames: list[int], times: list[float] | None = None) -> None:
        """The TRAKE path the reviewer assembled: one frame per event, in order."""
        if not frames:
            raise ValueError("a sequence needs at least one frame")
        if times is not None and len(times) != len(frames):
            raise ValueError("a sequence needs one time per frame")
        with self._lock:
            self._sequence = {"video_id": video_id, "frames": [int(f) for f in frames]}
            if times is not None:
                self._sequence["times"] = [float(t) for t in times]
            self._record("sequence", **self._sequence)
            self._signal()

    def set_order(self, order: list[str]) -> None:
        """The order the reviewer was looking at when they finished, best first.

        Candidates nobody judged are ranked by the agent's retrieval fusion, and the reviewer's
        view re-ranks them by what the words around each say. What is submitted must be what was
        on the screen, so the view tells the run.
        """
        with self._lock:
            self._order = [str(cid) for cid in order][:MAX_ORDER]
            self._record("order", n=len(self._order))
            self._signal()

    def push(self, command: dict) -> None:
        """Queue a search command for the orchestrator: `more`, `scope`, `neighbours`, `add_frame`."""
        kind = command.get("type")
        if kind not in COMMAND_TYPES:
            raise ValueError(f"unknown command {kind!r}")
        if kind == "more":
            command = {**command, "hint": " ".join(str(command.get("hint") or "").split())[:MAX_HINT]}
        with self._lock:
            self._commands.append(dict(command))
            self._record("command", **command)
            self._signal()

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            self._record("stop")
            self._signal()

    def finish(self) -> None:
        """Submit: the reviewer is done, whatever the run still has in flight."""
        with self._lock:
            self._finished = True
            self._record("finish")
            self._signal()

    # -- what the orchestrator reads -------------------------------------------------
    def decisions(self) -> dict[str, str]:
        with self._lock:
            return dict(self._decisions)

    def verdicts(self) -> dict[str, float]:
        """Every candidate the reviewer has judged, as a 0-10 score."""
        with self._lock:
            found = {cid: HUMAN_ACCEPT if decision == "accept" else HUMAN_REJECT
                     for cid, decision in self._decisions.items()}
            found.update(self._scores)
            return found

    def take_commands(self) -> list[dict]:
        with self._lock:
            commands = list(self._commands)
            self._commands.clear()
            return commands

    @property
    def answer(self) -> str | None:
        with self._lock:
            return self._answer

    @property
    def order(self) -> list[str]:
        with self._lock:
            return list(self._order)

    @property
    def sequence(self) -> dict | None:
        with self._lock:
            return dict(self._sequence) if self._sequence else None

    @property
    def stopped(self) -> bool:
        with self._lock:
            return self._stopped

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._finished

    def decided(self) -> bool:
        """True once a human decision makes further waiting pointless."""
        with self._lock:
            return self._decided()

    def log(self) -> list[dict]:
        """Every reviewer action with the second it happened at, for measuring the human."""
        with self._lock:
            return list(self._log)

    def snapshot(self) -> dict:
        with self._lock:
            return {"decisions": dict(self._decisions), "answer": self._answer,
                    "stopped": self._stopped, "decided": self._decided(),
                    "judge": self.judge, "scores": dict(self._scores),
                    "sequence": dict(self._sequence) if self._sequence else None,
                    "finished": self._finished, "pending_commands": len(self._commands)}

    # -- internals -------------------------------------------------------------------
    def _decided(self) -> bool:
        if self._stopped or self._finished:
            return True
        # With a model judging, one acceptance settles the query; with a person judging
        # it only ranks the candidate, and they say when they are done.
        return self.judge == "model" and "accept" in self._decisions.values()

    def _record(self, action: str, **details) -> None:
        if len(self._log) < MAX_LOG:
            self._log.append({"t": round(self._clock() - self._started, 2), "action": action, **details})

    def _signal(self) -> None:
        self.changed.set()
        if self._decided():
            self.event.set()
        else:
            self.event.clear()
