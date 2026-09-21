"""A human checking the agent's candidates while Gemini does, and overruling it.

Gemini judges every candidate the agent submits, but at a competition venue its
replies can take far longer than the ~4 s they take on a quiet network. A
teammate looking at the same keyframes with the same speech, OCR and caption
context can often decide sooner. This is the channel between the two:

- the UI records decisions here (accept or reject a candidate, type a Q&A
  answer, stop the run), from the request thread;
- the orchestrator reads them from its own thread, applies them as verdicts
  that Gemini can never overwrite, and stops waiting on Gemini as soon as a
  decision makes the wait pointless.

`decided` is the moment a human decision ends the run: an accepted candidate,
or an explicit stop. A rejection alone does not: it only removes a candidate.
"""
from __future__ import annotations

import threading

HUMAN_ACCEPT = 10.0
HUMAN_REJECT = 0.0
MAX_ANSWER = 100


class HumanReview:
    def __init__(self):
        self._lock = threading.Lock()
        self._decisions: dict[str, str] = {}
        self._answer: str | None = None
        self._stopped = False
        #: Set when a decision ends the run; the orchestrator waits on it beside Gemini.
        self.event = threading.Event()
        #: Set on every change, so a waiting orchestrator can re-apply decisions.
        self.changed = threading.Event()

    def decide(self, candidate_id: str, decision: str) -> None:
        """`accept`, `reject`, or `clear` (withdraw an earlier decision)."""
        if decision not in ("accept", "reject", "clear"):
            raise ValueError(f"unknown decision {decision!r}")
        with self._lock:
            if decision == "clear":
                self._decisions.pop(candidate_id, None)
            else:
                self._decisions[candidate_id] = decision
            self._signal()

    def set_answer(self, answer: str | None) -> None:
        with self._lock:
            text = " ".join((answer or "").split())[:MAX_ANSWER]
            self._answer = text or None
            self._signal()

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            self._signal()

    def decisions(self) -> dict[str, str]:
        with self._lock:
            return dict(self._decisions)

    @property
    def answer(self) -> str | None:
        with self._lock:
            return self._answer

    @property
    def stopped(self) -> bool:
        with self._lock:
            return self._stopped

    def decided(self) -> bool:
        """True once a human decision makes further model calls pointless."""
        with self._lock:
            return self._decided()

    def snapshot(self) -> dict:
        with self._lock:
            return {"decisions": dict(self._decisions), "answer": self._answer,
                    "stopped": self._stopped, "decided": self._decided()}

    def _decided(self) -> bool:
        return self._stopped or "accept" in self._decisions.values()

    def _signal(self) -> None:
        self.changed.set()
        if self._decided():
            self.event.set()
        else:
            self.event.clear()
