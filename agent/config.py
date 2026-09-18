"""Agent settings from the environment, with the repository `.env` as a fallback."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_dotenv(path: Path) -> dict[str, str]:
    """`KEY=value` lines only; comments, blanks and malformed lines are skipped."""
    values = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


@dataclass(frozen=True)
class AgentSettings:
    server_url: str
    server_user: str | None
    server_password: str | None
    # Direct-to-uvicorn development only: HAProxy normally sets X-AIC-User after Basic
    # Auth, so without it the server rejects every route but /healthz.
    trusted_user: str | None
    deepseek_api_key: str
    deepseek_model: str
    deepseek_base_url: str
    # Loop turns reason better with thinking; planning is 5-10x faster without it
    # and loses little (measured on the round-1 2026 queries).
    deepseek_thinking: bool
    deepseek_plan_thinking: bool
    gemini_api_key: str
    gemini_model: str
    gemini_thinking_level: str
    # Only for a server whose transcript index is raw ASR (numbers spelled out).
    spoken_numbers: bool = False
    # auto | batch | compat -- see SearchClient.
    search_api: str = "auto"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None, dotenv: Path = ROOT / ".env"):
        merged = {**read_dotenv(dotenv), **(os.environ if env is None else env)}

        def required(name):
            value = merged.get(name, "").strip()
            if not value:
                raise RuntimeError(f"{name} is not set (environment or {dotenv.name})")
            return value

        return cls(
            server_url=merged.get("AIC_AGENT_URL", "http://127.0.0.1:8000").rstrip("/"),
            server_user=merged.get("AIC_AGENT_USER") or None,
            server_password=merged.get("AIC_AGENT_PASSWORD") or None,
            trusted_user=merged.get("AIC_AGENT_TRUSTED_USER") or None,
            deepseek_api_key=required("DEEPSEEK_API_KEY"),
            deepseek_model=merged.get("DEEPSEEK_MODEL", "deepseek-flash"),
            deepseek_base_url=merged.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            deepseek_thinking=merged.get("DEEPSEEK_THINKING", "enabled").lower() != "disabled",
            deepseek_plan_thinking=merged.get("DEEPSEEK_PLAN_THINKING", "disabled").lower() == "enabled",
            gemini_api_key=required("GEMINI_API_KEY"),
            gemini_model=merged.get("GEMINI_MODEL", "gemini-3.8-flash"),
            gemini_thinking_level=merged.get("GEMINI_THINKING_LEVEL", "low"),
            spoken_numbers=merged.get("AIC_AGENT_SPOKEN_NUMBERS", "0").lower() in ("1", "true", "yes"),
            search_api=merged.get("AIC_AGENT_SEARCH_API", "auto"),
        )
