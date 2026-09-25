"""Agent settings from the environment, with the repository `.env` as a fallback."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: The settings that describe *this deployment's* vision backend rather than one
#: run. A server-hosted run has to inherit them: the GPU is on the server, so a
#: teammate's browser cannot choose the backend, and the keys they paste are their
#: own model credentials, not a statement about where the vision work happens.
VISION_ENV = (
    "AIC_AGENT_VISION", "AIC_VLM_URL", "AIC_VLM_MODEL", "AIC_VLM_DIR",
    "AIC_VLM_DEVICE", "AIC_VLM_MAX_NEW_TOKENS", "AIC_VLM_IMAGE_SIDE",
    "AIC_VLM_ANSWER_SIDE", "AIC_VLM_MAX_TILES", "AIC_AGENT_CAPTIONS",
    "OPENROUTER_MODEL", "AIC_AGENT_JUDGE_STYLE",
)
#: How candidates are judged (`AgentOptions.judge_style`).
JUDGE_STYLES = ("keyframe", "moment")

#: Which vision backends `AIC_AGENT_VISION` accepts. `openrouter` is a remote API like
#: `gemini`, and is what the team falls back to when the Gemini quota is gone.
VISION_BACKENDS = ("gemini", "local", "deepseek", "openrouter")


#: Where the team's OpenRouter key may be found, first match wins. `QWEN_API_KEY` is the name
#: the key was first handed round under (it is the key the Qwen judge runs on), so both work.
OPENROUTER_KEY_NAMES = ("OPENROUTER_API_KEY", "QWEN_API_KEY")


def openrouter_key(env=None) -> str:
    """The OpenRouter key under either of its names, or ""."""
    env = os.environ if env is None else env
    for name in OPENROUTER_KEY_NAMES:
        value = (env.get(name) or "").strip()
        if value:
            return value
    return ""


def default_vision(env=None) -> str:
    """`AIC_AGENT_VISION` if set; otherwise Qwen on OpenRouter when there is a key for it.

    Gemini judged every frame until the team moved the judging to `qwen/qwen3.7-flash` on
    OpenRouter for cost (2026-09): Gemini stays the web-search client, and is the judge only
    where no OpenRouter key exists or a deployment asks for it by name.
    """
    env = os.environ if env is None else env
    chosen = (env.get("AIC_AGENT_VISION") or "").strip().lower()
    if chosen:
        return chosen
    return "openrouter" if openrouter_key(env) else "gemini"


def deployment_vision() -> str:
    """Which vision backend this process was deployed with."""
    return default_vision()


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
    # One key, many models. Required only when it is the thing doing the looking
    # (vision == "openrouter", or checklist_vision_model == "openrouter").
    openrouter_api_key: str = ""
    openrouter_model: str = "qwen/qwen3.7-flash"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Only for a server whose transcript index is raw ASR (numbers spelled out).
    spoken_numbers: bool = False
    # auto | batch | compat -- see SearchClient.
    search_api: str = "auto"
    # gemini | local | deepseek -- which backend does the five vision jobs (see
    # agent.vlm). `gemini` stays the default: it is the measured baseline, and it is
    # what a run falls back to if a local model turns out not to hold up. `deepseek`
    # is the same shape as gemini — a remote API — and exists because it may judge
    # more accurately, not because it helps with a congested uplink.
    vision: str = "gemini"
    # Only read when vision == "local". Set it to run the model in another process
    # or on another host, exactly as AIC_CORE_URL splits backend/search_core.
    vlm_url: str | None = None
    vlm_model: str | None = None
    vlm_device: str | None = None
    vlm_max_new_tokens: int = 128
    vlm_image_side: int = 448
    vlm_answer_side: int = 1280
    vlm_max_tiles: int = 6
    # Read the CapRL shot caption into each candidate's label. Ignored by Gemini,
    # which looks at the frame itself; it is the local models that benefit.
    captions: bool = True
    # human judge only: show a model each candidate's own frame while building the checklist
    # (agent/evidence.py). A caption is a guess made once offline; the pixels are ground truth.
    checklist_vision: bool = True
    # human judge only, checklist_vision: which client reads those pixels -- deepseek (default,
    # no extra key) or gemini (agent.gemini; see AgentOptions.checklist_vision_model).
    checklist_vision_model: str = "deepseek"
    # keyframe | moment -- see AgentOptions.judge_style. Defaults to `moment` for the
    # OpenRouter judge, which is the shape it was built for, and `keyframe` otherwise.
    judge_style: str = "keyframe"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None, dotenv: Path = ROOT / ".env"):
        merged = {**read_dotenv(dotenv), **(os.environ if env is None else env)}

        def required(name):
            value = merged.get(name, "").strip()
            if not value:
                raise RuntimeError(f"{name} is not set (environment or {dotenv.name})")
            return value

        def integer(name, default):
            raw = merged.get(name, "").strip()
            return int(raw) if raw else default

        vision = default_vision(merged)
        if vision not in VISION_BACKENDS:
            allowed = ", ".join(repr(name) for name in VISION_BACKENDS)
            raise RuntimeError(f"AIC_AGENT_VISION must be one of {allowed}, not {vision!r}")
        judge_style = (merged.get("AIC_AGENT_JUDGE_STYLE") or "").strip().lower() or (
            "moment" if vision == "openrouter" else "keyframe")
        if judge_style not in JUDGE_STYLES:
            raise RuntimeError(f"AIC_AGENT_JUDGE_STYLE must be 'keyframe' or 'moment', not {judge_style!r}")
        checklist_vision_model = merged.get("AIC_AGENT_CHECKLIST_MODEL", "deepseek").strip().lower()
        if checklist_vision_model not in ("deepseek", "gemini", "openrouter"):
            raise RuntimeError(
                f"AIC_AGENT_CHECKLIST_MODEL must be 'deepseek', 'gemini' or 'openrouter', "
                f"not {checklist_vision_model!r}")

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
            # Optional when the vision backend is local: an offline competition day
            # has no Gemini anyway, and only the web lookup goes without. Also optional
            # for checklist_vision_model == "gemini": unlike the autonomous judge, a
            # missing key there just leaves the checklist without its vision half
            # (`vision_summarize` catches a `judge()` failure per candidate and falls
            # back to the text-only checklist) rather than failing the whole run --
            # a teammate without their own Gemini key can still run human-judge mode.
            gemini_api_key=(required("GEMINI_API_KEY") if vision == "gemini"
                            else merged.get("GEMINI_API_KEY", "").strip()),
            gemini_model=merged.get("GEMINI_MODEL", "gemini-3.8-flash"),
            gemini_thinking_level=merged.get("GEMINI_THINKING_LEVEL", "low"),
            # Required only when OpenRouter is the one doing the looking. As a checklist
            # model a missing key costs the checklist its vision half and nothing else,
            # exactly as a missing Gemini key does there.
            openrouter_api_key=((openrouter_key(merged) or required("OPENROUTER_API_KEY"))
                                if vision == "openrouter" else openrouter_key(merged)),
            openrouter_model=merged.get("OPENROUTER_MODEL", "qwen/qwen3.7-flash"),
            openrouter_base_url=merged.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
            spoken_numbers=merged.get("AIC_AGENT_SPOKEN_NUMBERS", "0").lower() in ("1", "true", "yes"),
            search_api=merged.get("AIC_AGENT_SEARCH_API", "auto"),
            vision=vision,
            vlm_url=(merged.get("AIC_VLM_URL") or "").rstrip("/") or None,
            vlm_model=merged.get("AIC_VLM_MODEL") or None,
            vlm_device=merged.get("AIC_VLM_DEVICE") or None,
            vlm_max_new_tokens=integer("AIC_VLM_MAX_NEW_TOKENS", 128),
            vlm_image_side=integer("AIC_VLM_IMAGE_SIDE", 448),
            vlm_answer_side=integer("AIC_VLM_ANSWER_SIDE", 1280),
            vlm_max_tiles=integer("AIC_VLM_MAX_TILES", 6),
            captions=merged.get("AIC_AGENT_CAPTIONS", "1").lower() not in ("0", "false", "no"),
            checklist_vision=merged.get("AIC_AGENT_CHECKLIST_VISION", "1").lower() not in ("0", "false", "no"),
            checklist_vision_model=checklist_vision_model,
            judge_style=judge_style,
        )
