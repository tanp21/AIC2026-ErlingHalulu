# Repository Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Organize the project for team development and publish it without data, secrets, caches, generated indexes, or frontend build output.

**Architecture:** Keep runtime modules at the repository root for deployment compatibility. Group offline tools by purpose under `scripts/`, keep clean notebooks under `notebooks/`, and build the React frontend in a Docker build stage.

**Tech Stack:** Python 3.12, FastAPI, React/Vite, Docker Compose, Git.

**Spec:** `docs/superpowers/specs/2026-08-18-vbs-search-app-design.md`

## Global Constraints

- Never modify, move, delete, chmod, or stage `data/` or its symlink target.
- Preserve the existing Git history and never force-push.
- Keep HTTP APIs, runtime environment variables, and Docker mounts compatible.

## Tasks

- [ ] Group scripts, notebooks, and reports; update imports and relative paths.
- [ ] Build frontend assets in a Node 22 Docker stage and ignore local dist output.
- [ ] Expand Git/Docker ignores and document local, Docker, notebook, and CLI workflows.
- [ ] Run backend, frontend, shell, CLI, Docker, ignore, and secret checks.
- [ ] Commit on `main`, add the approved SSH remote, and push only when remote history is safe.
