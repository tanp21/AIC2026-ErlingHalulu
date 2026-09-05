# Text Embedding Micro-batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Batch concurrent OpenCLIP text requests from the admin and four teammates into efficient GPU calls without adding user quotas.

**Architecture:** A lazy process-wide `TextEmbeddingBatcher` owns one standard-library worker, waits up to 10 ms, flattens at most five request-items, and slices one raw encoder result back into synchronous callers. The application closes and resets the worker during lifespan shutdown, and image-only rerank bypasses text encoding.

**Tech Stack:** Python 3.11, `concurrent.futures`, `threading.Condition`, NumPy, PyTorch/OpenCLIP, unittest/pytest.

**Spec:** `docs/superpowers/specs/2026-08-28-text-embedding-microbatch-design.md`

## Global Constraints

- Keep all five trusted users admitted concurrently, with no query-rate or hardware quota.
- Preserve the synchronous `core.encode_text()` and `core.encode_texts()` interfaces.
- Wait at most 10 ms and drain at most five request-items per text batch.
- Add no dependency and do not change authentication, HAProxy shaping, or search semantics.
- A failed encoder call must fail every item in that batch without poisoning later batches.

---

### Task 1: Shared OpenCLIP text batcher

**Files:**
- Create: `tests/test_text_batcher.py`
- Modify: `core.py:1-75`

**Interfaces:**
- Produces: `TextEmbeddingBatcher(encoder, window_seconds=0.01, max_requests=5)` with synchronous `encode(texts) -> np.ndarray` and `close() -> None`.
- Produces: `_encode_text_batch(texts) -> np.ndarray`, `close_text_batcher() -> None`.
- Preserves: `encode_text(text) -> np.ndarray`, `encode_texts(texts) -> np.ndarray`.

- [ ] **Step 1: Write failing concurrency, slicing, recovery, and reset tests**

Create tests that use five caller threads plus a barrier. The fake encoder records flattened inputs and returns one distinct row per string. Assert a single call, including a multi-string request, and exact slices. A second fake raises once; assert all concurrent callers receive the error and a later call succeeds. Patch the global encoder to assert `close_text_batcher()` permits lazy recreation.

- [ ] **Step 2: Run tests to verify RED**

Run: `.venv/bin/python -m pytest tests/test_text_batcher.py -q`

Expected: failure because `TextEmbeddingBatcher` and `close_text_batcher` do not exist.

- [ ] **Step 3: Implement the minimal batcher**

Use `ThreadPoolExecutor(max_workers=1)`, `Future`, `Condition`, and `deque`. The worker waits until five request-items or the 10 ms deadline, flattens strings, calls `_encode_text_batch` once, validates the row count, then sets sliced results or the caught exception on each future. `close()` marks the queue closed, drains accepted work, and joins the executor. Guard lazy global creation/reset with a lock.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_text_batcher.py tests/test_temporal_api.py -q`

Expected: all tests pass and the existing multi-string encoding behavior remains normalized.

### Task 2: Image-only rerank fast path and lifecycle cleanup

**Files:**
- Modify: `app.py:217-233,322-359`
- Modify: `tests/test_image_search.py:150-205`
- Modify: `tests/test_startup_warmup.py:59-90`

**Interfaces:**
- Consumes: `core.close_text_batcher() -> None` from Task 1.
- Preserves: `image_conditioned_search(req, image_data=None)` response schema.

- [ ] **Step 1: Write failing fast-path and shutdown tests**

Add an image-search test with `visual_text` and `image_weight=1`; patch `core.encode_text` and assert it is not called while the image vector reaches visual search. Add a lifespan test asserting `core.close_text_batcher` runs after the context exits, including when application work raises.

- [ ] **Step 2: Run tests to verify RED**

Run: `.venv/bin/python -m pytest tests/test_image_search.py tests/test_startup_warmup.py -q`

Expected: the image-only request still calls `encode_text`, and lifespan does not close the batcher.

- [ ] **Step 3: Implement the two minimal branches**

Only call `core.encode_text(visual_text)` when text is non-empty and `req.image_weight < 1`. Wrap warmup and lifespan yield in `try/finally`, calling `core.close_text_batcher()` in `finally`.

- [ ] **Step 4: Run focused tests to verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_image_search.py tests/test_startup_warmup.py tests/test_request_validation.py tests/test_temporal_api.py -q`

Expected: all pass.

### Task 3: Regression and runtime verification

**Files:**
- Verify only; no production file is required.

**Interfaces:**
- Consumes the Task 1 and Task 2 behavior through existing API endpoints.

- [ ] **Step 1: Run the relevant backend regression set**

Run: `.venv/bin/python -m pytest tests/test_access_control.py tests/test_startup_warmup.py tests/test_image_search.py tests/test_temporal_api.py tests/test_remote_load.py -q`

Expected: all pass.

- [ ] **Step 2: Run the complete suite**

Run: `.venv/bin/python -m pytest -q`

Expected: all existing tests pass (environment-dependent skips remain skips).

- [ ] **Step 3: Rebuild and smoke-test the application**

Run: `docker compose up -d --build app ingress`

Run: `docker compose ps`

Expected: both services are healthy.

- [ ] **Step 4: Run the five-client benchmark**

Run the existing `scripts/benchmarks/remote_load.py` against the reachable endpoint with the saved credentials and compare search/rerank errors and latency against `/tmp/aic-unlimited-compute-report.json`.

Expected: no timeouts or load-related HTTP errors; video remains loadable; search/rerank latency is materially lower than the unbatched run.
