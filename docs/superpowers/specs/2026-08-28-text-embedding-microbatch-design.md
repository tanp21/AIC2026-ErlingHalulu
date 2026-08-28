# Text Embedding Micro-batching Design

## Context

The public benchmark now admits the admin and four teammates concurrently, without per-user query-rate or compute limits. HAProxy gives teammate video traffic a shared 24 MB/s ceiling, while localhost admin traffic bypasses that Internet shaping.

Under five simultaneous searches, admission wait is zero and FAISS takes about 153 ms, but total search latency exceeds four seconds. GPU memory remains below its limit and GPU utilization is modest. The bottleneck is therefore multiple concurrent calls into the shared OpenCLIP text encoder. Image rerank also encodes text unnecessarily when `image_weight=1`.

## Goals

- Keep all five trusted users admitted concurrently, with no query-rate or hardware quota.
- Combine simultaneous text-embedding work into efficient GPU batches.
- Preserve the existing synchronous `core.encode_text()` and `core.encode_texts()` interfaces.
- Allow FAISS work to proceed concurrently after each embedding result is returned.
- Avoid new dependencies and clean up background resources during application shutdown.

## Non-goals

- Changing authentication, public port forwarding, HAProxy bandwidth policy, or search result semantics.
- Serializing whole search requests or reserving GPU capacity for the localhost admin.
- Batching image embeddings, FAISS searches, or video delivery in this change.
- Adding quotas for trusted users.

## Design

### Shared text batcher

`core.py` will contain a small `TextEmbeddingBatcher` backed by standard-library synchronization and futures. Each submitted item contains an ordered tuple of input strings and a `Future` for its resulting NumPy rows.

The first queued item schedules one worker. The worker waits for a batching window of at most 10 ms, then drains up to five request-items. It flattens their strings into one list, invokes the raw OpenCLIP text encoder exactly once, and slices the returned matrix back into the original request boundaries. This means five ordinary searches become one five-string model call, while a temporal request containing several event strings shares the same call and receives its own correctly ordered rows.

The limit applies to request-items, not strings. Five is aligned with the currently trusted admin-plus-four-teammate workload and keeps batching bounded without restricting admission. If more work is queued, the worker immediately processes subsequent batches.

`encode_texts()` submits to the shared batcher and waits on its future. `encode_text()` remains a one-row wrapper around `encode_texts()`. The current model invocation moves into a private raw encoder function so the worker cannot recursively submit to itself.

The global batcher is created lazily, allowing command-line and test callers to keep using `core` outside the web application. Application lifespan shutdown closes and resets it. Closing stops new submissions, completes already accepted work, and releases the worker executor so repeated test lifespans do not leak threads.

### Failure behavior

If tokenization or model inference fails, every request-item included in that GPU batch receives the same exception through its future. The worker remains usable and processes the next batch; one failed inference must not permanently wedge the queue.

### Image rerank fast path

Image-conditioned search will only encode `visual_text` when its contribution can affect the fused vector. With `image_weight=1`, it will pass no text vector and use the existing image-only fusion path. Other weights retain current behavior.

### Concurrency and priority

The existing admission coordinator continues to allow five concurrent trusted searches and continues to reserve no compute quota. Micro-batching coordinates only the short text-encoder phase. Once a request obtains its embedding, its FAISS and result-processing stages run under the existing concurrency model.

Admin traffic through localhost remains outside HAProxy bandwidth shaping. Public teammate video traffic continues to share the configured Internet ceiling; query compute is not shaped by role.

## Testing

- Unit-test five simultaneous request-items and assert one raw model call plus correct per-request slices and ordering.
- Include a multi-string temporal item to verify flattening and reconstruction.
- Verify one failed batch propagates errors to all included futures and a later batch succeeds.
- Verify application shutdown closes and resets the batcher without leaving it unusable for the next lifespan.
- Verify `image_weight=1` performs no text encoding, while mixed weights still do.
- Run the focused access-control, startup, image-search, temporal-search, and remote-benchmark tests, followed by the full test suite.
- Rebuild the application and repeat the five-client benchmark, checking for zero request errors and comparing admin/teammate search and rerank latency with the unbatched baseline.

## Acceptance criteria

- Five concurrent ordinary text searches use one OpenCLIP text-encoding call when they arrive within the 10 ms window.
- No trusted user receives a rate-limit response or an artificial compute quota.
- Batcher failures do not deadlock or poison later requests.
- Image-only rerank does not call the text encoder.
- Existing search correctness tests pass, and the five-client benchmark completes without timeouts or load-related HTTP errors.
