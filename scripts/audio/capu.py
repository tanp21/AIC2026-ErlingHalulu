"""Vietnamese capitalisation and punctuation restoration.

Not one of the 873 transcripts contains a full stop, a comma, or a capital
letter — the recogniser emits a single lowercase run-on per video. Two things
suffer:

* **Subtitle cues.** ``backend.audio_search.build_transcript_cues`` splits on sentence
  punctuation first and only falls back to an 8-second gap or 160 characters.
  With no punctuation anywhere, the fallback is the *only* rule that ever fires.
* **Passage embeddings.** A 60-second window arrives at the encoder as one
  unpunctuated block.

BM25 is deliberately not on that list. Elasticsearch's ``standard`` analyser
lowercases and drops punctuation, so restoration cannot move a lexical score by
construction — measure this with ``bench_audio_search --dense``, never the
default lexical mode.

This is the only module in :mod:`scripts.audio` that loads a model, so
everything else stays testable without a GPU.

Both candidates come from the same Interspeech 2022 work (*Vietnamese
Capitalization and Punctuation Recovery Models*) and neither is a plain
``AutoModelForTokenClassification``: ``config.json`` declares the custom
``Seq2LabelsModel`` architecture, carries no ``id2label``, and keeps its label
vocabulary in a ``vocabulary/`` directory alongside a ``gec_model.py``. They
therefore need ``trust_remote_code=True`` and a one-time download while the
network is up. That runs here, on the dev machine — never in the container,
which is read-only with ``HF_HUB_OFFLINE=1``.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path

from scripts.audio.postprocess import Word, apply_capu_output

#: Where the snapshots are materialised. Outside the repo and outside the
#: Hugging Face blob cache, because these need real filenames on disk.
CAPU_DIR = Path(os.environ.get(
    "AIC_CAPU_DIR", Path.home() / ".cache" / "aic" / "capu"
))

#: Evaluated by ``scripts.benchmarks.bench_audio_capu``; the winner is recorded
#: in ``benchmarks/audio/audio_config.json`` rather than hardcoded here.
CANDIDATES = ("dragonSwing/xlm-roberta-capu", "dragonSwing/vibert-capu")

#: Words per request. The underlying encoders are 512-subword models and
#: Vietnamese averages well over one subword per syllable, so this leaves room
#: for the model's own chunking to stay a no-op.
CHUNK_WORDS = 120

#: Chunks per forward pass. Caps GPU memory on long videos; see `restore_words`.
BATCH_CHUNKS = 16


def _repair_model_output(module) -> None:
    """Re-apply ``@dataclass`` to the model's output type.

    ``Seq2LabelsOutput`` subclasses ``ModelOutput`` but was never decorated.
    Transformers used to tolerate that and now raises at construction time. The
    class already carries the field annotations a dataclass needs, so applying
    the decorator after import is enough.
    """
    import dataclasses

    output = getattr(module, "Seq2LabelsOutput", None)
    if output is not None and not dataclasses.is_dataclass(output):
        module.Seq2LabelsOutput = dataclasses.dataclass(output)


@contextmanager
def _legacy_embedding_resize():
    """Make ``resize_token_embeddings`` behave the way 2022 transformers did.

    ``modeling_seq2labels.py`` grows the embedding matrix by one row inside
    ``from_pretrained``. Current transformers defaults that call to
    ``mean_resizing=True``, which samples from the existing embeddings — and the
    weights are still meta tensors at that point, so it raises
    ``Tensor.item() cannot be called on meta tensors``. The extra row is
    immediately overwritten by the checkpoint, so how it is initialised does not
    affect the result.
    """
    from transformers import PreTrainedModel

    original = PreTrainedModel.resize_token_embeddings

    def patched(self, *args, **kwargs):
        kwargs["mean_resizing"] = False
        return original(self, *args, **kwargs)

    PreTrainedModel.resize_token_embeddings = patched
    try:
        yield
    finally:
        PreTrainedModel.resize_token_embeddings = original


class CapuModel:
    """A loaded restorer. :meth:`restore` takes plain text and returns it punctuated.

    The published entry point is the authors' own ``GecBERTModel`` wrapper,
    which owns the label vocabulary and the sub-word-to-word regrouping.
    Reimplementing that from the weights would be guesswork — the labels are in
    ``vocabulary/``, not in ``config.json`` — so the wrapper is imported from the
    downloaded snapshot and used as documented.
    """

    def __init__(self, model_id: str, device: str | None = None,
                 chunk_size: int = 48, overlap_size: int = 12):
        self.model_id = model_id
        self.device = device or os.environ.get("AUDIO_EMBED_DEVICE", "cuda")
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self._model = None

    def snapshot(self) -> str:
        """Materialise the repo as real files and return the directory.

        The default Hugging Face cache stores every file as a symlink into
        ``blobs/``, where names are content hashes. ``gec_model.py`` resolves
        ``verb-form-vocab.txt`` relative to its own ``__file__``, which follows
        that symlink into ``blobs/`` and finds nothing. Downloading into a plain
        directory keeps the filenames the module expects.
        """
        from huggingface_hub import snapshot_download

        target = CAPU_DIR / self.model_id.replace("/", "--")
        return snapshot_download(self.model_id, local_dir=str(target))

    def load(self) -> "CapuModel":
        import sys

        local = self.snapshot()
        if local not in sys.path:
            sys.path.insert(0, local)
        import modeling_seq2labels

        _repair_model_output(modeling_seq2labels)
        from gec_model import GecBERTModel

        with _legacy_embedding_resize():
            self._model = GecBERTModel(
                vocab_path=os.path.join(local, "vocabulary"),
                model_paths=self.model_id,
                device=self.device,
                # Their own overlap-aware chunker, so a 120-word request is not
                # silently truncated at the encoder's 64-token default.
                split_chunk=True,
                chunk_size=self.chunk_size,
                overlap_size=self.overlap_size,
            )
        return self

    def restore(self, text: str) -> str:
        return self.restore_many([text])[0]

    def restore_many(self, texts) -> list[str]:
        """Restore a batch in one forward pass.

        ``GecBERTModel.forward`` accepts a list and returns one string per
        input, so a whole video goes out as a single request instead of one per
        120-word chunk. The full corpus takes roughly ten minutes on an RTX 3060
        either way — the model re-splits each request into 48-token pieces and
        runs three correction iterations regardless — but batching keeps the
        per-call overhead off the critical path.
        """
        texts = list(texts)
        if not texts:
            return []
        if self._model is None:
            self.load()
        result = self._model(texts)
        if isinstance(result, str):
            return [result]
        return list(result)


def restore_words(words, model, *, chunk_words: int = CHUNK_WORDS,
                  batch_chunks: int = BATCH_CHUNKS) -> list[Word]:
    """Restore punctuation over a whole video, in batched requests.

    Each chunk is accepted only if the model gave back exactly as many words as
    it was given (see :func:`scripts.audio.postprocess.apply_capu_output`), so a
    chunk the model mangles degrades to the raw text instead of corrupting the
    timeline. Rejection is per chunk, not per video.

    ``batch_chunks`` bounds how much goes to the GPU at once. It matters more
    than it looks: the longest video in the 2026 corpus is 8,858 words, and the
    model re-splits every 120-word chunk into 48-token pieces, so an unbounded
    batch would arrive as several hundred sequences at once.
    """
    chunks = [words[start:start + chunk_words]
              for start in range(0, len(words), chunk_words)]
    restored: list[Word] = []
    for start in range(0, len(chunks), batch_chunks):
        batch = chunks[start:start + batch_chunks]
        requests = [" ".join(word.text for word in chunk) for chunk in batch]
        if hasattr(model, "restore_many"):
            texts = model.restore_many(requests)
        else:
            texts = [model.restore(request) for request in requests]
        for chunk, text in zip(batch, texts):
            restored.extend(apply_capu_output(chunk, text))
    return restored
