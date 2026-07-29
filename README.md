# Author Corpus RAG

Author Corpus RAG is a source-agnostic experiment for exploring a body of
written work with exact metadata queries, semantic retrieval, cached summaries,
and conversational question answering.

The project deliberately separates private corpus configuration and data from
committed code:

- committed examples and tests use synthetic authors and documents;
- local paths and corpus-specific metadata live in an ignored
  `corpus.local.yaml`;
- source documents and generated indexes remain outside version control;
- committed notebooks contain no execution output.

## Project status

The current milestones provide normalized Markdown, text, and PDF ingestion,
validation, an exact SQLite catalog, persistent dense and BM25 retrieval,
inspectable reciprocal-rank fusion, citation-constrained grounded answers,
repeatable document- and passage-level retrieval evaluation, structure-aware
Markdown chunks, conservative quotation provenance, durable query traces, and
initial exact-span evidence-ledger models. Per-document summaries are treated
as experimental navigation aids. Corpus-wide synthesis is paused until its
claims can be audited against raw source spans.

## Setup

Install Python 3.14 and
[uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync --dev --extra local
uv run python -m ipykernel install --user \
  --name author-corpus-rag \
  --display-name "Python 3.14 (author-corpus-rag)"
cp corpus.example.yaml corpus.local.yaml
cp retrieval-eval.example.yaml retrieval-eval.local.yaml
```

Edit the ignored `corpus.local.yaml`, then point the notebook at it:

```bash
export AUTHOR_CORPUS_CONFIG="$PWD/corpus.local.yaml"
export AUTHOR_CORPUS_RETRIEVAL_EVAL="$PWD/retrieval-eval.local.yaml"
```

Open `notebooks/author_corpus_analyzer.ipynb` and select the
`Python 3.14 (author-corpus-rag)` kernel. Registering a distinct kernel prevents
Jupyter or VS Code from silently using a virtual environment from another
project.

The notebook is safe to run without starting an unexpected model request or
long index build. To opt in to either operation, set the corresponding ignored
local environment value before restarting the kernel:

```bash
AUTHOR_CORPUS_ALLOW_INDEX_BUILD=1
AUTHOR_CORPUS_RUN_GROUNDED_ANSWER=1
```

The `local` extra installs local LLM and Hugging Face embedding integrations.
PyTorch comes from its CPU-only wheel index because the default notebook reserves
GPU memory for generation. Install the `ui` extra when the Gradio milestone is
added.

Build the current versioned index outside Jupyter:

```bash
uv run --extra local author-corpus build-index
```

The command separately times corpus loading, structure/voice-aware chunking,
embedding-model loading, embedding/index construction, persistence, and the
complete operation. A changed corpus, embedding model, chunk setting, or
index-pipeline version produces a new cache fingerprint instead of overwriting a
previous index.

The notebook builds or reloads a small BM25 index over the exact persisted
vector-index nodes. Broad discovery uses dense retrieval and retains one passage
per document. Focused evidence gathering uses a 30-passage dense/BM25 candidate
pool, reciprocal-rank fusion, up to three passages per document, and six final
passages. These are explicit profiles rather than interchangeable score
thresholds.

Run the notebook smoke test headlessly without persisting its private outputs:

```bash
uv run python scripts/smoke_notebook.py
```

The smoke runner forces model calls and missing-index builds off even if the
interactive notebook is normally configured to allow them.

## Query boundaries

Use the SQLite catalog for exact counts, complete lists, authorship, and source
inventory. Use dense discovery for broad subject exploration and hybrid focused
evidence for answer generation. Retrieval deliberately reports
`exhaustive=False` and returns its passage text and source URIs so retrieval can
be inspected before an LLM writes an answer. Grounded answers retain the exact
evidence supplied to the model and render only the sources the answer cites as
clickable links. Each generated answer is also saved to the ignored
`.cache/query_traces.sqlite3` database with its retrieved passage text, passage
hashes, document content hashes, model and prompt settings, corpus fingerprint,
citations, retrieval strategy, score semantics, component ranks and scores, and
elapsed generation time. A historical trace can therefore be marked stale when
its source documents change.

Dense cosine similarity, BM25 relevance, and reciprocal-rank-fusion values are
different score types. They must not be compared as if they shared a confidence
scale. Fusion combines ranks rather than adding incomparable raw scores, and
each result preserves the contributing dense and lexical ranks for inspection.

Keep corpus-specific retrieval cases in the ignored `retrieval-eval.local.yaml`.
Each case maps a natural-language query to one or more known relevant logical
document IDs. A case can additionally identify required passage text, heading
path, and minimum document-author or quoted-speech proportion. Document hit rate
and mean reciprocal rank remain useful discovery metrics; passage hit rate
detects the more important failure where the right article returns the wrong
evidence. The notebook compares dense, lexical, and hybrid strategies on the
same cases and reports hit rate, mean reciprocal rank, nDCG, and elapsed time at
both document and passage levels. New rerankers or models should become defaults
only after improving this local benchmark.

## Structure and voice provenance

Markdown headings define natural sections before the sentence splitter applies
the configured token ceiling and overlap. Embeddings receive the document title,
heading path, and section lead as context. Retrieved evidence remains the
untouched chunk text rather than a generated proposition or contextual rewrite.

Direct quotation marks and Markdown blockquotes are labeled separately from
document-author narration. Malformed or unmatched quote delimiters produce an
`uncertain` span rather than being silently credited to the author. Oversized
quotations retain their label when they cross chunk boundaries. Explicit nearby
names such as `Avery Stone says` may be recorded as speakers; pronouns and
uncertain attributions remain unnamed.
`document_author` means prose attributable to the document's listed author set.
It must not be interpreted as one particular person's voice for coauthored
documents.

This deterministic layer is intentionally conservative. It does not yet resolve
indirect speech, free indirect discourse, transcription errors, or every
publication's pull-quote markup. Those limitations belong in passage-level
evaluation rather than being hidden behind an LLM guess.

The design follows findings that document segmentation and retrieval granularity
materially affect RAG quality, while preserving raw evidence for attribution:

- [Dense X Retrieval](https://arxiv.org/abs/2312.06648)
- [Late Chunking](https://arxiv.org/abs/2409.04701)
- [Document Segmentation Matters for Retrieval-Augmented Generation](https://aclanthology.org/2025.findings-acl.422/)
- [ARES](https://aclanthology.org/2024.naacl-long.20/)
- [Reciprocal Rank Fusion](https://dl.acm.org/doi/10.1145/1571941.1572114)
- [BM25S](https://arxiv.org/abs/2407.03618)

## Cached navigation summaries

Build or resume per-document summaries outside Jupyter:

```bash
uv run --extra local author-corpus build-knowledge
```

The ignored `.cache/knowledge.sqlite3` commits each document summary
immediately. Repeating the command reuses summaries whose document content,
model ID, and prompt version are unchanged. These summaries help navigate the
corpus but are not authoritative evidence.

The earlier hierarchical corpus synthesis can still be run for an explicit
experiment:

```bash
uv run --extra local author-corpus build-knowledge \
  --include-experimental-synthesis
```

That synthesis may smooth over corrections, qualifications, or disagreements,
so it must not be treated as a verified corpus conclusion.

## Evidence audit direction

The evidence ledger freezes exact character ranges, source hashes, attribution,
qualifiers, and claim relationships such as `supports`, `qualifies`,
`contradicts`, and `updates`. Its source validator detects missing or changed
documents before historical evidence is reused.

The current milestone supplies the deterministic storage and validation layer;
automatic LLM claim extraction and entailment classification remain evaluation
work. A model's classification will be treated as a heuristic, not proof.

## Privacy boundary

`.env`, `*.local.yaml`, `.cache/`, and `data/private/` are ignored. Notebook
outputs are stripped by pre-commit. Keep public examples synthetic and do not
copy a private corpus into this repository.
