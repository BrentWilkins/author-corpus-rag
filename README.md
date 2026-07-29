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

See [the architecture guide](docs/architecture.md) for system-context,
ingestion, routed-query, conversation-sequence, and provenance diagrams.

## Project status

The current milestones provide normalized Markdown, text, and PDF ingestion,
validation, an exact SQLite catalog, persistent dense and BM25 retrieval,
inspectable reciprocal-rank fusion, citation-constrained grounded answers,
repeatable document- and passage-level retrieval evaluation, deterministic
query routing with a labeled benchmark, structure-aware Markdown chunks,
conservative quotation provenance, durable query traces, and initial exact-span
evidence-ledger models. A typed query service now preserves the exact,
discovery, and focused-evidence boundaries across bounded conversational turns,
and an optional local Gradio interface uses that same service. Per-document
summaries are treated as experimental navigation aids. Corpus-wide synthesis is
paused until its claims can be audited against raw source spans.

## Setup

Install Python 3.14 and
[uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv sync --dev --extra local --extra ui
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
GPU memory for generation. The `ui` extra installs the local Gradio chat.

Build the current versioned index outside Jupyter:

```bash
uv run --extra local author-corpus build-index
```

After the index exists, launch the conversational interface locally:

```bash
uv run --extra local --extra ui author-corpus chat --inbrowser
```

The server binds to `127.0.0.1` by default, disables analytics and saved UI
history, and does not request a public share link. Chat history is maintained by
the browser session; the retrieval context layer uses at most the previous user
question and never feeds a generated assistant answer back as evidence.

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
evidence for answer generation. `route_query` makes that decision with
inspectable deterministic rules. It reports a route, matching rule, literal
signals, rationale, and exact catalog tool where applicable; it deliberately
does not report an uncalibrated confidence score. Unmatched questions take the
focused-evidence path, which is the conservative source-grounded default.

Questions such as “how many articles discuss a topic?” cannot be answered
exhaustively by the metadata catalog. The router sends them to broad discovery
and preserves the non-exhaustive warning instead of fabricating a total.
For exact metadata questions, the notebook validates author names and document
types against catalog values, executes the selected SQLite operation, and skips
both vector retrieval and model generation. Unknown or ambiguous filters return
a clarification request rather than a misleading zero or unfiltered result.
Set the private `AUTHOR_CORPUS_DEFAULT_AUTHOR` environment value to resolve
phrases such as “the author”; an explicitly named author always takes
precedence, and an unknown explicit name never falls back to the default.

Exact results retain their matching logical documents and all source URIs as
auditable support. A single first or last name resolves only when it identifies
one catalog author; ambiguous aliases stop for clarification. The notebook
renders a concise source sample while the complete set remains available in
`exact_result.documents`. Exact catalog execution is timed separately from
routing.

Semantic retrieval deliberately reports `exhaustive=False` and returns its
passage text and source URIs so retrieval can be inspected before an LLM writes
an answer. Grounded answers retain the exact evidence supplied to the model and
render only the sources the answer cites as clickable links. Each generated
answer is also saved to the ignored
`.cache/query_traces.sqlite3` database with its retrieved passage text, passage
hashes, document content hashes, model and prompt settings, corpus fingerprint,
citations, retrieval strategy, score semantics, component ranks and scores, and
elapsed generation time. A historical trace can therefore be marked stale when
its source documents change.

Dense cosine similarity, BM25 relevance, and reciprocal-rank-fusion values are
different score types. They must not be compared as if they shared a confidence
scale. Fusion combines ranks rather than adding incomparable raw scores, and
each result preserves the contributing dense and lexical ranks for inspection.

The committed `evaluation/query-routing.yaml` benchmark labels exact catalog,
broad discovery, and focused evidence questions, including the expected exact
tool. The notebook reports overall route accuracy, per-route precision and
recall, exact-tool accuracy, mistakes, and elapsed time. Set
`AUTHOR_CORPUS_ROUTING_EVAL` to an ignored `*.local.yaml` file to evaluate
private, real-world wording. The committed set is a small regression baseline,
not evidence that the rules generalize to every question. Label private cases
from the intended coverage contract rather than copying the router's current
decision, retain misses as development regressions, and keep collecting unseen
questions before considering a learned classifier.

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
