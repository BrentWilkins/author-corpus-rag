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
validation, an exact SQLite catalog, persistent vector retrieval, inspectable
source-aware search results, citation-constrained grounded answers, repeatable
top-k retrieval evaluation, durable query traces, and initial exact-span
evidence-ledger models. Per-document summaries are treated as experimental
navigation aids. Corpus-wide synthesis is paused until its claims can be audited
against raw source spans.

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

Run the notebook smoke test headlessly without persisting its private outputs:

```bash
uv run python scripts/smoke_notebook.py
```

The smoke runner forces model calls and missing-index builds off even if the
interactive notebook is normally configured to allow them.

## Query boundaries

Use the SQLite catalog for exact counts, complete lists, authorship, and source
inventory. Use semantic retrieval for subject-matter questions where topically
similar passages are useful. Semantic search deliberately reports
`exhaustive=False` and returns its passage text and source URIs so retrieval can
be inspected before an LLM writes an answer. Grounded answers retain the exact
evidence supplied to the model and render only the sources the answer cites as
clickable links. Each generated answer is also saved to the ignored
`.cache/query_traces.sqlite3` database with its retrieved passage text, passage
hashes, document content hashes, model and prompt settings, corpus fingerprint,
citations, and elapsed generation time. A historical trace can therefore be
marked stale when its source documents change.

Keep corpus-specific retrieval cases in the ignored `retrieval-eval.local.yaml`.
Each case maps a natural-language query to one or more known relevant logical
document IDs. Hit rate and mean reciprocal rank provide a small regression test
before changing embedding models, chunking, or retrieval settings.

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
