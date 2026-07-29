# Getting started

This guide covers local installation, private corpus configuration, index
building, notebook use, and the Gradio interface. For the system design, see
[Architecture](architecture.md). For query behavior, see
[Retrieval and query behavior](retrieval.md).

## Prerequisites

Install Python 3.14 and
[uv](https://docs.astral.sh/uv/getting-started/installation/), then install the
project with its local-model and UI extras:

```bash
uv sync --dev --extra local --extra ui
```

Register the project environment as a distinct Jupyter kernel:

```bash
uv run python -m ipykernel install --user \
  --name author-corpus-rag \
  --display-name "Python 3.14 (author-corpus-rag)"
```

Using a distinct kernel prevents Jupyter or VS Code from silently selecting a
virtual environment from another project.

## Configure a private corpus

Copy the committed examples:

```bash
cp corpus.example.yaml corpus.local.yaml
cp retrieval-eval.example.yaml retrieval-eval.local.yaml
```

Edit `corpus.local.yaml`, then expose the ignored local files to the runtime:

```bash
export AUTHOR_CORPUS_CONFIG="$PWD/corpus.local.yaml"
export AUTHOR_CORPUS_RETRIEVAL_EVAL="$PWD/retrieval-eval.local.yaml"

# Optional private claim/evidence regression cases:
export AUTHOR_CORPUS_CLAIM_EVAL="$PWD/claim-classification.local.yaml"
```

The normalized model treats a document as a logical work rather than a file.
One document can retain several source references, including a local Markdown
file, its canonical publication URL, and an archived copy.

Set `AUTHOR_CORPUS_DEFAULT_AUTHOR` when phrases such as “the author” should
resolve to one configured person. Optional trusted nicknames or pen names can
be supplied as a JSON array in
`AUTHOR_CORPUS_DEFAULT_AUTHOR_ALIASES`. Aliases are checked against catalog
credits and rejected when they collide with another author; the resolver never
guesses with fuzzy matching.

## Build the indexes

Build the current content-addressed index outside Jupyter:

```bash
uv run --extra local author-corpus build-index
```

The command separately times corpus loading, structure- and voice-aware
chunking, embedding-model loading, embedding/index construction, persistence,
and the complete operation. A change to the corpus, embedding model, chunk
settings, or index-pipeline version creates a new cache fingerprint instead of
overwriting the previous index.

The vector and BM25 indexes contain the same versioned passage nodes. The exact
SQLite catalog is rebuilt from normalized metadata and is used for exhaustive
counts, lists, authorship, and source inventories.

## Use the notebook

Open `notebooks/author_corpus_analyzer.ipynb` and select the
`Python 3.14 (author-corpus-rag)` kernel.

The notebook is safe to run without unexpectedly starting a model request or a
long missing-index build. Opt in before restarting the kernel when those
operations are intended:

```bash
export AUTHOR_CORPUS_ALLOW_INDEX_BUILD=1
export AUTHOR_CORPUS_RUN_GROUNDED_ANSWER=1
```

The `local` dependency extra installs the local LLM and Hugging Face embedding
integrations. PyTorch is installed from its CPU-only wheel index because the
default notebook reserves GPU memory for generation.

Run the notebook headlessly without persisting private outputs:

```bash
uv run python scripts/smoke_notebook.py
```

The smoke runner uses an isolated synthetic corpus and temporary cache, forces
an exact catalog query, and skips the two cells that require an embedding model
and vector index. It executes no model calls and does not read or persist
private corpus output.

## Launch the local interface

After building the index, launch the conversational and review interface:

```bash
uv run --extra local --extra ui author-corpus chat --inbrowser
```

The server binds to `127.0.0.1`, disables analytics and saved UI history, and
does not request a public share link. Browser-session chat history is bounded;
retrieval uses at most the previous user question and never treats a generated
assistant answer as evidence.

The interface provides:

- grounded chat with exact, broad-discovery, and focused-evidence routes;
- one-author, multi-author comparison, and entire-corpus scopes;
- optional bounded claim-level reasoning;
- conservative and experimental semantic verifier choices;
- read-only inspection of generated claims, citations, and failed drafts;
- explicit claim and whole-answer review workflows.

The author selector is a hard retrieval scope. A selected author and a name in
the query must agree. Unknown selections, conflicting names, and incomplete
comparison coverage return a clarification or insufficiency rather than
silently widening the search.

## Build cached navigation summaries

Build or resume per-document summaries:

```bash
uv run --extra local author-corpus build-knowledge
```

The ignored `.cache/knowledge.sqlite3` commits each document summary
immediately. Repeated runs reuse summaries whose document content, model ID,
and prompt version are unchanged. These summaries help navigate the corpus;
they are not authoritative evidence.

An earlier hierarchical corpus synthesis remains available only as an explicit
experiment:

```bash
uv run --extra local author-corpus build-knowledge \
  --include-experimental-synthesis
```

That synthesis may smooth over corrections, qualifications, or disagreements
and must not be treated as a verified corpus conclusion.

## Privacy boundary

`.env`, `*.local.yaml`, `.cache/`, and `data/private/` are ignored. Notebook
outputs are stripped by pre-commit. Keep public examples synthetic and do not
copy a private corpus into the repository.
