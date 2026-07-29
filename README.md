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

The current milestones provide normalized ingestion, exact SQLite catalog
queries, persistent dense/BM25 retrieval, citation-constrained answers,
structure and voice provenance, repeatable retrieval and claim evaluation, and
durable exact-span traces. An opt-in bounded reasoning path decomposes hard
questions, retrieves per component, verifies citation-bound claims, and makes
at most one corrective pass. Its default conservative verifier admits only
supported exact-span claims. An experimental structured semantic verifier can
also replace an overbroad draft with an explicit evidence-supported narrower
claim, but remains opt-in pending private holdout results. Neither verifier is
proof.

Explicit author scopes now flow through traces, reasoning, synthesis, and
reviewed training examples while the existing single configured focal author
remains the runtime default. Full multi-author filtering and comparison are
still deferred. Claim and whole-answer reviews are append-only, source-freshness
checked, and separate from generated suggestions. Evidence-bound synthesis
combines resolved ledgers without smoothing contradictions or promoting cached
navigation summaries to fact.

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
# Optional override for private claim/evidence regression cases:
export AUTHOR_CORPUS_CLAIM_EVAL="$PWD/claim-classification.local.yaml"
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

After the index exists, launch the conversational and claim-review interface
locally:

```bash
uv run --extra local --extra ui author-corpus chat --inbrowser
```

The server binds to `127.0.0.1` by default, disables analytics and saved UI
history, and does not request a public share link. Chat history is maintained by
the browser session; the retrieval context layer uses at most the previous user
question and never feeds a generated assistant answer back as evidence. When
`AUTHOR_CORPUS_CLAIM_EVAL` points to provenance-bearing local cases, a separate
review tab shows their exact passages and stable proposals. Every durable action
requires a reviewer name. Accept preserves the proposal, revise exposes the
resolved statement, status, attribution, and qualifiers, and reject creates no
audited claim. Repeated actions on the same proposal are refused. A second,
read-only subtab inspects generated-answer traces: it lists cited candidate
sentences, uncited prose, exact-span coverage, and per-evidence heuristic labels
without creating proposals or changing either audit database.

The chat's collapsed **Reasoning options** panel enables bounded claim-level
reasoning for an individual question. This slower path is off by default. Its
verifier selector defaults to the deterministic conservative baseline; the
structured semantic alternative adds batched local-model calls and is visibly
labeled experimental. Reasoning output reports retrieval, generation, claim
verification, total reasoning, and trace-persistence timings separately. A
separate answer-review tab can accept, revise, or reject a historical answer.
Accept and revise require current exact spans for every citation; ordinary chat
use never counts as approval.

Evaluate private labels tied to generated trace/candidate/evidence IDs:

```bash
uv run author-corpus evaluate-generated-claims \
  --cases generated-claims.local.yaml
```

The ignored YAML identifies a stable trace, extracted candidate, cited evidence
number, and human relationship label:

```yaml
cases:
  - name: example-support-case
    trace_id: replace-with-local-trace-id
    candidate_id: replace-with-local-candidate-id
    evidence_number: 1
    expected_label: supports
    category: factual_support
```

Compare the conservative and semantic aggregate verifiers on identical,
source-current private claims:

```bash
uv run --extra local author-corpus evaluate-verifiers \
  --cases verifier-holdout.local.yaml
```

Each aggregate label applies to a complete generated claim and all of its
citations:

```yaml
cases:
  - name: example-entailment-case
    trace_id: replace-with-local-trace-id
    candidate_id: replace-with-local-candidate-id
    expected_status: supported
    category: direct_entailment
```

The command reports exact-status accuracy, non-abstention coverage, accuracy
among answered cases, false acceptance of claims that should not be answered,
and elapsed time for each verifier. Keep tuned development cases separate from
unseen query and source groups. Do not promote the semantic verifier merely
because it has higher coverage; it must preserve an acceptable false-acceptance
rate on the unseen split.

Export accepted and revised answers together with the frozen evidence shown to
the model:

```bash
uv run author-corpus export-reviewed
```

The default export path is the ignored
`data/private/reviewed-answers.jsonl`.

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

Semantic retrieval can additionally use trusted nicknames or pen names from the
private `AUTHOR_CORPUS_DEFAULT_AUTHOR_ALIASES` JSON array. These aliases are
validated against catalog author credits and rejected when they collide with
another author. The resolver never guesses aliases with fuzzy matching. It
normalizes configured author references to a corpus-author role and can add
generic biographical search terms for personal-profile questions, while the
grounded model receives the user's actual question with only those trusted
references canonicalized. The resolution is inspectable on
`CorpusQueryResult.author_resolution`.

Semantic retrieval deliberately reports `exhaustive=False` and returns its
passage text and source URIs so retrieval can be inspected before an LLM writes
an answer. Grounded answers retain the exact evidence supplied to the model and
render only the sources the answer cites as clickable links. Each generated
answer is also saved to the ignored `.cache/query_traces.sqlite3` database with
its retrieved passage text, exact character ranges in normalized
`CorpusDocument.content`,
passage and document hashes, model and prompt settings, corpus fingerprint,
citations, retrieval strategy, score semantics, component ranks and scores,
and elapsed generation time. Citation markers resolve to versioned
`EvidenceSpan` records rather than only copied passage text. Historical records
created before span-aware indexing still load, but are explicitly reported as
unversioned; current traces can be validated against the exact normalized
source range as well as the document hash.

Human claim decisions and whole-answer decisions are stored separately in the
ignored `.cache/claim_reviews.sqlite3` and `.cache/answer_reviews.sqlite3`
databases. These stores retain corpus fingerprints while remaining available
across index rebuilds. Reviewed behavioral exports include their evidence
context and are not a mechanism for moving corpus facts into model weights.

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

## Bounded reasoning and evidence-bound synthesis

`CorpusQueryService.ask(..., reason=True)` enables the opt-in two-round
reasoning path. Exact catalog queries still bypass retrieval and generation.
The reasoning trace records its explicit `AuthorScope`, retrieval questions,
round timings, generated claims, per-evidence decisions, and claims omitted
from the final answer. Missing exact spans, contradictions, updates, attributed
reports, and unresolved evidence trigger one corrective retrieval pass or an
abstention.

`build_evidence_bound_synthesis` operates only on resolved `EvidenceLedger`
records. It validates every dependent source span, retains claim relationships,
groups claims without rewriting them, and reports documents inspected,
contributing evidence, and unresolved coverage. Immutable results are cached
by corpus fingerprint and author scope in
`.cache/evidence_syntheses.sqlite3`. This does not convert model output into
audited truth automatically.

## Evidence audit direction

The evidence ledger freezes exact character ranges, source hashes, attribution,
qualifiers, and claim relationships such as `supports`, `qualifies`,
`contradicts`, and `updates`. Its source validator detects missing or changed
documents before historical evidence is reused.

The deterministic claim/evidence classifier remains outside the answer-writing
path. It reports lexical overlap, negation mismatch, discourse markers, voice
provenance, a rationale, and one of seven evidence roles. It deliberately
abstains with `uncertain`, treats quotations as `attributed_report`, and exposes
false-support rate alongside accuracy. The committed
`evaluation/claim-classification.yaml` file is a synthetic regression baseline,
not evidence of general semantic-entailment accuracy. Set
`AUTHOR_CORPUS_CLAIM_EVAL` to an ignored local file to measure real corrections
and disagreements without publishing them. Private cases may include an exact
`EvidenceSpan`; the notebook then verifies the document hash, character range,
text, and text hash against the current corpus before reporting metrics. The
baseline intentionally retains an entity-role-reversal case that the lexical
heuristic currently misclassifies, so its false-support limitation remains
visible instead of being tuned out.

Classifier predictions do not update `AuditedClaim.status` automatically.
`ClaimReviewProposal` binds a suggestion to exact evidence and a corpus
fingerprint without promoting it. An identified reviewer must explicitly
accept, revise, or reject it with `review_claim_proposal`; the resulting
`ClaimReviewRecord` can be persisted in `ClaimReviewStore`. Only accepted or
revised records can be passed to `apply_claim_review`, which returns a new
validated `EvidenceLedger`. Rejections remain auditable but cannot mutate a
ledger.

Only the conservative `supports`, `contradicts`, and `insufficient` mappings can
be accepted unchanged. `qualifies` requires a reviewer-authored qualifier;
`updates`, `attributed_report`, and `uncertain` also require a revised claim
because they do not determine a safe final status by themselves. The local
review UI surfaces configured provenance-bearing evaluation cases, and
generated answers have a separate read-only claim preview. Admitting any
extracted candidate into the durable review queue remains future work.
Classifier output is still a heuristic rather than proof.

Private trace-derived labels can be measured separately with
`evaluate-generated-claims`. Keep development examples separate from unseen
query/source groups; production rules should address demonstrated failure
classes rather than individual corpus questions.

Aggregate decisions can be compared with `evaluate-verifiers`. The semantic
verifier runs only when explicitly selected in the UI/service or by that
command. Before changing the default, grow the private set across multiple
authors, document types, source sites, coauthored work, quotations, direct
support, qualifications, contradictions, and missing evidence. This split is
the main defense against fitting thresholds or prompts to one corpus.

## Privacy boundary

`.env`, `*.local.yaml`, `.cache/`, and `data/private/` are ignored. Notebook
outputs are stripped by pre-commit. Keep public examples synthetic and do not
copy a private corpus into this repository.
