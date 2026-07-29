# Retrieval and query behavior

Author Corpus RAG does not send every question to a vector index. It routes
questions according to the kind of evidence needed and keeps exact metadata
operations separate from non-exhaustive semantic retrieval.

## Query routes

| Route | Appropriate questions | Execution |
| --- | --- | --- |
| Exact catalog | Counts, complete lists, authorship, coauthors, source inventory | Validated SQLite query; no embedding search or generation |
| Broad discovery | Recurring topics, style patterns, corpus overview, several personal facts | Dense retrieval with document diversity |
| Focused evidence | Specific facts, mechanisms, recommendations, quotations, comparisons, corrections | Dense and BM25 candidates combined with reciprocal-rank fusion |

`route_query` uses inspectable deterministic rules. It reports the matched rule,
literal signals, rationale, and exact catalog tool where applicable. It does
not report an uncalibrated confidence score. An unmatched question takes the
focused-evidence route, the conservative source-grounded default.

Questions such as “how many articles discuss a topic?” cannot be answered
exhaustively from metadata. They use broad discovery and retain a
non-exhaustive warning instead of presenting top-k retrieval as a complete
count.

## Exact catalog behavior

Exact queries validate author names and document types against catalog values,
execute the selected SQLite operation, and skip retrieval and model generation.
Unknown or ambiguous filters request clarification rather than returning a
misleading zero or silently dropping the filter.

Exact results retain every matching logical document and all source URIs as
auditable support. The interface may render a concise source sample while the
complete result remains available to callers.

A single first or last name resolves only when it identifies one catalog
author. Complete catalog names and explicitly configured aliases can resolve;
fuzzy identity guessing is disabled.

## Hard author scopes

Explicit author scopes flow through exact queries, retrieval, reasoning,
synthesis, traces, and reviewed examples.

Author filtering occurs before dense, BM25, and fused ranking, followed by a
defensive credit check on every retained passage. This prevents a dominant
author from crowding a rare author out of the candidate pool and prevents
another author's sole-authored passages from leaking into a scoped result.

A single-author scope retains coauthored documents with their complete credit
sets. A comparison scope requires both retrieved and cited evidence for every
selected author. Missing coverage causes an insufficiency rather than an
invented comparison.

`document_author` means prose attributable to the document's complete listed
author set. It is never treated as proof that one particular coauthor wrote a
specific sentence.

## Retrieval profiles

Broad discovery uses dense retrieval and retains one passage per document to
favor corpus coverage.

Focused evidence gathers a 30-passage candidate pool from dense and BM25
retrieval, combines ranks with reciprocal-rank fusion, retains at most three
passages per document, and returns six final passages by default.

These profiles are not interchangeable:

- cosine similarity describes dense-vector proximity;
- BM25 scores describe lexical relevance;
- reciprocal-rank-fusion values combine rank positions.

The raw values do not share a confidence scale and must not be compared as if
they did. Fused results retain the contributing component ranks and scores for
inspection.

Semantic retrieval explicitly reports `exhaustive=False`. Passage text, source
URIs, document identity, author credits, score semantics, and exact source
ranges remain available before a model writes an answer.

## Structure-aware passages

Markdown headings define natural sections before a sentence splitter applies
the configured token ceiling and overlap. Embeddings receive document title,
heading path, and section lead as context. Retrieved evidence remains the
untouched source chunk rather than a generated proposition or contextual
rewrite.

Every current passage can retain an `EvidenceSpan` containing:

- document ID and content hash;
- exact start and end character offsets;
- exact source text and its hash;
- all source URIs;
- structure and voice metadata.

Current spans can therefore be checked against the normalized document after an
index rebuild. Historical records created before span-aware indexing remain
loadable but are marked unversioned.

## Author voice and quotations

Direct quotation marks and Markdown blockquotes are labeled separately from
document-author narration. Malformed or unmatched delimiters produce an
`uncertain` span instead of being credited to the author. Oversized quotations
retain their label across chunk boundaries.

Explicit nearby names such as `Avery Stone says` may be recorded as speakers.
Pronouns and uncertain attributions remain unnamed. The deterministic layer
does not yet resolve indirect speech, free indirect discourse, transcription
errors, or every publication's pull-quote markup. Those limitations belong in
passage-level evaluation rather than behind an LLM guess.

## Grounded generation and traces

The generator receives the user's question with only trusted author references
canonicalized. Answers may cite only the supplied numbered evidence. Rendered
answers link only the sources they actually cite.

Generation is fail-closed. If the first draft and one repair both omit valid
evidence markers, chat returns a visible grounded abstention. Raw failed drafts
are retained for private diagnosis but are never parsed as claims or shown as
the answer.

Every generated answer is stored in the ignored
`.cache/query_traces.sqlite3` with:

- retrieved passages and exact source spans;
- document, passage, and corpus hashes;
- author scope and retrieval strategy;
- dense, lexical, and fused rank information;
- model and prompt settings;
- citations and generation outcome;
- routing, retrieval, generation, reasoning, and persistence timings.

Human claim and whole-answer decisions live in separate append-only stores.
Generated text never becomes audited truth merely because it was cached.

## Retrieval evaluation

The committed `evaluation/query-routing.yaml` file is a small synthetic
regression baseline. Point `AUTHOR_CORPUS_ROUTING_EVAL` to an ignored local YAML
file to evaluate real user wording. Label cases from the intended coverage
contract, retain misses as regressions, and keep an unseen split before
considering a learned router.

Keep corpus-specific relevance cases in `retrieval-eval.local.yaml`. Each case
maps a query to known relevant logical document IDs and can additionally
require passage text, a heading path, or minimum author-narration or
quoted-speech proportions.

The notebook compares dense, lexical, and hybrid strategies using document and
passage hit rate, mean reciprocal rank, nDCG, and elapsed time. Passage metrics
catch the important failure where retrieval finds the correct article but
returns irrelevant evidence. New embedding models, rerankers, or retrieval
profiles should become defaults only after improving an appropriate holdout.

## Research references

The retrieval design is informed by:

- [Dense X Retrieval](https://arxiv.org/abs/2312.06648)
- [Late Chunking](https://arxiv.org/abs/2409.04701)
- [Document Segmentation Matters for Retrieval-Augmented Generation](https://aclanthology.org/2025.findings-acl.422/)
- [ARES](https://aclanthology.org/2024.naacl-long.20/)
- [Reciprocal Rank Fusion](https://dl.acm.org/doi/10.1145/1571941.1572114)
- [BM25S](https://arxiv.org/abs/2407.03618)
