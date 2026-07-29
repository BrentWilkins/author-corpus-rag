# Author Corpus RAG architecture

This document describes the implemented system boundaries and the invariants
that keep corpus answers auditable. It is source-agnostic: private author names,
publisher details, file paths, and evaluation judgments remain in ignored local
configuration.

The diagrams describe implemented behavior. Future work is called out in prose
rather than silently presented as a current component.

## Level 0: system context

```mermaid
flowchart LR
    researcher[Researcher or editor]
    sources[Markdown, text, PDF,<br/>and source metadata]
    local_model[Local OpenAI-compatible<br/>language model]

    subgraph system[Author Corpus RAG]
        notebook[Jupyter analysis notebook]
        maintenance[Maintenance CLI]
        query_service[Typed query service]
        gradio[Gradio chat +<br/>claim-review UI]
        artifacts[(Local versioned artifacts)]
    end

    sources --> maintenance
    sources --> notebook
    maintenance --> artifacts
    notebook <--> artifacts
    researcher --> notebook
    researcher --> gradio
    gradio --> query_service
    query_service <--> artifacts
    query_service <--> local_model
    notebook <--> local_model
```

The system is local-first. Corpus inputs, generated indexes, model requests,
and query traces do not need to leave the machine. Gradio is an interface over
the same query service; it is not a second retrieval implementation.

## Level 1: ingestion and artifact pipeline

```mermaid
flowchart TB
    config[Ignored corpus.local.yaml]
    files[Markdown / TXT / PDF files]

    subgraph ingestion[Normalize logical works]
        loader[Format-tolerant loaders]
        metadata[Metadata normalization<br/>and provenance]
        validation[Collection validation<br/>warnings and errors]
        documents[CorpusDocument models]
    end

    subgraph exact[Exact metadata path]
        catalog[(SQLite catalog)]
        author_rel[Document-author relations]
        source_rel[One or more source URIs<br/>per logical work]
    end

    subgraph semantic[Semantic evidence path]
        sections[Markdown section boundaries]
        voice[Quotation and voice spans]
        cap[Sentence/token ceiling<br/>and overlap]
        chunks[Source-faithful chunks]
        dense[(Persistent vector index)]
        lexical[(Persistent BM25 index)]
    end

    fingerprint[Corpus + pipeline options<br/>fingerprint]
    manifest[(Artifact manifest)]

    config --> loader
    files --> loader
    loader --> metadata --> validation --> documents
    documents --> catalog
    catalog --> author_rel
    catalog --> source_rel
    documents --> sections --> voice --> cap --> chunks
    chunks --> dense
    chunks --> lexical
    documents --> fingerprint
    fingerprint --> catalog
    fingerprint --> dense
    fingerprint --> lexical
    fingerprint --> manifest
```

`CorpusDocument` represents a logical work, not a file or URL. A work can have
multiple publication locations through `source_uris`, and author credits are a
many-to-many relation. The exact catalog therefore counts logical works once
while retaining all credited authors and manifestations.

The semantic index embeds structural context—title, heading path, and section
lead—but returns untouched source passages as evidence. Voice analysis labels
document-author narration, quoted speech, mixed spans, and uncertainty before
retrieval. It does not claim that prose from a coauthored document belongs to
one individual coauthor.

## Level 2: routed question execution

```mermaid
flowchart TB
    question[Current user question]
    history[Conversation history]
    context[Bounded context resolver]
    identity[Conservative author identity resolver]
    router{Deterministic query router}

    exact_args[Validate author and type<br/>against catalog values]
    exact_query[Execute exhaustive SQL]
    exact_result[Exact result + coverage<br/>+ supporting source records]

    discovery[Dense discovery<br/>one passage per document]
    dense[Dense candidates]
    bm25[BM25 candidates]
    rrf[Reciprocal-rank fusion]
    evidence[One inspected evidence set]
    generator[Optional grounded generation]
    semantic_result[Cited answer or<br/>inspectable passages]

    question --> router
    question --> context
    history --> context
    context --> identity

    router -->|exact_catalog| exact_args --> exact_query --> exact_result
    router -->|broad_discovery| discovery --> evidence
    router -->|focused_evidence| dense --> rrf
    router -->|focused_evidence| bm25 --> rrf
    identity -. resolved semantic query .-> discovery
    identity -. resolved semantic query .-> dense
    identity -. resolved semantic query .-> bm25
    rrf --> evidence
    evidence --> generator --> semantic_result
    evidence -->|generation disabled| semantic_result
```

Routing always uses the literal current question. For semantic follow-ups, the
context resolver may append only the immediately preceding user question to the
retrieval query. It never appends generated assistant text, and it never
rewrites an exact catalog question. The identity resolver then replaces only
the configured default-author name, explicitly trusted aliases, and generic
phrases such as “the author” with a corpus-author retrieval role. It does not
guess nickname relationships. The literal or bounded-context question—not the
retrieval keywords—is passed to grounded generation with only explicitly
resolved author references canonicalized.

The three routes have deliberately different coverage contracts:

| Route | Engine | Coverage claim | Typical use |
| --- | --- | --- | --- |
| `exact_catalog` | SQLite | Exhaustive over ingested metadata | Counts, author lists, coauthors, source inventory |
| `broad_discovery` | Dense retrieval with document diversity | Non-exhaustive | Themes, style exploration, representative works |
| `focused_evidence` | Dense + BM25 + reciprocal-rank fusion | Non-exhaustive | Specific claims, explanations, comparisons, passages |

The query service retrieves semantic evidence exactly once. If generation is
enabled, `GroundedAnswerer` receives that already-inspected result; it does not
perform a second retrieval that could change the evidence or double latency.

## Level 3: one conversational turn

```mermaid
sequenceDiagram
    actor User
    participant UI as Notebook or chat UI
    participant Context as Conversation resolver
    participant Service as CorpusQueryService
    participant Router as Query router
    participant Identity as Author identity resolver
    participant Catalog as SQLite catalog
    participant Search as Retrieval profile
    participant Model as Local model

    User->>UI: Ask a question
    UI->>Context: Current question + UI history
    Context-->>UI: Literal question + optional bounded retrieval query
    UI->>Service: ask(question, retrieval_query)
    Service->>Router: Route literal question

    alt Exact metadata question
        Router-->>Service: exact_catalog + catalog tool
        Service->>Catalog: Validated exact operation
        Catalog-->>Service: Result + exhaustive coverage + sources
        Service-->>UI: ExactCatalogResult
    else Broad or focused semantic question
        Router-->>Service: discovery or focused evidence
        Service->>Identity: Resolve configured references in retrieval query
        Identity-->>Service: Retrieval query + canonicalized grounding question
        Service->>Search: Retrieve once with resolved query
        Search-->>Service: Ranked source passages + provenance
        opt Generation enabled
            Service->>Model: Actual question + same numbered passages
            Model-->>Service: Draft with evidence markers
            Service->>Service: Validate or repair citations
        end
        Service-->>UI: Passages or grounded answer + timings
    end

    UI-->>User: Markdown answer, sources, route, timings
```

Each phase records wall-clock time separately. Dense similarity, BM25 relevance,
and reciprocal-rank-fusion scores retain their score kinds and component ranks;
the system never treats them as a common confidence scale.

## Level 4: provenance and persistence model

```mermaid
erDiagram
    CORPUS_DOCUMENT ||--o{ DOCUMENT_AUTHOR : credits
    CORPUS_DOCUMENT ||--o{ SOURCE_REFERENCE : manifests_at
    CORPUS_DOCUMENT ||--|{ SOURCE_CHUNK : splits_into
    SOURCE_CHUNK ||--|{ VOICE_SPAN : contains
    QUERY_TRACE ||--|{ TRACED_EVIDENCE : records
    CORPUS_DOCUMENT ||--o{ TRACED_EVIDENCE : validated_by_hash
    CORPUS_DOCUMENT ||--o{ EVIDENCE_SPAN : freezes_range
    QUERY_TRACE ||--o{ EVIDENCE_SPAN : records
    TRACED_EVIDENCE }o--o| EVIDENCE_SPAN : resolves_to

    CORPUS_DOCUMENT {
        string document_id PK
        string content_hash
        string title
        string document_type
        string published_at
    }
    DOCUMENT_AUTHOR {
        string document_id FK
        string normalized_author_key
        string display_name
        int ordinal
    }
    SOURCE_REFERENCE {
        string document_id FK
        string uri
        bool is_canonical
        string publisher
        string content_hash
    }
    SOURCE_CHUNK {
        string node_id PK
        string document_id FK
        string section_path
        string source_text
        int source_start_char
        int source_end_char
    }
    VOICE_SPAN {
        string label
        float proportion
        string attributed_speaker
    }
    QUERY_TRACE {
        string trace_id PK
        string corpus_fingerprint
        string model_id
        string prompt_version
        float elapsed_seconds
    }
    TRACED_EVIDENCE {
        string document_id
        string passage_hash
        string document_content_hash
        string evidence_span_id
        int rank
        string score_kind
    }
    EVIDENCE_SPAN {
        string span_id PK
        string document_id FK
        string document_content_hash
        int start_char
        int end_char
        string text_hash
    }
```

Generated answers are evidence views, not durable corpus truth. Query traces
retain the literal user question, contextualized retrieval query, passage
hashes, document hashes, retrieval settings, prompt/model settings, citations,
exact half-open ranges in normalized `CorpusDocument.content`, and elapsed time.
Citation markers resolve through traced evidence to versioned `EvidenceSpan`
records, so validation can detect a missing document, changed document version,
out-of-bounds range, or changed text at that range. These are normalized-corpus
offsets, not byte offsets into original HTML or PDF files. Older trace JSON
remains readable but is marked unversioned when it has no exact span. Both the
notebook and the chat query service persist generated semantic answers through
the same local trace store.

## Level 5: offline claim-classification evaluation

```mermaid
flowchart LR
    public_cases[Committed synthetic<br/>claim/evidence labels]
    private_cases[Ignored private<br/>real-world labels + spans]
    corpus[Current normalized<br/>corpus documents]
    validation[Typed case validation]
    provenance[Exact-span provenance<br/>validation]
    classifier[Conservative heuristic<br/>classifier]
    signals[Inspectable overlap, negation,<br/>markers, and voice signals]
    abstention[Explicit uncertain<br/>abstention]
    metrics[Accuracy, coverage,<br/>selective accuracy,<br/>per-label precision/recall,<br/>false-support rate]
    review[Human inspection<br/>of every mistake]
    ledger[AuditedClaim status]

    public_cases --> validation
    private_cases --> validation
    validation --> provenance
    corpus --> provenance
    provenance --> classifier
    classifier --> signals
    classifier --> abstention
    signals --> metrics
    abstention --> metrics
    metrics --> review
    classifier -. never mutates .-> ledger
```

The classifier labels one claim/evidence pair as `supports`, `qualifies`,
`contradicts`, `updates`, `attributed_report`, `insufficient`, or `uncertain`.
It emits deterministic signals and a rationale, not a confidence score. The
benchmark is an offline regression harness: predictions never promote a claim
to audited truth or enter generation automatically. Committed cases are
synthetic; ignored local cases are required before drawing conclusions about a
private corpus. Provenance-bearing cases must still resolve to the same
versioned source text; stale labels stop notebook evaluation instead of silently
testing a different passage. The committed set deliberately retains a known
entity-role reversal miss, making the current false-support failure measurable.

## Level 6: explicit human claim review

```mermaid
flowchart LR
    decision[Classifier decision]
    spans[Exact evidence spans]
    proposal[Pending review proposal]
    ui[Local Gradio review tab]
    human{Identified reviewer}
    accept[Accept]
    revise[Revise]
    reject[Reject]
    record[Durable review record]
    store[(Local SQLite review log)]
    applicable{Accepted or revised?}
    ledger[New validated<br/>EvidenceLedger]
    blocked[No ledger mutation]

    decision --> proposal
    spans --> proposal
    proposal --> ui --> human
    human --> accept
    human --> revise
    human --> reject
    accept --> record
    revise --> record
    reject --> record
    record --> store
    record --> applicable
    applicable -->|yes| ledger
    applicable -. no .-> blocked
```

A proposal preserves the classifier rationale, exact source spans, and corpus
fingerprint but contains no audited claim. An explicit reviewer action creates a
durable record. Accept preserves the proposal exactly; revise requires a fully
specified `AuditedClaim`; reject records the decision without producing a
claim. Applying an approved record returns a new immutable ledger and refuses
cross-corpus evidence or evidence that was not available in the proposal.
Qualification suggestions require revision because a scope marker alone does
not provide a reviewer-authored qualifier.

## Cache and rebuild boundaries

| Artifact | Persistence | Rebuild trigger | Role |
| --- | --- | --- | --- |
| SQLite catalog | Fingerprinted local cache | Corpus or index options change | Exhaustive metadata facts |
| Vector index | Fingerprinted local cache | Corpus, embedding model, chunking, or pipeline version changes | Dense retrieval |
| BM25 index | Fingerprinted local cache with completion manifest | Vector node set or lexical pipeline changes | Exact-term retrieval arm |
| Document summaries | SQLite cache | Document hash, model, prompt, or summary settings change | Experimental navigation only |
| Query traces | SQLite cache | Append-only per generated answer | Citation-to-span provenance, reproducibility, and staleness checks |
| Claim reviews | SQLite audit log | Append-only per explicit reviewer action | Durable accept, revise, and reject decisions |

Caching improves latency; it does not upgrade generated summaries into source
evidence. Exact queries always read normalized metadata, and grounded answers
always cite retrieved source passages.

## Reliability invariants

1. Exact corpus totals and exhaustive lists come only from SQLite, never top-k
   retrieval or an LLM.
2. Unknown or ambiguous authors and document types stop for clarification; they
   never silently become zero, a default author, or an unfiltered corpus query.
3. A unique first or last name may resolve to one catalog author. A shared alias
   remains ambiguous.
4. Semantic results always report `exhaustive=False`.
5. Generation uses the exact passages already returned for inspection.
6. Factual answer paragraphs require valid numbered evidence markers.
7. Current-index citation markers resolve to exact versioned source ranges;
   historical traces without ranges remain explicitly unversioned.
8. Quoted speech is not attributed to a document author merely because it
   appears in that author's article.
9. Conversation context uses previous user wording only; prior model output is
   never treated as retrieval evidence.
10. Semantic author aliases are explicit private configuration, are checked for
    catalog collisions, and are never inferred with fuzzy matching.
11. Score kinds remain explicit and incomparable across retrieval methods.
12. Heuristic claim classifications never mutate manually audited claim status;
    uncertainty and attributed reports remain distinct from support.
13. Only an explicit identified reviewer can create an applicable review record;
    rejected reviews and cross-corpus proposals cannot update a ledger.
14. Private corpus configuration, evaluation labels, caches, and notebook output
    stay outside version control.

## Component map

| Concern | Primary module |
| --- | --- |
| Input normalization | `src/author_corpus/ingestion.py` |
| Logical document models | `src/author_corpus/models.py` |
| Exact SQLite catalog | `src/author_corpus/catalog.py` |
| Validated exact execution | `src/author_corpus/exact.py` |
| Structure and voice chunking | `src/author_corpus/structure.py`, `voice.py`, `indexing.py` |
| Dense/BM25 profiles and fusion | `src/author_corpus/hybrid.py`, `retrieval.py` |
| Query routing | `src/author_corpus/routing.py` |
| Conservative author identity | `src/author_corpus/identity.py` |
| Single-retrieval orchestration | `src/author_corpus/service.py` |
| Bounded conversation context | `src/author_corpus/conversation.py` |
| Private runtime loading | `src/author_corpus/runtime.py` |
| Local chat and claim-review UI | `src/author_corpus/ui.py` |
| Citation-constrained generation | `src/author_corpus/answering.py` |
| Exact evidence spans and answer audit trail | `src/author_corpus/audit.py`, `tracing.py` |
| Offline claim/evidence classification | `src/author_corpus/claim_classification.py` |
| Explicit human claim review | `src/author_corpus/review.py` |
| Notebook and maintenance entry points | `notebooks/`, `src/author_corpus/cli.py` |
