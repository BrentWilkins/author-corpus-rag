# Evaluation, reasoning, and review

The project distinguishes generated suggestions, deterministic heuristics,
experimental semantic judgments, and explicit human review. None is promoted
to audited truth merely because it appears plausible.

## Evaluation layers

The committed synthetic cases provide fast, public regression checks:

- `evaluation/query-routing.yaml` checks exact, broad, and focused routing;
- `evaluation/claim-classification.yaml` checks seven claim/evidence roles;
- unit tests cover ingestion, exact catalog behavior, retrieval scopes,
  citations, traces, reasoning, synthesis, and review invariants.

Private evaluations add real questions and exact corpus provenance without
publishing source material:

- routing cases from actual user wording;
- document- and passage-level retrieval judgments;
- generated claim/evidence relationships tied to stable trace IDs;
- aggregate verifier decisions tied to complete claims and citation sets;
- explicit claim and whole-answer review records.

Keep development cases separate from unseen query and source groups. Rules,
prompts, thresholds, and models should address demonstrated failure classes
rather than individual corpus questions.

## Deterministic claim/evidence classification

The conservative classifier reports lexical overlap, negation mismatch,
numbers, discourse markers, voice provenance, a rationale, and one of:

- `supports`
- `qualifies`
- `contradicts`
- `updates`
- `attributed_report`
- `insufficient`
- `uncertain`

It abstains on uncertain voice and treats quotations as attributed reports. The
committed baseline deliberately retains an entity-role-reversal case that the
lexical heuristic misclassifies as support. Its exposed false-support rate is a
safety metric, not an inconvenience to tune away.

Private cases may include an exact `EvidenceSpan`. Before scoring, the notebook
then verifies the document hash, character range, text, and text hash against
the current corpus.

Classifier output remains outside the default answer-writing path and cannot
change an audited claim automatically.

## Evaluate generated claim relationships

Evaluate private labels tied to stable trace, candidate, and evidence IDs:

```bash
uv run author-corpus evaluate-generated-claims \
  --cases generated-claims.local.yaml
```

Example ignored case:

```yaml
cases:
  - name: example-support-case
    trace_id: replace-with-local-trace-id
    candidate_id: replace-with-local-candidate-id
    evidence_number: 1
    expected_label: supports
    category: factual_support
```

The exact passage is recovered from the trace rather than copied into the
label file. Source hashes and ranges must still match the current corpus.

## Compare aggregate verifiers

Compare the conservative and opt-in semantic aggregate verifiers on identical
source-current claims:

```bash
uv run --extra local author-corpus evaluate-verifiers \
  --cases verifier-holdout.local.yaml
```

Each label applies to a complete generated claim and all of its citations:

```yaml
cases:
  - name: example-entailment-case
    trace_id: replace-with-local-trace-id
    candidate_id: replace-with-local-candidate-id
    expected_status: supported
    category: direct_entailment
```

The comparison reports:

- exact-status accuracy;
- non-abstention coverage;
- accuracy among answered cases;
- false acceptance of claims that should not be answered;
- elapsed time for each verifier.

The semantic verifier may replace an overbroad draft with an explicitly
evidence-supported narrower claim. It remains experimental and opt-in. Higher
coverage is not sufficient for promotion; it must preserve an acceptable
false-acceptance rate on unseen authors, source sites, document types,
coauthored work, quotations, corrections, contradictions, and missing
evidence.

## Bounded claim-level reasoning

`CorpusQueryService.ask(..., reason=True)` enables a maximum two-round
reasoning path. Exact catalog questions still bypass retrieval and generation.

The first round decomposes a hard question into focused retrieval components,
generates citation-bound claims, and verifies each claim against its cited
evidence. Missing spans, qualifications, contradictions, updates, attributed
reports, and unresolved evidence can trigger one corrective retrieval pass.
After that pass, unsupported content is omitted or the answer abstains.

The durable trace records:

- the exact `AuthorScope`;
- component retrieval questions and evidence;
- generated and omitted claims;
- per-evidence verifier decisions;
- retrieval, generation, verification, and total timings.

The bounded workflow makes its work inspectable. It does not prove that every
relevant passage was found or that every semantic judgment is correct.

## Explicit claim review

`ClaimReviewProposal` binds a generated suggestion to exact evidence and a
corpus fingerprint without promoting it.

An identified reviewer must explicitly accept, revise, or reject the proposal:

- accept preserves a safely supported proposal;
- revise records reviewer-authored claim text, attribution, status, and
  qualifiers;
- reject remains auditable but cannot mutate a ledger.

Only accepted and revised records can create a new validated
`EvidenceLedger`. Repeated decisions on the same proposal are refused.

Only conservative `supports`, `contradicts`, and `insufficient` mappings can be
accepted unchanged. `qualifies` requires a reviewer-authored qualifier.
`updates`, `attributed_report`, and `uncertain` require a revised claim because
they do not determine a safe final status on their own.

The UI also offers read-only extraction from historical generated answers. It
shows cited candidate sentences, uncited prose, exact-span coverage, and
heuristic evidence labels without creating proposals or altering an audit
database.

## Whole-answer review and export

Whole-answer review is stored separately from claim review. Accepting or
revising an answer requires all citations to resolve to current exact spans;
ordinary chat use never counts as approval.

Export accepted and revised answers with the frozen evidence shown to the
model:

```bash
uv run author-corpus export-reviewed
```

The default ignored output is
`data/private/reviewed-answers.jsonl`. This creates evidence-bearing behavioral
examples for later analysis or optional fine-tuning. It is not a mechanism for
moving corpus facts into model weights.

## Evidence-bound synthesis

The evidence ledger freezes exact character ranges, source hashes,
attribution, qualifiers, and relationships such as `supports`, `qualifies`,
`contradicts`, and `updates`. Its validator detects missing or changed source
documents before old evidence is reused.

`build_evidence_bound_synthesis` accepts only resolved ledgers. It validates
every dependent span, groups claims without rewriting them, retains conflicting
relationships, and reports documents inspected, evidence contributing, and
unresolved coverage.

Immutable results are cached by corpus fingerprint and author scope in
`.cache/evidence_syntheses.sqlite3`. This makes repeated synthesis faster
without turning generated output into truth or smoothing disagreements into a
single narrative.

## Review boundary

The central rule is:

> Retrieval produces evidence. Models produce suggestions. Review produces
> audited decisions.

Cached summaries, generated answers, verifier decisions, and extracted claims
remain distinct artifacts with different authority.
