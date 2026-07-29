# Public multi-author demo

## Goal

Provide a reproducible, privacy-safe demonstration of exact catalog queries,
hard author-scoped retrieval, coauthor attribution, and source inspection.
Private corpora, local paths, query traces, model endpoints, and reviewed
answers must never enter the demo build.

## Corpus policy

Use a small pinned corpus with at least two authors and multiple works per
author. Favor writers whose works are unambiguously old enough to be public
domain in the United States and other common life-plus-70 jurisdictions.
Project Gutenberg is a reasonable source, but every selected ebook must be
checked individually for its embedded rights statement.

The committed demo should contain normalized source text plus a manifest with:

- canonical author and title;
- Project Gutenberg ebook ID and source URL;
- original publication year;
- retrieval date and SHA-256 hash;
- the ebook's rights statement;
- normalization steps and the normalized-content hash.

Do not scrape `gutenberg.org` from CI. Project Gutenberg reserves its main site
for human use and directs automation to its robot harvest, mirrors, and offline
catalog feeds. Prefer a one-time, explicit maintainer import followed by
committed, checksum-pinned demo inputs. CI should verify those inputs rather
than silently refreshing them.

Project Gutenberg's permissions guidance says that most ebooks are unrestricted
by United States copyright law, but its name and license are separate trademark
materials and non-US users must check local law. Keep source acknowledgements
factual, avoid presenting the demo as endorsed by Project Gutenberg, and retain
the provenance manifest.

Official references:

- <https://www.gutenberg.org/policy/permission>
- <https://www.gutenberg.org/policy/license>
- <https://www.gutenberg.org/policy/robot_access.html>

## GitHub Actions

The repository CI workflow can validate Python 3.14, the locked `uv`
environment, formatting, linting, strict typing, tests, and the output-free
notebook smoke run without private configuration.

A later Pages workflow should:

1. validate the public corpus manifest and source hashes;
2. ingest only `data/public/`;
3. build a deterministic static catalog and passage bundle;
4. run synthetic and public multi-author regression cases;
5. build the browser application;
6. upload the static artifact with `actions/upload-pages-artifact`;
7. deploy that saved artifact with `actions/deploy-pages`.

The workflow should use read-only repository permissions during the build and
grant Pages deployment permissions only to the deployment job.

## What GitHub Pages can host

GitHub Pages can host a useful static evidence explorer:

- exact author, work, source, and coauthor filters;
- hard author selection and comparison controls;
- client-side lexical passage search;
- precomputed, inspectable retrieval examples;
- source links, author credits, and coverage warnings;
- architecture and evaluation reports.

Pages cannot run this project's Python process, SQLite service, Gradio server,
embedding model, or local OpenAI-compatible model endpoint. A browser-only
semantic experiment could ship precomputed passage embeddings and a JavaScript
query model, but that increases download size and creates a second runtime to
evaluate. It should not be the first public demo.

The first Pages release should therefore demonstrate catalog and retrieval
behavior without dynamic LLM answers. A full live RAG demo would require a
separate backend host; the static site can link to it later.

## Proposed phases

1. Add a pinned two-author public corpus and provenance manifest.
2. Add deterministic public-corpus ingestion and regression tests.
3. Export static catalog and lexical-search assets.
4. Build and locally test the static evidence explorer.
5. Add a Pages deployment workflow after a Git remote and Pages environment
   exist.
6. Consider an optional hosted full-RAG backend only after the static demo and
   its evaluation reports are stable.
