"""Load heterogeneous files into normalized logical corpus documents."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import yaml

from author_corpus.models import (
    CorpusDocument,
    CorpusLoadResult,
    LoadIssue,
    SourceReference,
)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".pdf", ".txt"}
FRONT_MATTER_BOUNDARY = re.compile(r"^---\s*$")


@dataclass(frozen=True, slots=True)
class CorpusInput:
    """A file or directory plus optional trusted metadata overrides."""

    path: Path
    metadata: dict[str, object] = field(default_factory=dict)


def load_corpus(
    inputs: Iterable[str | Path | CorpusInput],
    *,
    name: str | None = None,
) -> CorpusLoadResult:
    """Load files without requiring every source to provide every field."""
    documents: list[CorpusDocument] = []
    issues: list[LoadIssue] = []

    for item in inputs:
        corpus_input = item if isinstance(item, CorpusInput) else CorpusInput(path=Path(item).expanduser())
        paths = _expand_path(corpus_input.path, issues)
        for path in paths:
            document = _load_file(path, corpus_input.metadata, issues)
            if document is not None:
                documents.append(document)

    _validate_collection(documents, issues)
    return CorpusLoadResult(
        documents=tuple(documents),
        issues=tuple(issues),
        name=name,
    )


def load_corpus_config(path: str | Path) -> CorpusLoadResult:
    """Load a local YAML configuration whose inputs may use relative paths."""
    config_path = Path(path).expanduser().resolve()
    try:
        config_value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return CorpusLoadResult(
            documents=(),
            issues=(
                LoadIssue(
                    severity="error",
                    code="config_unreadable",
                    message=str(exc),
                    path=config_path,
                ),
            ),
        )

    if not isinstance(config_value, Mapping):
        return CorpusLoadResult(
            documents=(),
            issues=(
                LoadIssue(
                    severity="error",
                    code="config_invalid",
                    message="Corpus configuration must be a YAML mapping.",
                    path=config_path,
                ),
            ),
        )

    raw_inputs = config_value.get("inputs")
    if not isinstance(raw_inputs, list):
        return CorpusLoadResult(
            documents=(),
            issues=(
                LoadIssue(
                    severity="error",
                    code="config_inputs_missing",
                    message="Corpus configuration must contain an inputs list.",
                    path=config_path,
                ),
            ),
            name=_optional_string(config_value.get("name")),
        )

    inputs: list[CorpusInput] = []
    config_issues: list[LoadIssue] = []
    for index, raw_input in enumerate(raw_inputs):
        if isinstance(raw_input, str):
            raw_path = raw_input
            metadata: dict[str, object] = {}
        elif isinstance(raw_input, Mapping) and isinstance(raw_input.get("path"), str):
            raw_path = raw_input["path"]
            raw_metadata = raw_input.get("metadata", {})
            if not isinstance(raw_metadata, Mapping):
                config_issues.append(
                    LoadIssue(
                        severity="error",
                        code="config_metadata_invalid",
                        message=f"Input {index} metadata must be a mapping.",
                        path=config_path,
                    )
                )
                continue
            metadata = dict(raw_metadata)
        else:
            config_issues.append(
                LoadIssue(
                    severity="error",
                    code="config_input_invalid",
                    message=f"Input {index} must be a path or mapping with a path.",
                    path=config_path,
                )
            )
            continue

        input_path = Path(raw_path).expanduser()
        if not input_path.is_absolute():
            input_path = config_path.parent / input_path
        inputs.append(CorpusInput(path=input_path, metadata=metadata))

    result = load_corpus(
        inputs,
        name=_optional_string(config_value.get("name")),
    )
    return CorpusLoadResult(
        documents=result.documents,
        issues=tuple(config_issues) + result.issues,
        name=result.name,
    )


def _expand_path(path: Path, issues: list[LoadIssue]) -> list[Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        if _is_sidecar(path) or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            issues.append(
                LoadIssue(
                    severity="warning",
                    code="unsupported_file",
                    message=f"Unsupported input file type: {path.suffix or '<none>'}",
                    path=path,
                )
            )
            return []
        return [path]
    if path.is_dir():
        return sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file()
            and candidate.suffix.lower() in SUPPORTED_SUFFIXES
            and not _is_hidden(candidate.relative_to(path))
            and not _is_sidecar(candidate)
        )

    issues.append(
        LoadIssue(
            severity="error",
            code="input_missing",
            message="Input path does not exist.",
            path=path,
        )
    )
    return []


def _load_file(
    path: Path,
    overrides: Mapping[str, object],
    issues: list[LoadIssue],
) -> CorpusDocument | None:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return _load_markdown(path, overrides, issues)
    if suffix == ".pdf":
        return _load_pdf(path, overrides, issues)
    if suffix == ".txt":
        return _load_text(path, overrides, issues)
    return None


def _load_markdown(
    path: Path,
    overrides: Mapping[str, object],
    issues: list[LoadIssue],
) -> CorpusDocument | None:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        issues.append(_file_error(path, "file_unreadable", exc))
        return None

    body, front_matter, front_matter_error = _parse_front_matter(raw_text)
    if front_matter_error:
        issues.append(
            LoadIssue(
                severity="warning",
                code="front_matter_recovered",
                message=front_matter_error,
                path=path,
            )
        )

    metadata, provenance = _combined_metadata(
        path,
        front_matter,
        overrides,
        issues,
    )
    return _make_document(
        path=path,
        content=body.strip(),
        metadata=metadata,
        provenance=provenance,
        fallback_type="document",
        issues=issues,
    )


def _load_text(
    path: Path,
    overrides: Mapping[str, object],
    issues: list[LoadIssue],
) -> CorpusDocument | None:
    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        issues.append(_file_error(path, "file_unreadable", exc))
        return None

    metadata, provenance = _combined_metadata(path, {}, overrides, issues)
    return _make_document(
        path=path,
        content=content,
        metadata=metadata,
        provenance=provenance,
        fallback_type="document",
        issues=issues,
    )


def _load_pdf(
    path: Path,
    overrides: Mapping[str, object],
    issues: list[LoadIssue],
) -> CorpusDocument | None:
    try:
        from llama_index.core import SimpleDirectoryReader

        pages = SimpleDirectoryReader(
            input_files=[path],
            raise_on_error=True,
        ).load_data()
    except Exception as exc:  # Reader integrations expose several error types.
        issues.append(_file_error(path, "pdf_unreadable", exc))
        return None

    content = "\n\n".join(page.get_content().strip() for page in pages if page.get_content().strip())
    metadata, provenance = _combined_metadata(path, {}, overrides, issues)
    metadata.setdefault("page_count", len(pages))
    provenance.setdefault("page_count", "pdf_reader")
    return _make_document(
        path=path,
        content=content,
        metadata=metadata,
        provenance=provenance,
        fallback_type="pdf",
        issues=issues,
    )


def _parse_front_matter(
    raw_text: str,
) -> tuple[str, dict[str, object], str | None]:
    lines = raw_text.splitlines()
    if not lines or not FRONT_MATTER_BOUNDARY.match(lines[0]):
        return raw_text, {}, None

    end_index = next(
        (index for index, line in enumerate(lines[1:], start=1) if FRONT_MATTER_BOUNDARY.match(line)),
        None,
    )
    if end_index is None:
        return raw_text, {}, "Opening YAML boundary has no closing boundary."

    raw_front_matter = "\n".join(lines[1:end_index])
    body = "\n".join(lines[end_index + 1 :])
    try:
        value = yaml.safe_load(raw_front_matter) or {}
        if not isinstance(value, Mapping):
            raise TypeError("YAML front matter must be a mapping.")
        return body, dict(value), None
    except (TypeError, yaml.YAMLError) as exc:
        recovered = _recover_front_matter(lines[1:end_index])
        return body, recovered, f"Invalid YAML front matter was partially recovered: {exc}"


def _recover_front_matter(lines: list[str]) -> dict[str, object]:
    recovered: dict[str, object] = {}
    for line in lines:
        match = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if not match:
            continue
        key, raw_value = match.groups()
        if key in {"author", "authors"}:
            quoted_names = re.findall(r"""["']([^"']+)["']""", raw_value)
            if quoted_names:
                recovered["authors"] = quoted_names
                continue
        try:
            parsed = yaml.safe_load(f"{key}: {raw_value}")
        except yaml.YAMLError:
            continue
        if isinstance(parsed, Mapping) and key in parsed:
            recovered[key] = parsed[key]
    return recovered


def _combined_metadata(
    path: Path,
    embedded: Mapping[str, object],
    overrides: Mapping[str, object],
    issues: list[LoadIssue],
) -> tuple[dict[str, object], dict[str, str]]:
    metadata = dict(embedded)
    provenance = {key: "yaml_front_matter" for key in embedded}

    sidecar = _load_sidecar(path, issues)
    metadata.update(sidecar)
    provenance.update({key: "sidecar" for key in sidecar})

    metadata.update(overrides)
    provenance.update({key: "corpus_config" for key in overrides})
    return metadata, provenance


def _load_sidecar(path: Path, issues: list[LoadIssue]) -> dict[str, object]:
    candidates = (
        path.with_suffix(path.suffix + ".metadata.yaml"),
        path.with_suffix(".metadata.yaml"),
    )
    sidecar = next((candidate for candidate in candidates if candidate.exists()), None)
    if sidecar is None:
        return {}
    try:
        value = yaml.safe_load(sidecar.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        issues.append(_file_error(sidecar, "sidecar_unreadable", exc))
        return {}
    if not isinstance(value, Mapping):
        issues.append(
            LoadIssue(
                severity="warning",
                code="sidecar_invalid",
                message="Sidecar metadata must be a YAML mapping.",
                path=sidecar,
            )
        )
        return {}
    return dict(value)


def _make_document(
    *,
    path: Path,
    content: str,
    metadata: dict[str, object],
    provenance: dict[str, str],
    fallback_type: str,
    issues: list[LoadIssue],
) -> CorpusDocument | None:
    if not content:
        issues.append(
            LoadIssue(
                severity="error",
                code="content_empty",
                message="No document content was extracted.",
                path=path,
            )
        )
        return None

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    authors = _coerce_authors(metadata)
    if not authors:
        issues.append(
            LoadIssue(
                severity="warning",
                code="authors_missing",
                message="No author metadata was found; semantic indexing can continue.",
                path=path,
            )
        )

    title = _optional_string(metadata.get("title")) or path.stem
    if "title" not in provenance:
        provenance["title"] = "filename"
    sources = _coerce_sources(path, metadata, content_hash)
    identity = (
        _optional_string(metadata.get("document_id"))
        or _optional_string(metadata.get("work_id"))
        or next(
            (source.uri for source in sources if source.is_canonical),
            path.resolve().as_uri(),
        )
    )
    document_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]

    return CorpusDocument(
        document_id=document_id,
        title=title,
        authors=authors,
        content=content,
        content_hash=content_hash,
        sources=sources,
        published_at=_optional_string(metadata.get("published_at", metadata.get("date"))),
        document_type=(_optional_string(metadata.get("document_type")) or fallback_type),
        raw_metadata=metadata,
        metadata_provenance=provenance,
    )


def _coerce_authors(metadata: Mapping[str, object]) -> tuple[str, ...]:
    value = metadata.get("authors", metadata.get("author"))
    raw_authors: list[object]
    if value is None:
        return ()
    if isinstance(value, Mapping):
        raw_authors = [value]
    elif isinstance(value, list | tuple):
        raw_authors = list(value)
    elif isinstance(value, str):
        quoted_names = re.findall(r"""["']([^"']+)["']""", value)
        raw_authors = quoted_names if len(quoted_names) > 1 else re.split(r"\s+(?:and|&)\s+", value)
    else:
        raw_authors = [value]

    names: list[str] = []
    for item in raw_authors:
        if isinstance(item, Mapping):
            item = item.get("name")
        if not isinstance(item, str):
            continue
        name = re.sub(r"^\s*by\s+", "", item, flags=re.IGNORECASE)
        name = re.sub(r"\s+", " ", name).strip(" \t\r\n\"'")
        if name and name not in names:
            names.append(name)
    return tuple(names)


def _coerce_sources(
    path: Path,
    metadata: Mapping[str, object],
    content_hash: str,
) -> tuple[SourceReference, ...]:
    sources: list[SourceReference] = []
    raw_sources = metadata.get("sources", [])
    if isinstance(raw_sources, str | Mapping):
        raw_sources = [raw_sources]
    if isinstance(raw_sources, list | tuple):
        for raw_source in raw_sources:
            source = _source_reference(raw_source)
            if source is not None:
                sources.append(source)

    raw_uris = metadata.get("source_uris", [])
    if isinstance(raw_uris, str):
        raw_uris = [raw_uris]
    if isinstance(raw_uris, list | tuple):
        for uri in raw_uris:
            source = _source_reference(uri)
            if source is not None:
                sources.append(source)

    singular_uri = metadata.get("source_uri", metadata.get("source_url"))
    if isinstance(singular_uri, str):
        source = _source_reference(
            {
                "uri": singular_uri,
                "source_type": "webpage",
                "is_canonical": not any(item.is_canonical for item in sources),
            }
        )
        if source is not None:
            sources.append(source)

    sources.append(
        SourceReference(
            uri=path.resolve().as_uri(),
            source_type=path.suffix.lower().lstrip(".") or "file",
            content_hash=content_hash,
            metadata={"local_path": str(path.resolve())},
        )
    )

    deduplicated: list[SourceReference] = []
    seen: set[str] = set()
    for source in sources:
        if source.uri in seen:
            continue
        seen.add(source.uri)
        deduplicated.append(source)
    return tuple(deduplicated)


def _source_reference(value: object) -> SourceReference | None:
    if isinstance(value, str):
        uri = value.strip()
        if not uri:
            return None
        return SourceReference(
            uri=uri,
            source_type=_source_type_for_uri(uri),
        )
    if not isinstance(value, Mapping):
        return None
    mapping_uri = _optional_string(value.get("uri", value.get("url")))
    if not mapping_uri:
        return None
    known_keys = {
        "uri",
        "url",
        "source_type",
        "is_canonical",
        "publisher",
        "retrieved_at",
        "content_hash",
    }
    return SourceReference(
        uri=mapping_uri,
        source_type=(_optional_string(value.get("source_type")) or _source_type_for_uri(mapping_uri)),
        is_canonical=bool(value.get("is_canonical", False)),
        publisher=_optional_string(value.get("publisher")),
        retrieved_at=_optional_string(value.get("retrieved_at")),
        content_hash=_optional_string(value.get("content_hash")),
        metadata={key: item for key, item in value.items() if key not in known_keys},
    )


def _source_type_for_uri(uri: str) -> str:
    scheme = urlparse(uri).scheme.lower()
    return "webpage" if scheme in {"http", "https"} else "file"


def _validate_collection(
    documents: list[CorpusDocument],
    issues: list[LoadIssue],
) -> None:
    if not documents:
        issues.append(
            LoadIssue(
                severity="error",
                code="corpus_empty",
                message="No supported documents were loaded.",
            )
        )
        return

    ids: dict[str, CorpusDocument] = {}
    hashes: dict[str, CorpusDocument] = {}
    source_owners: dict[str, str] = {}
    for document in documents:
        if document.document_id in ids:
            issues.append(
                LoadIssue(
                    severity="error",
                    code="duplicate_document_id",
                    message=(f"{document.title!r} and {ids[document.document_id].title!r} share an ID."),
                )
            )
        ids[document.document_id] = document

        if document.content_hash in hashes:
            issues.append(
                LoadIssue(
                    severity="warning",
                    code="duplicate_content",
                    message=(f"{document.title!r} duplicates {hashes[document.content_hash].title!r}."),
                )
            )
        hashes[document.content_hash] = document

        for source in document.sources:
            owner = source_owners.get(source.uri)
            if owner is not None and owner != document.document_id:
                issues.append(
                    LoadIssue(
                        severity="warning",
                        code="source_reused",
                        message=f"Source {source.uri!r} belongs to multiple works.",
                    )
                )
            source_owners[source.uri] = document.document_id


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _is_sidecar(path: Path) -> bool:
    return path.name.endswith((".metadata.yaml", ".metadata.yml"))


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _file_error(path: Path, code: str, exc: Exception) -> LoadIssue:
    return LoadIssue(
        severity="error",
        code=code,
        message=str(exc),
        path=path,
    )
