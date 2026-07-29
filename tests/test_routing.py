"""Tests for auditable query routing and its labeled evaluation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from author_corpus.routing import (
    CatalogTool,
    QueryRoute,
    QueryRouteDecision,
    RoutingCase,
    evaluate_query_router,
    load_routing_cases,
    route_query,
)

FIXTURE_PATH = Path(__file__).parents[1] / "evaluation" / "query-routing.yaml"


@pytest.mark.parametrize(
    ("query", "expected_route", "expected_tool"),
    [
        ("How many articles has Avery Stone written here?", QueryRoute.EXACT_CATALOG, CatalogTool.COUNT_DOCUMENTS),
        ("How many articles has the author written here?", QueryRoute.EXACT_CATALOG, CatalogTool.COUNT_DOCUMENTS),
        ("Who are all of those authors?", QueryRoute.EXACT_CATALOG, CatalogTool.LIST_AUTHORS),
        ("What themes appear across the corpus?", QueryRoute.BROAD_DISCOVERY, None),
        ("How does the essay explain adaptation?", QueryRoute.FOCUSED_EVIDENCE, None),
    ],
)
def test_route_query_selects_expected_execution_path(
    query: str,
    expected_route: QueryRoute,
    expected_tool: CatalogTool | None,
) -> None:
    """Choose exact, discovery, or focused work without invoking a model."""
    decision = route_query(query)

    assert decision.route is expected_route
    assert decision.catalog_tool is expected_tool
    assert decision.rule_id
    assert decision.rationale


def test_content_inventory_does_not_claim_catalog_exhaustiveness() -> None:
    """Keep semantic content inventories out of metadata-only exact tools."""
    decision = route_query("How many articles discuss injury prevention?")

    assert decision.route is QueryRoute.BROAD_DISCOVERY
    assert decision.exhaustive is False
    assert decision.catalog_tool is None
    assert decision.rule_id == "semantic.content_inventory"


def test_focused_evidence_is_the_conservative_default() -> None:
    """Send unrecognized questions to citable hybrid evidence by default."""
    decision = route_query("Why might this recommendation change for older athletes?")

    assert decision.route is QueryRoute.FOCUSED_EVIDENCE
    assert decision.rule_id == "semantic.focused_default"
    assert decision.matched_signals == ()


def test_route_query_rejects_blank_input() -> None:
    """Reject a blank question before selecting an execution path."""
    with pytest.raises(ValueError, match="must not be empty"):
        route_query("   ")


def test_route_decision_enforces_exact_coverage_contract() -> None:
    """Prevent semantic routes from claiming exhaustive catalog behavior."""
    with pytest.raises(ValidationError, match="must not claim"):
        QueryRouteDecision(
            query="A question",
            route=QueryRoute.BROAD_DISCOVERY,
            rule_id="invalid",
            rationale="Invalid test decision.",
            catalog_tool=CatalogTool.LIST_DOCUMENTS,
            exhaustive=True,
        )


def test_routing_case_requires_tool_for_exact_label() -> None:
    """Require the exact operation to be labeled, not only the broad route."""
    with pytest.raises(ValidationError, match="require expected_catalog_tool"):
        RoutingCase(name="invalid", query="How many works?", expected_route=QueryRoute.EXACT_CATALOG)


def test_evaluate_router_reports_routes_tools_and_confusion() -> None:
    """Measure the committed generic routing cases before changing defaults."""
    cases = load_routing_cases(FIXTURE_PATH)
    evaluation = evaluate_query_router(cases)

    assert len(cases) == 22
    assert evaluation.accuracy == 1.0
    assert evaluation.exact_tool_accuracy == 1.0
    assert all(metric.precision == 1.0 for metric in evaluation.route_metrics)
    assert all(metric.recall == 1.0 for metric in evaluation.route_metrics)
    assert evaluation.confusion["exact_catalog"]["broad_discovery"] == 0
