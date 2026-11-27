"""Post-validation consolidation; reuse fictional fixtures and mocked OpenAI."""

from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from test_compliance import api, finding, passages, response_data, run, trip
from travel_assistant import compliance as checker


def citation(quote="A fictional policy clause.", page=1, source="fictional.pdf", **extra):
    return {"number": 9, "source": source, "page": page, "quote": quote, **extra}


def row(status="Insufficient information", **changes):
    value = {"requirement": "Hotel nightly rate", "status": status,
             "explanation": "This check needs review.", "citations": [citation()]}
    value.update(changes)
    return value


@pytest.fixture
def business_scenario(trip, passages, response_data):
    exception = "Business class may be used only with verified written approval."
    passages["flight"].append({"text": exception, "source": "fictional.pdf", "page": 3})
    response_data["flight"][0] = finding("cabin", "flight_3", exception, allowed_cabins=["Business"],
                                         status="Insufficient information", approval_dependent=True)
    response_data["flight"].append(finding("approval", "flight_3", exception, approval_dependent=True))
    trip.update(cabin="Business", flight_approval="Claimed approved")
    return trip, passages


def test_repeated_model_evidence_is_displayed_once(trip, passages, response_data, api):
    response_data["flight"][1]["evidence"] *= 3
    response_data["accommodation"][0]["evidence"] *= 5
    result = run(trip, passages)
    for section in checker.SECTIONS:
        assert result[section]["status"] == "Compliant"
        assert all(len(item["citations"]) == 1 and item["citations"][0]["number"] == 1
                   for item in result[section]["findings"])
        assert result[section]["display_findings"] == result[section]["findings"]
    assert api.responses.create.call_count == 1


def test_whitespace_equivalent_quotes_keep_original_validated_span(trip, passages, response_data, api):
    original = passages["accommodation"][0]["text"]
    response_data["accommodation"][0]["evidence"].append({"passage_id": "accommodation_1", "quote": original.replace("nightly rates", "nightly\n   rates")})
    result = run(trip, passages)
    assert result["accommodation"]["findings"][0]["citations"] == [citation(original, page=3, number=1)]


def test_distinct_source_spans_on_same_page_are_preserved(trip, passages, response_data, api):
    first = passages["accommodation"][0]["text"]
    second = "The stated currency is SGD."
    passages["accommodation"][0]["text"] = first + " " + second
    response_data["accommodation"][0]["evidence"].append({"passage_id": "accommodation_1", "quote": second})
    result = run(trip, passages)["accommodation"]["findings"][0]
    assert [item["quote"] for item in result["citations"]] == [first, second]
    assert [item["number"] for item in result["citations"]] == [1, 2]


def test_identical_text_on_distinct_pages_does_not_disappear(trip, passages, response_data, api):
    extra = {**passages["accommodation"][0], "page": 6}
    passages["accommodation"].append(extra)
    response_data["accommodation"][0]["evidence"].append({"passage_id": "accommodation_3", "quote": extra["text"]})
    result = run(trip, passages)["accommodation"]
    assert [item["page"] for item in result["findings"][0]["citations"]] == [3, 6]
    assert result["status"] == "Insufficient information"  # ambiguous coverage is not repaired by presentation


@pytest.mark.parametrize("difference", ["page", "source", "collection", "quote"])
def test_evidence_identity_preserves_material_differences(difference):
    first = citation(collection="policy_v1")
    second = {**first, difference: {"page": 2, "source": "another.pdf", "collection": "policy_v2", "quote": "A different clause."}[difference]}
    assert len(checker._unique_citations([first, second, first], "policy_v1")) == 2


def test_policy_version_participates_in_identity():
    assert checker._evidence_key(citation(), "policy_v1") != checker._evidence_key(citation(), "policy_v2")


def test_duplicate_model_findings_share_one_display_row(trip, passages, response_data, api):
    response_data["accommodation"].append(deepcopy(response_data["accommodation"][0]))
    result = run(trip, passages)["accommodation"]
    assert result["status"] == "Compliant"
    assert len(result["findings"]) == 3  # checked ledger remains complete
    assert [item["requirement"] for item in result["display_findings"]] == ["Hotel nightly rate", "Hotel star rating"]


def test_business_exception_and_ordinary_coverage_share_heading_not_coverage(business_scenario, api):
    trip, passages = business_scenario
    result = run(trip, passages)
    assert result["flight"]["status"] == "Insufficient information"
    assert result["accommodation"]["status"] == "Compliant"
    assert len([item for item in result["flight"]["findings"] if item["requirement"] == "Cabin class"]) == 2
    display = result["flight"]["display_findings"]
    assert [item["requirement"] for item in display] == ["Cabin class", "Airfare"]
    group = display[0]
    assert group["status"] == "Insufficient information"
    assert [item["page"] for item in group["citations"]] == [3, 1]
    assert len(group["parts"]) == 2
    assert any(part.get("coverage_missing") and "not independently assessed" in part["explanation"] for part in group["parts"])
    assert any(part["role"] == "Conditional cabin exception and approval verification" and "self-reported" in part["explanation"] for part in group["parts"])


@pytest.mark.parametrize("quote", [
    "All business travel requires prior written approval.",
    "Flight bookings require prior written approval.",
])
def test_independent_approvals_are_not_absorbed_into_cabin_group(business_scenario, response_data, api, quote):
    trip, passages = business_scenario
    passages["flight"].append({"text": quote, "source": "fictional.pdf", "page": 8})
    response_data["flight"].append(finding("approval", "flight_4", quote))
    result = run(trip, passages)["flight"]
    separate = [item for item in result["display_findings"] if item["requirement"] == "Approval verification"]
    assert separate and any(c["quote"] == quote for item in separate for c in item["citations"])
    assert any(item.get("parts") for item in result["display_findings"])


def test_separate_numeric_obligations_are_not_merged_by_title(trip, passages, response_data, api):
    quote = "Hotel nightly rates must be at least SGD 180."
    passages["accommodation"].append({"text": quote, "source": "fictional.pdf", "page": 3})
    response_data["accommodation"].append(finding("nightly_rate", "accommodation_3", quote, policy_limit="180", policy_currency="SGD", operator="at_least"))
    result = run(trip, passages)["accommodation"]
    assert result["status"] == "Non-compliant"
    rates = [item for item in result["display_findings"] if item["requirement"] == "Hotel nightly rate"]
    assert [item["status"] for item in rates] == ["Compliant", "Non-compliant"]


def test_distinct_bounds_in_one_quote_do_not_merge_generic_abstentions(trip, passages, response_data, api):
    quote = "Hotel nightly rates must be at most SGD 180 and at least SGD 100."
    passages["accommodation"][0]["text"] = quote
    first = response_data["accommodation"][0]
    first.update(status="Insufficient information", explanation="Applicability requires review.",
                 evidence=[{"passage_id": "accommodation_1", "quote": quote}])
    second = deepcopy(first)
    second.update(policy_limit="100", operator="at_least")
    response_data["accommodation"].append(second)
    result = run(trip, passages)["accommodation"]
    assert result["status"] == "Insufficient information"
    rates = [item for item in result["display_findings"] if item["requirement"] == "Hotel nightly rate"]
    assert len(rates) == 2
    assert rates[0]["explanation"] == rates[1]["explanation"]


def test_multiple_approval_obligations_prevent_exception_note_merging(business_scenario, response_data, api):
    trip, passages = business_scenario
    quote = "Business class requires director approval and board approval."
    passages["flight"][2]["text"] = quote
    for item in (response_data["flight"][0], response_data["flight"][2]):
        item["evidence"][0]["quote"] = quote
    result = run(trip, passages)["flight"]
    assert result["status"] == "Insufficient information"
    assert not any(item.get("parts") for item in result["display_findings"])


@pytest.mark.parametrize("statuses", [("Compliant", "Insufficient information"), ("Compliant", "Non-compliant"), ("Non-compliant", "Insufficient information")])
def test_conflicting_statuses_are_preserved(trip, statuses):
    rows = [row(status=status) for status in statuses]
    assert [item["status"] for item in checker._display_findings(rows, trip, "policy_v1")] == list(statuses)


@pytest.mark.parametrize("failure", ["quote", "operator", "unresolved"])
def test_violation_and_distinct_warning_both_remain(trip, passages, response_data, api, failure):
    trip["nightly_rate"] = 200
    extra = deepcopy(response_data["accommodation"][0])
    if failure == "quote":
        extra["evidence"][0]["quote"] = "Invented hotel rate rule."
    elif failure == "operator":
        extra["operator"] = "at_least"
    else:
        extra.update(status="Insufficient information", conditions_resolved=False)
    response_data["accommodation"].append(extra)
    result = run(trip, passages)["accommodation"]
    assert result["status"] == "Non-compliant"
    assert any(item["status"] == "Non-compliant" for item in result["display_findings"])
    assert any(item["status"] == "Insufficient information" for item in result["display_findings"])
    if failure != "unresolved":
        assert any(item.get("validation_error") for item in result["display_findings"])


def test_raw_duplicate_cannot_bypass_passage_validation(trip, passages, response_data, api):
    response_data["accommodation"][0]["evidence"].append({"passage_id": "invented_id", "quote": passages["accommodation"][0]["text"]})
    result = run(trip, passages)["accommodation"]
    assert result["status"] == "Insufficient information"
    assert result["display_findings"][0]["validation_error"]
    assert result["display_findings"][0]["citations"] == []
    assert any(item.get("coverage_missing") for item in result["display_findings"])


@pytest.mark.parametrize("flag", ["validation_error", "coverage_missing", "no_evidence"])
def test_warnings_are_not_removed_as_exact_duplicates(trip, flag):
    item = row(**({flag: True} if flag != "no_evidence" else {"citations": []}))
    assert len(checker._display_findings([item, deepcopy(item)], trip)) == 2


@pytest.mark.parametrize("category", ["ordinary", "exception", "version"])
def test_distinct_cabin_obligations_prevent_speculative_grouping(business_scenario, api, category):
    trip, passages = business_scenario
    ledger = run(trip, passages)["flight"]["findings"]
    if category == "version":
        ledger = deepcopy(ledger)
        ledger[-1]["citations"][0]["collection"] = "another_policy"
    else:
        extra = deepcopy(ledger[-1] if category == "ordinary" else ledger[0])
        extra["citations"][0]["page"] = 9
        ledger = ledger + [extra]
    assert not any(item.get("parts") for item in checker._display_findings(ledger, trip, "policy_v1"))


def test_consolidation_is_idempotent_and_does_not_mutate_ledger(business_scenario, api):
    trip, passages = business_scenario
    ledger = run(trip, passages)["flight"]["findings"]
    saved = deepcopy(ledger)
    first = checker._display_findings(ledger, trip, "policy_v1")
    assert checker._display_findings(first, trip, "policy_v1") == first
    assert checker._display_findings(ledger, trip, "policy_v1") == first
    assert ledger == saved


def test_first_seen_evidence_and_row_order_are_stable(trip):
    first, second = citation("First clause."), citation("Second clause.")
    rows = [row(citations=[first, second]), row(citations=[second, first]), row(explanation="A separate explanation.")]
    result = checker._display_findings(rows, trip, "policy_v1")
    assert len(result) == 2
    assert [item["quote"] for item in result[0]["citations"]] == ["First clause.", "Second clause."]
    assert [item["number"] for item in result[0]["citations"]] == [1, 2]


def test_policy_replacement_has_no_cross_assessment_consolidation(trip, passages, response_data, api, monkeypatch):
    response_data["accommodation"][0]["evidence"] *= 5
    monkeypatch.setattr(checker, "get_active_policy", MagicMock(side_effect=[{"collection": v} for v in ("policy_v1", "policy_v1", "policy_v2", "policy_v2")]))
    retrieval = MagicMock(side_effect=[passages["flight"], passages["accommodation"]] * 2)
    monkeypatch.setattr(checker, "retrieve_policy", retrieval)
    unique = MagicMock(wraps=checker._unique_citations)
    monkeypatch.setattr(checker, "_unique_citations", unique)
    first = checker.assess_trip(trip, "mock-key", expected_collection="policy_v1")
    saved = deepcopy(first)
    second = checker.assess_trip(trip, "mock-key", expected_collection="policy_v2")
    assert first == saved and first["collection"] != second["collection"]
    assert {call.args[1] for call in unique.call_args_list} == {"policy_v1", "policy_v2"}
    assert len(second["accommodation"]["display_findings"][0]["citations"]) == 1
    assert [call.kwargs["expected_collection"] for call in retrieval.call_args_list] == ["policy_v1", "policy_v1", "policy_v2", "policy_v2"]


def test_streamlit_uses_consolidated_view_and_keeps_coverage_visible(business_scenario, api, monkeypatch):
    from test_compliance_page import setup_page, widget
    from travel_assistant import compliance_page

    trip, passages = business_scenario
    result = {"collection": "policy_v1", **run(trip, passages)}
    monkeypatch.setattr(compliance_page, "assess_trip", MagicMock(return_value=result))
    app = setup_page(monkeypatch)
    widget(app, "text_input", "Origin").set_value("Fictional A")
    widget(app, "text_input", "Destination").set_value("Fictional B").run()
    widget(app, "button", "Check compliance").click().run()
    assert not app.exception
    assert len([item for item in app.markdown if item.value == "Cabin class"]) == 1
    assert any("coverage incomplete" in item.value for item in app.text)
    assert any("self-reported" in item.value for item in app.text)
    assert len([item for item in app.text if "fictional.pdf — page 3" in item.value]) == 2  # grouped cabin and nightly-rate are independent


@pytest.fixture
def overlapping_cabin_scenario(business_scenario, response_data):
    """Live E.3 shape: one exception finding also cites its separate scope note."""
    trip, passages = business_scenario
    note = "This exception changes only the cabin rule."
    passages["flight"][2]["text"] += " " + note
    response_data["flight"][0]["evidence"].append({"passage_id": "flight_3", "quote": note})
    general = "All business travel requires prior written approval before booking."
    for section in checker.SECTIONS:
        passages[section].append({"text": general, "source": "fictional.pdf", "page": 4})
        response_data[section].append(finding("approval", f"{section}_{len(passages[section])}", general))
    trip.update(accommodation_approval="Claimed approved",
                context="Fictional approver Alex Sample confirmed written approval on 2026-11-27.")
    return trip, passages


def test_live_e3_overlapping_cabin_findings_share_one_group(overlapping_cabin_scenario, response_data, api):
    import jsonschema

    trip, passages = overlapping_cabin_scenario
    result = run(trip, passages)
    jsonschema.validate(response_data, api.responses.create.call_args.kwargs["text"]["format"]["schema"])
    assert result["flight"]["status"] == result["accommodation"]["status"] == "Insufficient information"
    rows = result["flight"]["display_findings"]
    assert [item["requirement"] for item in rows] == ["Cabin class", "Airfare", "Approval verification"]
    group = rows[0]
    assert group["parts"] and group["status"] == "Insufficient information"
    assert [item["quote"] for item in group["citations"]] == [
        "Business class may be used only with verified written approval.",
        "This exception changes only the cabin rule.",
        "Flights must use Economy class.",
    ]
    assert [item["number"] for item in group["citations"]] == [1, 2, 3]
    assert any(part.get("coverage_missing") and "not independently assessed" in part["explanation"] for part in group["parts"])
    assert any("self-reported" in part["explanation"] for part in group["parts"])
    assert rows[-1]["citations"][0]["quote"].startswith("All business travel requires")
    assert all(not item.get("validation_error") for item in result["flight"]["findings"])
    assert api.responses.create.call_count == 1


@pytest.mark.parametrize("reverse", [False, True])
def test_partial_overlap_and_citation_order_preserve_quote_union(overlapping_cabin_scenario, response_data, api, reverse):
    trip, passages = overlapping_cabin_scenario
    if reverse:
        response_data["flight"][0]["evidence"].reverse()
    group = run(trip, passages)["flight"]["display_findings"][0]
    assert group.get("parts")
    assert len(group["citations"]) == 3
    assert {item["quote"] for item in group["citations"]} == {
        "Business class may be used only with verified written approval.",
        "This exception changes only the cabin rule.", "Flights must use Economy class."}


def test_duplicate_scope_note_is_validated_then_deduplicated(overlapping_cabin_scenario, response_data, api):
    trip, passages = overlapping_cabin_scenario
    response_data["flight"][0]["evidence"].append({
        "passage_id": "flight_3", "quote": "This exception\n  changes only the cabin rule."})
    result = run(trip, passages)
    assert len(result["flight"]["findings"][0]["citations"]) == 2
    assert len(result["flight"]["display_findings"][0]["citations"]) == 3


def test_invalid_scope_quotation_remains_a_warning_not_verified_evidence(overlapping_cabin_scenario, response_data, api):
    trip, passages = overlapping_cabin_scenario
    invalid = "This exception changes every policy rule."
    response_data["flight"][0]["evidence"][1]["quote"] = invalid
    result = run(trip, passages)
    display = result["flight"]["display_findings"]
    assert result["flight"]["status"] == "Insufficient information"
    assert any(item.get("validation_error") and not item["citations"] for item in display)
    assert all(item["quote"] != invalid for row in display for item in row["citations"])
    assert any(part.get("coverage_missing") for row in display for part in row.get("parts", []))


def test_scope_linkage_does_not_change_ledger_coverage_or_status(overlapping_cabin_scenario, api, monkeypatch):
    trip, passages = overlapping_cabin_scenario
    consolidated = run(trip, passages)
    monkeypatch.setattr(checker, "_display_findings", lambda ledger, *args: ledger)
    ungrouped = run(trip, passages)
    for section in checker.SECTIONS:
        assert consolidated[section]["findings"] == ungrouped[section]["findings"]
        assert consolidated[section]["status"] == ungrouped[section]["status"]
    assert any(item.get("coverage_missing") and item["requirement"] == "Cabin class" for item in consolidated["flight"]["findings"])


@pytest.mark.parametrize("difference", ["not_adjacent", "different_page", "different_version", "no_context"])
def test_unproven_scope_relationship_remains_separate(overlapping_cabin_scenario, api, difference):
    trip, passages = overlapping_cabin_scenario
    ledger = deepcopy(run(trip, passages)["flight"]["findings"])
    context = deepcopy([p for rows in passages.values() for p in rows])
    if difference == "not_adjacent":
        for passage in context:
            passage["text"] = passage["text"].replace("approval. This exception", "approval. Another independent provision intervenes. This exception")
    elif difference == "different_page":
        ledger[0]["citations"][1]["page"] = 9
    elif difference == "different_version":
        ledger[0]["citations"][1]["collection"] = "policy_other"
    else:
        context = None
    display = checker._display_findings(ledger, trip, "policy_v1", context)
    assert len([item for item in display if item["requirement"] == "Cabin class"]) == 2
    assert any(item["citations"] and item["citations"][0]["quote"].startswith("All business travel") for item in display)


@pytest.mark.parametrize("note", [
    "This cabin exception also changes hotel limits.",
    "This exception changes only the cabin rule unless another arrangement applies.",
    "This cabin exception requires another director approval.",
])
def test_ambiguous_or_independent_extra_clause_is_not_absorbed(overlapping_cabin_scenario, response_data, api, note):
    trip, passages = overlapping_cabin_scenario
    passages["flight"][2]["text"] = "Business class may be used only with verified written approval. " + note
    response_data["flight"][0]["evidence"][1]["quote"] = note
    result = run(trip, passages)
    rows = result["flight"]["display_findings"]
    assert any(not item.get("parts") and item["requirement"] == "Cabin class"
               and any(c["quote"] == note for c in item["citations"]) for item in rows)
    assert result["flight"]["status"] == "Insufficient information"


@pytest.mark.parametrize("status", ["Compliant", "Non-compliant"])
def test_materially_conflicting_cabin_record_is_not_hidden(overlapping_cabin_scenario, api, status):
    trip, passages = overlapping_cabin_scenario
    ledger = deepcopy(run(trip, passages)["flight"]["findings"])
    conflicting = {**deepcopy(ledger[0]), "status": status}
    display = checker._display_findings(ledger + [conflicting], trip, "policy_v1", [p for rows in passages.values() for p in rows])
    assert any(item.get("parts") for item in display)
    assert any(item["status"] == status and item["requirement"] == "Cabin class" and not item.get("parts") for item in display)


@pytest.mark.parametrize("field,value,flight_status,hotel_status", [
    ("airfare", 1000, "Non-compliant", "Insufficient information"),
    ("nightly_rate", 200, "Insufficient information", "Non-compliant"),
])
def test_numeric_violation_precedence_is_independent_of_scope_group(overlapping_cabin_scenario, api, field, value, flight_status, hotel_status):
    trip, passages = overlapping_cabin_scenario
    trip[field] = value
    result = run(trip, passages)
    assert result["flight"]["status"] == flight_status
    assert result["accommodation"]["status"] == hotel_status
    assert result["flight"]["display_findings"][0]["status"] == "Insufficient information"


def test_scope_quote_does_not_change_accommodation(overlapping_cabin_scenario, response_data, api):
    trip, passages = overlapping_cabin_scenario
    with_scope = run(trip, passages)
    response_data["flight"][0]["evidence"].pop()
    without_scope = run(trip, passages)
    assert with_scope["accommodation"] == without_scope["accommodation"]


def test_overlapping_scope_consolidation_is_idempotent(overlapping_cabin_scenario, api):
    trip, passages = overlapping_cabin_scenario
    ledger = run(trip, passages)["flight"]["findings"]
    saved = deepcopy(ledger)
    context = [p for rows in passages.values() for p in rows]
    first = checker._display_findings(ledger, trip, "policy_v1", context)
    assert checker._display_findings(first, trip, "policy_v1", context) == first
    assert checker._display_findings(ledger, trip, "policy_v1", context) == first
    assert ledger == saved


def test_streamlit_renders_one_overlapping_cabin_group(overlapping_cabin_scenario, api, monkeypatch):
    from test_compliance_page import setup_page, widget
    from travel_assistant import compliance_page

    trip, passages = overlapping_cabin_scenario
    result = {"collection": "policy_v1", **run(trip, passages)}
    monkeypatch.setattr(compliance_page, "assess_trip", MagicMock(return_value=result))
    app = setup_page(monkeypatch)
    widget(app, "text_input", "Origin").set_value("Singapore")
    widget(app, "text_input", "Destination").set_value("Japan").run()
    widget(app, "button", "Check compliance").click().run()
    assert not app.exception
    assert len([item for item in app.markdown if item.value == "Cabin class"]) == 1
    assert len([item for item in app.text if item.value == "This exception changes only the cabin rule."]) == 1
    assert len([item for item in app.text if item.value.startswith("All business travel requires prior written approval")]) == 2
    assert any("coverage incomplete" in item.value for item in app.text)
    assert any("self-reported" in item.value for item in app.text)
