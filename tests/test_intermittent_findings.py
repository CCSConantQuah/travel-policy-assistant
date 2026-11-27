"""Step 4.19F: schema-valid intermittent findings with all OpenAI calls mocked."""

from copy import deepcopy

import pytest

from test_compliance import api, cabin_rule, finding, passages, response_data, run, trip
from travel_assistant import compliance as checker

SUPPORT = "Ordinary Economy bookings within limits need no cabin exception."
CONTEXT = "These rules apply to every destination in this fictional policy."
GENERAL = "All business travel requires prior written approval before booking."


@pytest.fixture
def ordinary_scenario(trip, passages, response_data):
    passages["flight"][0]["text"] += " " + CONTEXT
    passages["flight"].append({"text": "Business class requires verified written approval as an exception. " + SUPPORT,
                               "source": "fictional.pdf", "page": 3})
    response_data["flight"][0]["evidence"].append({"passage_id": "flight_3", "quote": SUPPORT})
    for section in checker.SECTIONS:
        passages[section].append({"text": GENERAL, "source": "fictional.pdf", "page": 5})
        response_data[section].append(finding("approval", f"{section}_{len(passages[section])}", GENERAL))
        response_data[section].append(finding("approval", "", "", evidence=[], status="Insufficient information"))
    response_data["flight"].append(finding("other", "flight_1", CONTEXT, status="Insufficient information"))
    trip.update(origin="Singapore", destination="Japan", hotel="Standard Hotel", context="")
    return trip, passages


def test_all_three_live_issues_are_fixed_with_schema_valid_response(ordinary_scenario, response_data, api):
    import jsonschema

    trip, passages = ordinary_scenario
    result = run(trip, passages)
    jsonschema.validate(response_data, api.responses.create.call_args.kwargs["text"]["format"]["schema"])
    for section in checker.SECTIONS:
        assert result[section]["status"] == "Insufficient information"
        rows = result[section]["display_findings"]
        assert [item["status"] for item in rows] == ["Compliant", "Compliant", "Insufficient information"]
        approvals = [item for item in rows if item["requirement"] == "Approval verification"]
        assert len(approvals) == 1 and approvals[0]["citations"][0]["quote"] == GENERAL
        # The redundant empty check is retained in the ledger, not rewritten as verified.
        assert len([item for item in result[section]["findings"] if item["requirement"] == "Approval verification"]) == 2
    cabin = result["flight"]["findings"][0]
    assert cabin["status"] == "Compliant" and not cabin.get("validation_error")
    assert SUPPORT in [item["quote"] for item in cabin["citations"]]
    context = next(item for item in result["flight"]["findings"] if item.get("context_only"))
    assert context["citations"][0]["quote"] == CONTEXT
    assert CONTEXT in [item["quote"] for item in result["flight"]["display_findings"][0]["citations"]]
    assert not any(item.get("coverage_missing") for item in result["flight"]["findings"])
    assert api.responses.create.call_count == 1


@pytest.mark.parametrize("all_limits,reverse", [(False, False), (True, False), (True, True)])
def test_corroborating_cabin_note_is_order_independent(ordinary_scenario, response_data, api, all_limits, reverse):
    trip, passages = ordinary_scenario
    if all_limits:
        quote = SUPPORT.replace("within limits", "within all limits")
        passages["flight"][2]["text"] = passages["flight"][2]["text"].replace(SUPPORT, quote)
        response_data["flight"][0]["evidence"][1]["quote"] = quote
    if reverse:
        response_data["flight"][0]["evidence"].reverse()
    assert run(trip, passages)["flight"]["findings"][0]["status"] == "Compliant"


def test_no_exception_note_cannot_establish_permission_alone(ordinary_scenario, response_data, api):
    trip, passages = ordinary_scenario
    response_data["flight"][0]["evidence"].pop(0)
    cabin = run(trip, passages)["flight"]["findings"][0]
    assert cabin["status"] == "Insufficient information" and cabin["validation_error"]


@pytest.mark.parametrize("permitted", ["Premium economy", "Business", "First"])
def test_corroboration_uses_the_active_policy_not_a_fixed_cabin(ordinary_scenario, response_data, api, permitted):
    trip, passages = ordinary_scenario
    rule = f"Flights must use {permitted} class only."
    support = SUPPORT.replace("Economy", permitted)
    passages["flight"][0]["text"] = rule + " " + CONTEXT
    passages["flight"][2]["text"] = support
    response_data["flight"][0].update(allowed_cabins=[permitted], evidence=[
        {"passage_id": "flight_1", "quote": rule}, {"passage_id": "flight_3", "quote": support}])
    trip["cabin"] = permitted
    result = run(trip, passages)
    assert result["flight"]["findings"][0]["status"] == "Compliant"
    assert [item["status"] for item in result["flight"]["display_findings"]] == ["Compliant", "Compliant", "Insufficient information"]


@pytest.mark.parametrize("change", ["premium_rule", "premium_note", "conditional_note", "another_source", "conflict"])
def test_unsupported_cabin_relationship_is_not_ignored(ordinary_scenario, response_data, api, change):
    trip, passages = ordinary_scenario
    if change == "premium_rule":
        quote = "Flights must use Premium economy class only."
        passages["flight"][0]["text"] = quote + " " + CONTEXT
        response_data["flight"][0].update(allowed_cabins=["Premium economy"])
        response_data["flight"][0]["evidence"][0]["quote"] = quote
        trip["cabin"] = "Premium economy"
    elif change in ("premium_note", "conditional_note"):
        quote = (SUPPORT.replace("Economy", "Premium economy") if change == "premium_note" else SUPPORT[:-1] + " unless another condition applies.")
        passages["flight"][2]["text"] = quote
        response_data["flight"][0]["evidence"][1]["quote"] = quote
    elif change == "another_source":
        passages["flight"][2]["source"] = "another.pdf"
    else:
        quote = "Flights must use Premium economy class only."
        passages["flight"][0]["text"] += " " + quote
        response_data["flight"][0]["evidence"].append({"passage_id": "flight_1", "quote": quote})
    cabin = run(trip, passages)["flight"]["findings"][0]
    assert cabin["status"] == "Insufficient information" and cabin["validation_error"]


@pytest.mark.parametrize("airfare", [None, 1000])
def test_within_limits_note_does_not_waive_independent_airfare_check(ordinary_scenario, api, airfare):
    trip, passages = ordinary_scenario
    trip["airfare"] = airfare
    result = run(trip, passages)
    assert result["flight"]["findings"][0]["status"] == "Compliant"
    assert result["flight"]["status"] == ("Non-compliant" if airfare else "Insufficient information")
    assert result["flight"]["findings"][1]["status"] != "Compliant"


@pytest.mark.parametrize("quote", [
    "Flight bookings require director approval.",
    "Approval may be necessary depending on circumstances.",
    "All official travel requires finance approval.",
])
def test_independent_or_ambiguous_approval_prevents_placeholder_removal(ordinary_scenario, api, quote):
    trip, passages = ordinary_scenario
    passages["flight"].append({"text": quote, "source": "fictional.pdf", "page": 8})
    result = run(trip, passages)["flight"]
    assert result["status"] == "Insufficient information"
    assert any(item["requirement"] == "Approval verification" and not item["citations"] for item in result["display_findings"])


def test_applicable_business_exception_keeps_empty_check_distinct(ordinary_scenario, response_data, api):
    trip, passages = ordinary_scenario
    trip.update(cabin="Business", flight_approval="Claimed approved", context="Fictional approver Alex approved in writing yesterday.")
    response_data["flight"][0]["evidence"].pop()
    result = run(trip, passages)["flight"]
    assert result["status"] == "Insufficient information"
    assert any(item["requirement"] == "Approval verification" and not item["citations"] for item in result["display_findings"])


@pytest.mark.parametrize("failure", ["no_snapshot", "no_owner", "invalid_owner", "protected_placeholder"])
def test_placeholder_needs_complete_policy_and_valid_owner(ordinary_scenario, api, failure):
    trip, passages = ordinary_scenario
    ledger = deepcopy(run(trip, passages)["flight"]["findings"])
    complete = [p for rows in passages.values() for p in rows]
    if failure == "no_snapshot":
        complete = None
    elif failure == "no_owner":
        ledger = [item for item in ledger if not (item["requirement"] == "Approval verification" and item["citations"])]
    elif failure == "invalid_owner":
        for item in ledger:
            if item["requirement"] == "Approval verification" and item["citations"]:
                item["validation_error"] = True
    else:
        for item in ledger:
            if item["requirement"] == "Approval verification" and not item["citations"]:
                item["validation_error"] = True
    display_input = checker._approval_placeholder_view(ledger, trip, "flight", complete, "policy_v1")
    assert any(item["requirement"] == "Approval verification" and not item["citations"] for item in display_input)
    assert display_input == ledger


def test_invalid_general_quote_warning_is_not_hidden(ordinary_scenario, response_data, api):
    trip, passages = ordinary_scenario
    response_data["flight"][2]["evidence"][0]["quote"] = "All business travel is already approved."
    result = run(trip, passages)["flight"]
    assert result["status"] == "Insufficient information"
    assert any(item.get("validation_error") for item in result["display_findings"])
    assert all(c["quote"] != "All business travel is already approved." for item in result["display_findings"] for c in item["citations"])


def test_claimed_approver_and_date_do_not_resolve_general_authorisation(ordinary_scenario, api):
    trip, passages = ordinary_scenario
    trip.update(flight_approval="Claimed approved", accommodation_approval="Claimed approved",
                context="Fictional Alex Sample signed written approval on 2026-11-27.")
    result = run(trip, passages)
    for section in checker.SECTIONS:
        approvals = [item for item in result[section]["display_findings"] if item["requirement"] == "Approval verification"]
        assert len(approvals) == 1 and approvals[0]["status"] == "Insufficient information"
        assert "self-reported" in approvals[0]["explanation"]


@pytest.mark.parametrize("quote", [
    "These rules apply only to travel to Japan.",
    "These rules apply to every destination except restricted locations.",
    "These rules apply to every destination if prior approval is obtained.",
    "These rules apply to every destination. Security review is required before booking.",
])
def test_restrictive_or_conditional_scope_is_not_pure_context(ordinary_scenario, response_data, api, quote):
    trip, passages = ordinary_scenario
    passages["flight"][0]["text"] = passages["flight"][0]["text"].replace(CONTEXT, quote)
    response_data["flight"][-1]["evidence"][0]["quote"] = quote
    result = run(trip, passages)
    other = next(item for item in result["flight"]["findings"] if item["requirement"] == "Additional policy requirement" and not item.get("coverage_missing"))
    assert other["status"] == "Insufficient information" and not other.get("context_only")
    assert any(item["requirement"] == "Additional policy requirement" for item in result["flight"]["display_findings"])


@pytest.mark.parametrize("case", ["no_antecedent", "before_rules", "mixed_components"])
def test_context_ownership_must_be_established(ordinary_scenario, response_data, api, case):
    trip, passages = ordinary_scenario
    if case == "no_antecedent":
        passages["flight"].append({"text": CONTEXT, "source": "fictional.pdf", "page": 8})
        response_data["flight"][-1]["evidence"][0]["passage_id"] = "flight_5"
    elif case == "before_rules":
        passages["flight"][0]["text"] = CONTEXT + " Flights must use Economy class."
    else:
        passages["flight"][0]["text"] = "Flights must use Economy class. Hotels must have at most 4 stars. " + CONTEXT
    result = run(trip, passages)
    other = next(item for item in result["flight"]["findings"] if item["requirement"] == "Additional policy requirement")
    assert not other.get("context_only")


def test_context_cannot_hide_independent_destination_obligation(ordinary_scenario, api):
    trip, passages = ordinary_scenario
    restriction = "Flights to restricted locations must have a security review."
    passages["flight"][0]["text"] += " " + restriction
    result = run(trip, passages)["flight"]
    assert any(item.get("context_only") for item in result["findings"])
    assert any(item.get("coverage_missing") and any(c["quote"] == restriction for c in item["citations"]) for item in result["display_findings"])


def test_context_cannot_cover_omitted_native_requirements(ordinary_scenario, response_data, api):
    trip, passages = ordinary_scenario
    response_data["flight"] = response_data["flight"][2:]
    result = run(trip, passages)["flight"]
    assert result["status"] == "Insufficient information"
    for label in ("Cabin class", "Airfare"):
        assert any(item["requirement"] == label and item.get("coverage_missing") for item in result["display_findings"])


def test_context_and_corroboration_do_not_add_requirements_without_authorisation_rule(ordinary_scenario, response_data, api):
    trip, passages = ordinary_scenario
    for section in checker.SECTIONS:
        passages[section].pop()  # remove only the general-authorisation page
        response_data[section] = [item for item in response_data[section] if item["kind"] != "approval"]
    result = run(trip, passages)
    assert all(section["status"] == "Compliant" for section in result.values())
    assert any(item.get("context_only") for item in result["flight"]["findings"])
    assert all(item["status"] == "Compliant" for section in result.values() for item in section["display_findings"])


def test_accommodation_applicability_context_is_also_not_an_obligation(ordinary_scenario, response_data, api):
    trip, passages = ordinary_scenario
    passages["accommodation"][0]["text"] += " " + CONTEXT
    response_data["accommodation"].append(finding("other", "accommodation_1", CONTEXT, status="Insufficient information"))
    result = run(trip, passages)
    assert any(item.get("context_only") for item in result["accommodation"]["findings"])
    assert len(result["accommodation"]["display_findings"]) == 3
    assert CONTEXT in [c["quote"] for c in result["accommodation"]["display_findings"][0]["citations"]]


def test_invalid_context_quote_remains_a_validation_warning(ordinary_scenario, response_data, api):
    trip, passages = ordinary_scenario
    response_data["flight"][-1]["evidence"][0]["quote"] = "These rules waive every destination restriction."
    result = run(trip, passages)["flight"]
    assert any(item.get("validation_error") and item["requirement"] == "Additional policy requirement" for item in result["display_findings"])


def test_display_corrections_are_stable_and_do_not_mutate_checked_ledger(ordinary_scenario, api):
    trip, passages = ordinary_scenario
    ledger = run(trip, passages)["flight"]["findings"]
    saved = deepcopy(ledger)
    complete = [p for rows in passages.values() for p in rows]
    view = checker._approval_placeholder_view(ledger, trip, "flight", complete, "policy_v1")
    first = checker._display_findings(view, trip, "policy_v1", complete)
    assert checker._display_findings(first, trip, "policy_v1", complete) == first
    assert ledger == saved


def test_streamlit_displays_native_checks_and_one_unverified_general_obligation(ordinary_scenario, api, monkeypatch):
    from unittest.mock import MagicMock
    from test_compliance_page import setup_page, widget
    from travel_assistant import compliance_page

    trip, passages = ordinary_scenario
    result = {"collection": "policy_v1", **run(trip, passages)}
    monkeypatch.setattr(compliance_page, "assess_trip", MagicMock(return_value=result))
    app = setup_page(monkeypatch)
    widget(app, "text_input", "Origin").set_value("Singapore")
    widget(app, "text_input", "Destination").set_value("Japan").run()
    widget(app, "button", "Check compliance").click().run()
    assert not app.exception
    assert len([item for item in app.markdown if item.value == "Approval verification"]) == 2  # one per component
    assert not any(item.value == "Additional policy requirement" for item in app.markdown)
    assert not app.warning
    assert any(item.value == CONTEXT for item in app.text)
