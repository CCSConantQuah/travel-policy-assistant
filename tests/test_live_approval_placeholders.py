"""Real generated PDF layout plus mocked responses for the live placeholder bug."""

from pathlib import Path
import subprocess
import sys

import pytest

from test_compliance import api, finding, trip
from travel_assistant import compliance as checker
from travel_assistant.documents import chunk_pages, extract_pdf


@pytest.fixture
def complete_policy(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "create_synthetic_policy.py"
    subprocess.run([sys.executable, str(script), "--approval-scope"], cwd=tmp_path,
                   check=True, capture_output=True, text=True)
    outputs = list((tmp_path / "data").glob("*.pdf"))
    assert len(outputs) == 1
    pages, _ = extract_pdf(outputs[0].read_bytes(), outputs[0].name)
    return [{"text": chunk.page_content, **chunk.metadata} for chunk in chunk_pages(pages)]


@pytest.fixture
def passages(complete_policy):
    return {"flight": [complete_policy[0], complete_policy[2], complete_policy[3]],
            "accommodation": [complete_policy[1], complete_policy[2], complete_policy[3]]}


@pytest.fixture
def response_data(trip, passages):
    trip.update(origin="Singapore", destination="Japan", hotel="Standard Hotel")

    def rule(section, kind, **changes):
        quote = checker._rule_quote_options(passages[section][0]["text"], kind, trip, section)[0]
        return finding(kind, f"{section}_1", quote, **changes)

    return {
        "flight": [rule("flight", "cabin", allowed_cabins=["Economy"]),
                   rule("flight", "airfare", policy_limit="900", policy_currency="SGD", operator="at_most"),
                   finding("approval", "", "", evidence=[], status="Insufficient information")],
        "accommodation": [rule("accommodation", "nightly_rate", policy_limit="180", policy_currency="SGD", operator="at_most"),
                          rule("accommodation", "stars", policy_limit="4", operator="at_most")],
    }


def assess(trip, passages, complete_policy):
    return checker.generate_assessment(checker.validate_trip(trip), passages, "mock-key",
                                       coverage_passages=complete_policy, policy_version="policy_generated")


def test_generated_pdf_title_does_not_block_fallback_placeholder_consolidation(trip, passages, complete_policy, response_data, api):
    import jsonschema

    result = assess(trip, passages, complete_policy)
    jsonschema.validate(response_data, api.responses.create.call_args.kwargs["text"]["format"]["schema"])
    for section in checker.SECTIONS:
        assert result[section]["status"] == "Insufficient information"
        assert [row["status"] for row in result[section]["findings"][:2]] == ["Compliant", "Compliant"]
        approval = [row for row in result[section]["display_findings"] if row["requirement"] == "Approval verification"]
        assert len(approval) == 1
        assert approval[0]["explanation"] == "General travel authorisation remains a separate requirement. Approval status is self-reported and requires verification."
        assert "General Travel Authorisation" in approval[0]["citations"][0]["quote"]
    original = [row for row in result["flight"]["findings"] if row["requirement"] == "Approval verification"]
    assert len(original) == 2 and not original[0]["citations"]
    assert not any(row.get("validation_error") for section in result.values() for row in section["findings"])


@pytest.mark.parametrize("model_owner", [False, True])
@pytest.mark.parametrize("retrieved_general", [False, True])
def test_owner_metadata_and_retrieval_order_do_not_control_collapse(trip, passages, complete_policy, response_data, api, model_owner, retrieved_general):
    if model_owner:
        general = next(quote for quote in checker._rule_quote_options(complete_policy[-1]["text"], "approval") if checker._general_approval(quote))
        response_data["flight"].insert(2, finding("approval", "flight_3", general))
    if not retrieved_general:
        # A model owner still needs a retrieved ID, so use the fallback owner
        # for this shape; the full active-policy text always remains available.
        response_data["flight"] = [row for row in response_data["flight"] if not (row["kind"] == "approval" and row["evidence"])]
        passages["flight"] = passages["flight"][:2]
    result = assess(trip, passages, complete_policy)
    approvals = [row for row in result["flight"]["display_findings"] if row["requirement"] == "Approval verification"]
    assert len(approvals) == 1 and approvals[0]["citations"]
    assert result["flight"]["status"] == "Insufficient information"
    assert not any(row.get("validation_error") for row in result["flight"]["findings"])


@pytest.mark.parametrize("title", [
    "APPROVAL IS REQUIRED BY POLICY",
    "APPROVAL BEFORE BOOKING POLICY",
    "APPROVAL POLICY FOR RESTRICTED DESTINATIONS",
    "MANDATORY APPROVAL POLICY",
    "APPROVAL WAIVED POLICY",
    "FLIGHT APPROVAL POLICY",
    "Fictional Approval Scope Policy - Testing Purposes Only",
])
def test_uncertain_or_duty_like_title_is_not_silently_ignored(trip, passages, complete_policy, response_data, api, title):
    _, body = complete_policy[-1]["text"].split("\n\n", 1)
    complete_policy[-1]["text"] = title + "\n\n" + body
    result = assess(trip, passages, complete_policy)["flight"]
    assert not checker._approval_document_title(title, complete_policy[-1])
    assert any(row["requirement"] == "Approval verification" and not row["citations"] for row in result["display_findings"])
    assert result["status"] == "Insufficient information"


@pytest.mark.parametrize("title", ["TRAVEL APPROVAL POLICY", "FICTIONAL TRAVEL APPROVAL POLICY - DEVELOPMENT ONLY"])
def test_recognised_title_format_is_not_bound_to_one_document_name(trip, passages, complete_policy, response_data, api, title):
    _, body = complete_policy[-1]["text"].split("\n\n", 1)
    complete_policy[-1]["text"] = title + "\n\n" + body
    result = assess(trip, passages, complete_policy)["flight"]
    assert checker._approval_document_title(title, complete_policy[-1])
    assert len([row for row in result["display_findings"] if row["requirement"] == "Approval verification"]) == 1


@pytest.mark.parametrize("rule", [
    "Flight bookings require director approval.",
    "All official travel requires finance approval.",
    "Approval may be necessary depending on circumstances.",
])
def test_other_real_or_ambiguous_approval_still_blocks_placeholder_removal(trip, passages, complete_policy, response_data, api, rule):
    complete_policy.append({"text": rule, "source": complete_policy[0]["source"], "page": max(p["page"] for p in complete_policy) + 1})
    result = assess(trip, passages, complete_policy)["flight"]
    assert any(row["requirement"] == "Approval verification" and not row["citations"] for row in result["display_findings"])
    assert result["status"] == "Insufficient information"


def test_business_exception_and_general_authorisation_remain_independent(trip, passages, complete_policy, response_data, api):
    trip.update(cabin="Business", flight_approval="Claimed approved", context="Fictional approver Alex supplied a claimed written approval date.")
    result = assess(trip, passages, complete_policy)["flight"]
    assert result["status"] == "Insufficient information"
    assert any(row["requirement"] == "Approval verification" and not row["citations"] for row in result["display_findings"])
    assert any("General Travel Authorisation" in c["quote"] for row in result["display_findings"] for c in row["citations"])
    assert any("Business class" in c["quote"] for row in result["findings"] for c in row["citations"])


@pytest.mark.parametrize("failure", ["no_snapshot", "no_owner", "invalid_owner", "coverage_owner"])
def test_title_recognition_does_not_replace_owner_safeguards(trip, passages, complete_policy, response_data, api, failure):
    ledger = assess(trip, passages, complete_policy)["flight"]["findings"]
    from copy import deepcopy
    ledger = deepcopy(ledger)
    snapshot = complete_policy
    if failure == "no_snapshot":
        snapshot = None
    elif failure == "no_owner":
        ledger = [row for row in ledger if not (row["requirement"] == "Approval verification" and row["citations"])]
    else:
        for row in ledger:
            if row["requirement"] == "Approval verification" and row["citations"]:
                row["validation_error" if failure == "invalid_owner" else "coverage_missing"] = True
    view = checker._approval_placeholder_view(ledger, trip, "flight", snapshot, "policy_generated")
    assert view == ledger
    assert any(row["requirement"] == "Approval verification" and not row["citations"] for row in view)


def test_invalid_placeholder_warning_survives_title_correction(trip, passages, complete_policy, response_data, api):
    response_data["flight"][-1]["evidence"] = [{"passage_id": "invented_id", "quote": "A fabricated approval provision."}]
    result = assess(trip, passages, complete_policy)["flight"]
    assert any(row.get("validation_error") and not row["citations"] for row in result["display_findings"])
    assert result["status"] == "Insufficient information"


def test_cited_title_finding_is_not_dropped_as_an_empty_placeholder(trip, passages, complete_policy, response_data, api):
    title = complete_policy[-1]["text"].split("\n\n", 1)[0]
    response_data["flight"][-1]["evidence"] = [{"passage_id": "flight_3", "quote": title}]
    result = assess(trip, passages, complete_policy)["flight"]
    assert any(any(c["quote"] == title for c in row["citations"]) for row in result["display_findings"])


def test_source_page_and_destination_are_not_hardcoded(trip, passages, complete_policy, response_data, api):
    for passage in complete_policy:
        passage.update(source="another_fictional_document.pdf", page=passage["page"] + 10)
    trip.update(origin="Fictional City C", destination="Fictional City D")
    result = assess(trip, passages, complete_policy)["flight"]
    approvals = [row for row in result["display_findings"] if row["requirement"] == "Approval verification"]
    assert len(approvals) == 1 and approvals[0]["citations"][0]["source"] == "another_fictional_document.pdf"


@pytest.mark.parametrize("field,flight_status,hotel_status", [
    ("airfare", "Non-compliant", "Insufficient information"),
    ("nightly_rate", "Insufficient information", "Non-compliant"),
])
def test_violation_precedence_is_unchanged(trip, passages, complete_policy, response_data, api, field, flight_status, hotel_status):
    trip[field] = 1000
    result = assess(trip, passages, complete_policy)
    assert result["flight"]["status"] == flight_status
    assert result["accommodation"]["status"] == hotel_status


def test_checked_ledger_and_section_status_are_unchanged(trip, passages, complete_policy, response_data, api, monkeypatch):
    corrected = assess(trip, passages, complete_policy)
    monkeypatch.setattr(checker, "_approval_placeholder_view", lambda ledger, *args: ledger)
    unfiltered = assess(trip, passages, complete_policy)
    for section in checker.SECTIONS:
        assert corrected[section]["findings"] == unfiltered[section]["findings"]
        assert corrected[section]["status"] == unfiltered[section]["status"]
    assert len([r for r in unfiltered["flight"]["display_findings"] if r["requirement"] == "Approval verification"]) == 2


def test_streamlit_shows_one_fallback_authorisation_per_component(trip, passages, complete_policy, response_data, api, monkeypatch):
    from unittest.mock import MagicMock
    from test_compliance_page import setup_page, widget
    from travel_assistant import compliance_page

    result = {"collection": "policy_generated", **assess(trip, passages, complete_policy)}
    monkeypatch.setattr(compliance_page, "assess_trip", MagicMock(return_value=result))
    app = setup_page(monkeypatch)
    widget(app, "text_input", "Origin").set_value("Singapore")
    widget(app, "text_input", "Destination").set_value("Japan").run()
    widget(app, "button", "Check compliance").click().run()
    assert not app.exception
    assert len([item for item in app.markdown if item.value == "Approval verification"]) == 2
    assert not any("The retrieved policy passages do not provide sufficient evidence for this requirement." in item.value for item in app.text)
