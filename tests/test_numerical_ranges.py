"""Supported inclusive range coverage with every external API call mocked."""

from copy import deepcopy

import pytest

from test_compliance import api, finding, passages, response_data, run, trip
from travel_assistant import compliance as checker


def range_policy(passages, response_data, lower=2, upper=4, endpoint="upper"):
    quote = f"Hotels rated from {lower} to {upper} stars (inclusive) are permitted for official travel."
    passages["accommodation"][1]["text"] = quote
    response_data["accommodation"][1].update(
        policy_limit=str(upper if endpoint == "upper" else lower),
        operator="at_most" if endpoint == "upper" else "at_least",
        evidence=[{"passage_id": "accommodation_2", "quote": quote}],
    )
    return quote


@pytest.mark.parametrize("stars,endpoint", [(1, "upper"), (5, "lower")])
def test_single_endpoint_cannot_hide_violation_of_other_range_bound(trip, passages, response_data, api, stars, endpoint):
    import jsonschema

    range_policy(passages, response_data, endpoint=endpoint)
    trip["stars"] = stars
    result = run(trip, passages)
    jsonschema.validate(response_data, api.responses.create.call_args.kwargs["text"]["format"]["schema"])
    assert result["accommodation"]["findings"][1]["status"] == "Non-compliant"
    assert result["accommodation"]["status"] == "Non-compliant"
    assert result["flight"]["status"] == "Compliant"


@pytest.mark.parametrize("stars", [2, 3, 4])
@pytest.mark.parametrize("endpoint", ["lower", "upper"])
def test_inclusive_boundaries_and_interior_check_both_conditions(trip, passages, response_data, api, stars, endpoint):
    range_policy(passages, response_data, endpoint=endpoint)
    trip["stars"] = stars
    result = run(trip, passages)
    hotel = result["accommodation"]
    assert hotel["status"] == "Compliant"
    star = hotel["findings"][1]
    assert star["range_checks"] == [
        {"operator": "at_least", "policy_limit": "2", "status": "Compliant"},
        {"operator": "at_most", "policy_limit": "4", "status": "Compliant"},
    ]
    assert not any(row.get("coverage_missing") for row in hotel["findings"])


@pytest.mark.parametrize("endpoints", [("lower", "upper"), ("upper", "lower"), ("upper", "upper"), ("lower", "upper", "upper")])
@pytest.mark.parametrize("stars", [1, 3, 5])
def test_endpoint_order_and_duplicates_do_not_change_complete_evaluation(trip, passages, response_data, api, endpoints, stars):
    range_policy(passages, response_data)
    template = deepcopy(response_data["accommodation"][1])
    response_data["accommodation"] = response_data["accommodation"][:1]
    for endpoint in endpoints:
        response_data["accommodation"].append({**deepcopy(template),
            "operator": "at_least" if endpoint == "lower" else "at_most", "policy_limit": "2" if endpoint == "lower" else "4"})
    trip["stars"] = stars
    result = run(trip, passages)["accommodation"]
    expected = "Compliant" if 2 <= stars <= 4 else "Non-compliant"
    assert result["status"] == expected
    checks = [row for row in result["findings"] if row["requirement"] == "Hotel star rating"]
    assert len(checks) == len(endpoints)
    assert all(row["status"] == expected and len(row["range_checks"]) == 2 for row in checks)
    assert not any(row.get("coverage_missing") for row in result["findings"])
    assert all(row["status"] == expected for row in result["display_findings"] if row["requirement"] == "Hotel star rating")


def test_missing_rating_does_not_complete_range_coverage(trip, passages, response_data, api):
    range_policy(passages, response_data)
    trip["stars"] = None
    result = run(trip, passages)["accommodation"]
    assert result["status"] == "Insufficient information"
    assert not result["findings"][1].get("range_checks")
    assert any(row.get("coverage_missing") and row["requirement"] == "Hotel star rating" for row in result["findings"])


@pytest.mark.parametrize("failure", ["quote", "passage"])
def test_range_bounds_never_come_from_invalid_evidence(trip, passages, response_data, api, failure):
    range_policy(passages, response_data)
    reference = response_data["accommodation"][1]["evidence"][0]
    if failure == "quote":
        reference["quote"] = reference["quote"].replace("2 to 4", "1 to 4")
    else:
        reference["passage_id"] = "invented_passage"
    result = run(trip, passages)["accommodation"]
    star = result["findings"][1]
    assert result["status"] == "Insufficient information"
    assert star["validation_error"] and not star["citations"] and not star.get("range_checks")


@pytest.mark.parametrize("limit,operator", [("3", "at_most"), ("2", "at_most"), ("4", "at_least"), ("4", "equal")])
def test_model_endpoint_must_match_validated_range(trip, passages, response_data, api, limit, operator):
    range_policy(passages, response_data)
    response_data["accommodation"][1].update(policy_limit=limit, operator=operator)
    result = run(trip, passages)["accommodation"]
    assert result["status"] == "Insufficient information"
    assert not result["findings"][1].get("range_checks")


@pytest.mark.parametrize("quote", [
    "Hotels must have between 2 and 4 stars.",
    "Hotels must have from 2 to 4 stars (exclusive).",
    "Hotels must have from 2 to 4 stars.",
    "Hotels rated from 2 to 4 stars (inclusive) are permitted unless special conditions apply.",
    "Hotels rated from 2 to 4 stars (inclusive) are permitted, but not 3 stars.",
    "Hotels rated from 2 to 4 stars (inclusive) or from 3 to 5 stars (inclusive) are permitted.",
])
def test_ambiguous_or_unsupported_ranges_abstain(trip, passages, response_data, api, quote):
    range_policy(passages, response_data)
    passages["accommodation"][1]["text"] = quote
    response_data["accommodation"][1]["evidence"][0]["quote"] = quote
    result = run(trip, passages)["accommodation"]
    assert result["status"] == "Insufficient information"
    assert not result["findings"][1].get("range_checks")


def test_reversed_bounds_are_not_reinterpreted(trip, passages, response_data, api):
    range_policy(passages, response_data, lower=4, upper=2)
    result = run(trip, passages)["accommodation"]
    assert result["status"] == "Insufficient information"
    assert result["findings"][1]["validation_error"]
    assert any(row.get("coverage_missing") for row in result["findings"])


def test_distinct_ranges_in_separate_citations_are_not_blended(trip, passages, response_data, api):
    first = range_policy(passages, response_data)
    second = "Hotels rated from 4 to 5 stars (inclusive) are permitted for official travel."
    passages["accommodation"][1]["text"] = first + " " + second
    response_data["accommodation"][1]["evidence"].append({"passage_id": "accommodation_2", "quote": second})
    result = run(trip, passages)["accommodation"]
    assert result["status"] == "Insufficient information"
    assert result["findings"][1]["validation_error"] and not result["findings"][1].get("range_checks")


@pytest.mark.parametrize("lower,upper,stars,endpoint,expected", [
    (1, 3, 4, "lower", "Non-compliant"), (3, 5, 2, "upper", "Non-compliant"), (1, 5, 4, "upper", "Compliant"),
])
def test_range_thresholds_are_policy_driven(trip, passages, response_data, api, lower, upper, stars, endpoint, expected):
    range_policy(passages, response_data, lower, upper, endpoint)
    trip["stars"] = stars
    result = run(trip, passages)["accommodation"]
    assert result["status"] == expected
    assert [check["policy_limit"] for check in result["findings"][1]["range_checks"]] == [str(lower), str(upper)]


@pytest.mark.parametrize("stars", [2, 3, 4])
def test_equal_inclusive_endpoints_still_require_both_conditions(trip, passages, response_data, api, stars):
    range_policy(passages, response_data, lower=3, upper=3)
    trip["stars"] = stars
    result = run(trip, passages)["accommodation"]
    assert result["status"] == ("Compliant" if stars == 3 else "Non-compliant")
    assert len(result["findings"][1]["range_checks"]) == 2


@pytest.mark.parametrize("quote,limit,operator,stars,expected", [
    ("Hotels must have at most 4 stars.", "4", "at_most", 1, "Compliant"),
    ("Hotels must have at most 4 stars.", "4", "at_most", 5, "Non-compliant"),
    ("Hotels must have at least 2 stars.", "2", "at_least", 1, "Non-compliant"),
])
def test_single_bound_behaviour_is_unchanged(trip, passages, response_data, api, quote, limit, operator, stars, expected):
    passages["accommodation"][1]["text"] = quote
    response_data["accommodation"][1].update(policy_limit=limit, operator=operator,
        evidence=[{"passage_id": "accommodation_2", "quote": quote}])
    trip["stars"] = stars
    result = run(trip, passages)["accommodation"]
    assert result["status"] == expected and not result["findings"][1].get("range_checks")


@pytest.mark.parametrize("unresolved", ["approval_dependent", "conditions", "model_abstention"])
def test_unresolved_conditions_are_not_overridden(trip, passages, response_data, api, unresolved):
    range_policy(passages, response_data)
    if unresolved == "approval_dependent":
        response_data["accommodation"][1]["approval_dependent"] = True
        trip["accommodation_approval"] = "Claimed approved"
    elif unresolved == "conditions":
        response_data["accommodation"][1]["conditions_resolved"] = False
    else:
        response_data["accommodation"][1]["status"] = "Insufficient information"
    result = run(trip, passages)["accommodation"]
    assert result["status"] == "Insufficient information"
    assert not result["findings"][1].get("range_checks")
    assert any(row.get("coverage_missing") for row in result["findings"])


def test_range_violation_precedes_unverified_general_authorisation(trip, passages, response_data, api):
    range_policy(passages, response_data)
    trip.update(stars=1, flight_approval="Claimed approved", accommodation_approval="Claimed approved")
    for section in checker.SECTIONS:
        quote = "All business travel requires prior written approval before booking."
        passages[section].append({"text": quote, "source": "fictional.pdf", "page": 5})
        response_data[section].append(finding("approval", f"{section}_3", quote))
    result = run(trip, passages)
    assert result["accommodation"]["status"] == "Non-compliant"
    assert result["flight"]["status"] == "Insufficient information"
    assert any(row["requirement"] == "Approval verification" and row["status"] == "Insufficient information" for row in result["accommodation"]["display_findings"])


@pytest.mark.parametrize("section,index,kind,quote,limit", [
    ("flight", 1, "airfare", "Total airfare must be from SGD 500 to SGD 900 (inclusive).", "900"),
    ("accommodation", 0, "nightly_rate", "Hotel nightly rates must be from SGD 100 to SGD 180 (inclusive).", "180"),
])
def test_unsupported_monetary_range_grammar_is_not_expanded(trip, passages, response_data, api, section, index, kind, quote, limit):
    passages[section][index]["text"] = quote
    response_data[section][index].update(policy_limit=limit, operator="at_most", policy_currency="SGD",
        evidence=[{"passage_id": f"{section}_{index + 1}", "quote": quote}])
    result = run(trip, passages)
    assert result[section]["status"] == "Insufficient information"
    assert not result[section]["findings"][index].get("range_checks")


@pytest.mark.parametrize("remaining", ["at_least", "at_most"])
def test_coverage_does_not_accept_partial_range_check_metadata(trip, passages, response_data, api, remaining):
    range_policy(passages, response_data)
    result = run(trip, passages)
    ledger = deepcopy(result["accommodation"]["findings"])
    ledger[1]["range_checks"] = [check for check in ledger[1]["range_checks"] if check["operator"] == remaining]
    requirements = checker._coverage_requirements([p for rows in passages.values() for p in rows], checker.validate_trip(trip), "accommodation")
    missing = checker._missing_coverage(ledger, requirements)
    assert len(missing) == 1 and missing[0]["requirement"] == "Hotel star rating" and missing[0]["coverage_missing"]


def test_streamlit_displays_complete_range_not_only_model_endpoint(trip, passages, response_data, api, monkeypatch):
    from unittest.mock import MagicMock
    from test_compliance_page import setup_page, widget
    from travel_assistant import compliance_page

    range_policy(passages, response_data)
    trip["stars"] = 1
    result = {"collection": "policy_v1", **run(trip, passages)}
    monkeypatch.setattr(compliance_page, "assess_trip", MagicMock(return_value=result))
    app = setup_page(monkeypatch)
    widget(app, "text_input", "Origin").set_value("Fictional A")
    widget(app, "text_input", "Destination").set_value("Fictional B").run()
    widget(app, "button", "Check compliance").click().run()
    assert not app.exception
    assert any("Accommodation — Non-compliant" in message.value for message in app.info)
    assert any("at least 2 and at most 4 stars" in message.value and "Both bounds were evaluated" in message.value for message in app.text)
