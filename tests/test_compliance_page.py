"""Exercise the normal form entry point without credentials or API calls."""

from unittest.mock import MagicMock

from streamlit.testing.v1 import AppTest

from travel_assistant import compliance_page
from travel_assistant.documents import PolicyError


def render_compliance():
    from travel_assistant.compliance_page import compliance
    compliance()


def widget(app, kind, label):
    return next(item for item in getattr(app, kind) if item.label == label)


def setup_page(monkeypatch, active=True, key="mock-key"):
    monkeypatch.setattr(compliance_page, "get_active_policy", lambda: {"collection": "policy_v1"} if active else None)
    monkeypatch.setattr(compliance_page, "get_setting", lambda name: key if name == "OPENAI_API_KEY" else "")
    return AppTest.from_function(render_compliance, default_timeout=15).run()


def test_button_validation_and_currency_pairing(monkeypatch):
    app = setup_page(monkeypatch)
    assert not app.exception
    assert widget(app, "button", "Check compliance").disabled
    widget(app, "text_input", "Origin").set_value("Fictional A")
    widget(app, "text_input", "Destination").set_value("Fictional B").run()
    assert not widget(app, "button", "Check compliance").disabled
    widget(app, "number_input", "Estimated airfare").set_value(800.0).run()
    assert widget(app, "button", "Check compliance").disabled
    widget(app, "selectbox", "Airfare currency").set_value("SGD").run()
    assert not widget(app, "button", "Check compliance").disabled
    assert widget(app, "selectbox", "Hotel star rating")
    assert widget(app, "selectbox", "Flight approval status (self-reported)")


def test_ui_renders_individual_findings_and_advisory_summary(monkeypatch):
    finding = {"status": "Non-compliant", "explanation": "The supplied rate exceeds the cited nightly limit.",
               "citations": [{"number": 1, "source": "fictional.pdf", "page": 2, "quote": "The nightly limit is SGD 180."}]}
    result = {"collection": "policy_v1", "flight": {"status": "Insufficient information", "findings": []},
              "accommodation": {"status": "Non-compliant", "findings": [finding]}}
    assess = MagicMock(return_value=result)
    monkeypatch.setattr(compliance_page, "assess_trip", assess)
    app = setup_page(monkeypatch)
    widget(app, "text_input", "Origin").set_value("Fictional A")
    widget(app, "text_input", "Destination").set_value("Fictional B").run()
    widget(app, "button", "Check compliance").click().run()
    assert not app.exception
    assert any("advisory" in message.value for message in app.info)
    assert any("fictional.pdf — page 2" in item.value for item in app.text)
    assert assess.call_args.kwargs["expected_collection"] == "policy_v1"
    # Editing inputs removes results until a new assessment is explicitly requested.
    widget(app, "text_input", "Destination").set_value("Fictional C").run()
    assert not any("fictional.pdf — page 2" in item.value for item in app.text)
    assert assess.call_count == 1


def test_no_policy_or_missing_key_disables_assessment(monkeypatch):
    app = setup_page(monkeypatch, active=False)
    assert widget(app, "button", "Check compliance").disabled
    assert any("Policy Documents" in message.value for message in app.info)
    app = setup_page(monkeypatch, key="")
    widget(app, "text_input", "Origin").set_value("Fictional A")
    widget(app, "text_input", "Destination").set_value("Fictional B").run()
    assert widget(app, "button", "Check compliance").disabled


def test_assessment_error_is_displayed_without_stale_results(monkeypatch):
    monkeypatch.setattr(compliance_page, "assess_trip", MagicMock(side_effect=PolicyError("Please retry the assessment.")))
    app = setup_page(monkeypatch)
    widget(app, "text_input", "Origin").set_value("Fictional A")
    widget(app, "text_input", "Destination").set_value("Fictional B").run()
    widget(app, "button", "Check compliance").click().run()
    assert not app.exception
    assert app.error[0].value == "Please retry the assessment."


def test_failed_citation_is_labelled_and_visible_beside_confirmed_violation(monkeypatch):
    invalid = {"requirement": "Hotel nightly rate", "status": "Insufficient information",
               "explanation": "The supporting quote does not match the cited passage.",
               "citations": [], "validation_error": True}
    valid = {"requirement": "Hotel nightly rate", "status": "Non-compliant",
             "explanation": "The supplied rate exceeds the validated policy ceiling.", "citations": []}
    result = {"collection": "policy_v1", "flight": {"status": "Compliant", "findings": []},
              "accommodation": {"status": "Non-compliant", "findings": [valid, invalid]}}
    monkeypatch.setattr(compliance_page, "assess_trip", MagicMock(return_value=result))
    app = setup_page(monkeypatch)
    widget(app, "text_input", "Origin").set_value("Singapore")
    widget(app, "text_input", "Destination").set_value("Japan").run()
    widget(app, "button", "Check compliance").click().run()
    assert not app.exception
    assert any("Evidence validation failed" in item.value for item in app.warning)
    assert any("Hotel nightly rate" in item.value for item in app.markdown)
    assert any("Accommodation — Non-compliant" in item.value for item in app.info)
