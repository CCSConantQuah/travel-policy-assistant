"""Check the existing login and navigation without using real credentials."""

from pathlib import Path

import dotenv
import pytest
from streamlit.testing.v1 import AppTest

from travel_assistant.notices import POC_DISCLAIMER, SIDEBAR_DISCLAIMER


def test_login_navigation_and_logout(monkeypatch):
    from travel_assistant import chat_page, compliance_page, upload_page, pages

    monkeypatch.setenv("DEMO_USERNAME", "test-officer")
    monkeypatch.setenv("DEMO_PASSWORD", "test-password")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda: False)
    # UI smoke checks must not read or write a developer's real policy store.
    monkeypatch.setattr(pages, "get_active_policy", lambda: None)
    monkeypatch.setattr(upload_page, "get_active_policy", lambda: None)
    monkeypatch.setattr(chat_page, "get_active_policy", lambda: None)
    monkeypatch.setattr(compliance_page, "get_active_policy", lambda: None)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=15).run()
    assert not app.exception
    assert [item.value for item in app.warning] == [POC_DISCLAIMER]
    assert not app.sidebar.caption
    app.text_input[0].set_value("wrong")
    app.text_input[1].set_value("wrong")
    app.button[0].click().run()
    assert len(app.error) == 1 and len(app.radio) == 0
    app.text_input[0].set_value("test-officer")
    app.text_input[1].set_value("test-password")
    app.button[0].click().run()
    assert app.session_state["authenticated"]
    assert [item.value for item in app.main.warning] == [POC_DISCLAIMER]
    assert len(app.radio[0].options) == 6
    for page in app.radio[0].options:
        app.radio[0].set_value(page).run()
        assert not app.exception, (page, app.exception)
        assert any(item.value == SIDEBAR_DISCLAIMER for item in app.sidebar.caption)
        assert any(item.value == "Use synthetic policy documents only." for item in app.sidebar.caption)
        assert sum(item.value == POC_DISCLAIMER for item in app.main.warning) == (1 if page == "Home" else 0)
        if page == "Policy Documents":
            assert app.get("file_uploader")
            assert next(button for button in app.button if button.label == "Process policy").disabled
    next(button for button in app.button if button.label == "Log out").click().run()
    assert not app.exception and len(app.radio) == 0
    assert "authenticated" not in app.session_state
    assert [item.value for item in app.warning] == [POC_DISCLAIMER]


def test_login_disclaimer_remains_visible_when_credentials_are_unconfigured(monkeypatch):
    monkeypatch.setenv("DEMO_USERNAME", "")
    monkeypatch.setenv("DEMO_PASSWORD", "")
    app = AppTest.from_string("from travel_assistant.auth import require_login\nrequire_login()").run()
    assert not app.exception
    assert [item.value for item in app.warning] == [POC_DISCLAIMER]
    assert app.error


@pytest.mark.parametrize(("button", "page"), [
    ("Open Policy Documents", "Policy Documents"),
    ("Explore policy Q&A", "Travel Policy Q&A"),
    ("Review travel compliance", "Travel Policy Compliance Checker"),
])
def test_home_buttons_use_sidebar_navigation(monkeypatch, button, page):
    from travel_assistant import pages, chat_page, compliance_page, upload_page

    monkeypatch.setattr(dotenv, "load_dotenv", lambda: False)
    for module in (pages, chat_page, compliance_page, upload_page):
        monkeypatch.setattr(module, "get_active_policy", lambda: None)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=15)
    app.session_state["authenticated"] = True
    app.run()
    next(item for item in app.button if item.label == button).click().run()
    assert not app.exception
    assert app.radio[0].value == page


def test_home_displays_active_policy(monkeypatch):
    from travel_assistant import pages

    monkeypatch.setattr(dotenv, "load_dotenv", lambda: False)
    monkeypatch.setattr(pages, "get_active_policy", lambda: {"filename": "fictional.pdf", "status": "Ready"})
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=15)
    app.session_state["authenticated"] = True
    app.run()
    assert not app.exception
    assert any("fictional.pdf" in item.value for item in app.text)
    assert any("Ready" in item.value for item in app.success)


def test_policy_documents_formats_timestamp_and_collapses_retrieval(monkeypatch):
    from travel_assistant import upload_page

    monkeypatch.setattr(upload_page, "get_active_policy", lambda: {
        "filename": "fictional.pdf", "page_count": 2, "text_page_count": 2,
        "chunk_count": 4, "status": "Ready", "processed_at": "2026-11-27T08:05:12+00:00",
    })
    app = AppTest.from_string("from travel_assistant.upload_page import documents\ndocuments()").run()
    assert not app.exception
    assert any("27 Nov 2026, 08:05 UTC" in item.value for item in app.caption)
    assert not app.expander[0].proto.expanded
