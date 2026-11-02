"""Check the existing login and navigation without using real credentials."""

from pathlib import Path

import dotenv
from streamlit.testing.v1 import AppTest


def test_login_navigation_and_logout(monkeypatch):
    from travel_assistant import chat_page, upload_page

    monkeypatch.setenv("DEMO_USERNAME", "test-officer")
    monkeypatch.setenv("DEMO_PASSWORD", "test-password")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda: False)
    # UI smoke checks must not read or write a developer's real policy store.
    monkeypatch.setattr(upload_page, "get_active_policy", lambda: None)
    monkeypatch.setattr(chat_page, "get_active_policy", lambda: None)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=15).run()
    assert not app.exception
    app.text_input[0].set_value("wrong")
    app.text_input[1].set_value("wrong")
    app.button[0].click().run()
    assert len(app.error) == 1 and len(app.radio) == 0
    app.text_input[0].set_value("test-officer")
    app.text_input[1].set_value("test-password")
    app.button[0].click().run()
    assert app.session_state["authenticated"]
    assert len(app.radio[0].options) == 6
    for page in app.radio[0].options:
        app.radio[0].set_value(page).run()
        assert not app.exception, (page, app.exception)
        if page == "Document Upload":
            assert app.get("file_uploader")
            assert next(button for button in app.button if button.label == "Process policy").disabled
    next(button for button in app.button if button.label == "Log out").click().run()
    assert not app.exception and len(app.radio) == 0
    assert "authenticated" not in app.session_state
