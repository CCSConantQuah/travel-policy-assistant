"""Chat UI checks with fake policy state and mocked answer service."""

from unittest.mock import MagicMock

from streamlit.testing.v1 import AppTest

from travel_assistant import chat_page
from travel_assistant.documents import PolicyError


def render_chat():
    from travel_assistant.chat_page import questions
    questions()


def test_chat_history_sources_clear_and_replacement(monkeypatch):
    active = {"collection": "policy_v1", "filename": "fictional.pdf"}
    monkeypatch.setattr(chat_page, "get_active_policy", lambda: active)
    monkeypatch.setattr(chat_page, "get_setting", lambda name: "mock-key" if name == "OPENAI_API_KEY" else "")
    ask = MagicMock(return_value={"answer": "Economy class. [1]", "citations": [
        {"number": 1, "source": "fictional.pdf", "page": 1, "quote": "Use economy class."},
    ]})
    monkeypatch.setattr(chat_page, "ask_policy", ask)
    app = AppTest.from_function(render_chat).run()
    app.chat_input[0].set_value("Which cabin class?").run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert len(app.session_state["qa_messages"]) == 2
    assert any("fictional.pdf — page 1" in item.value for item in app.text)
    app.run()
    assert ask.call_count == 1  # Rendering history must not generate again.
    app.button[0].click().run()
    assert app.session_state["qa_messages"] == []
    app.chat_input[0].set_value("Which cabin class?").run()
    active["collection"] = "policy_v2"
    app.run()
    assert app.session_state["qa_messages"] == []
    assert len(app.chat_message) == 0


def test_chat_no_policy_and_missing_key(monkeypatch):
    active = None
    monkeypatch.setattr(chat_page, "get_active_policy", lambda: active)
    monkeypatch.setattr(chat_page, "get_setting", lambda name: "")
    ask = MagicMock()
    monkeypatch.setattr(chat_page, "ask_policy", ask)
    app = AppTest.from_function(render_chat).run()
    assert "Document Upload" in app.info[0].value
    assert len(app.chat_input) == 0
    active = {"collection": "policy_v1", "filename": "fictional.pdf"}
    app.run()
    assert app.chat_input[0].disabled
    ask.assert_not_called()


def test_chat_api_failure_does_not_save_failed_turn(monkeypatch):
    monkeypatch.setattr(chat_page, "get_active_policy", lambda: {"collection": "policy_v1", "filename": "fictional.pdf"})
    monkeypatch.setattr(chat_page, "get_setting", lambda name: "mock-key")
    monkeypatch.setattr(chat_page, "ask_policy", MagicMock(side_effect=PolicyError("Please retry.")))
    app = AppTest.from_function(render_chat).run()
    app.chat_input[0].set_value("Hotel?").run()
    assert not app.exception
    assert app.error[0].value == "Please retry."
    assert app.session_state["qa_messages"] == []


def test_replacement_during_answer_removes_old_visible_history(monkeypatch):
    active = {"collection": "policy_v1", "filename": "fictional.pdf"}
    monkeypatch.setattr(chat_page, "get_active_policy", lambda: active)
    monkeypatch.setattr(chat_page, "get_setting", lambda name: "mock-key")
    def replace_while_answering(*args, **kwargs):
        active["collection"] = "policy_v2"
        raise PolicyError("The active policy changed while answering.")
    monkeypatch.setattr(chat_page, "ask_policy", replace_while_answering)
    app = AppTest.from_function(render_chat).run()
    app.session_state["qa_messages"] = [{"role": "assistant", "content": "Old answer", "citations": []}]
    app.chat_input[0].set_value("Hotel?").run()
    assert not app.exception
    assert app.session_state["qa_policy"] == "policy_v2"
    assert app.session_state["qa_messages"] == []
    assert len(app.chat_message) == 0
    assert "Conversation cleared" in app.info[0].value
