"""Check that the technical diagrams render through native Streamlit elements."""

from streamlit.testing.v1 import AppTest


def test_methodology_renders_four_accessible_native_diagrams():
    app = AppTest.from_string("from travel_assistant.pages import methodology\nmethodology()").run()

    assert not app.exception
    assert app.title[0].value == "Methodology"
    charts = app.get("graphviz_chart")
    assert len(charts) == 4
    assert all(chart.proto.alt for chart in charts)
    assert not app.code  # Workflows are visual diagrams rather than code blocks.
    architecture, ingestion, qa, compliance = [chart.proto.spec for chart in charts]
    assert "LOCAL APPLICATION AND KNOWLEDGE BASE" in architecture
    assert "EXTERNAL OPENAI API PROCESSING" in architecture
    assert 'complete -> activate [label="Yes"]' in ingestion
    assert 'complete -> preserve [label="No"]' in ingestion
    assert 'enough -> answer [label="Yes"]' in qa
    assert 'enough -> insufficient [label="No"]' in qa
    for status in ("Compliant", "Non-compliant", "Insufficient information"):
        assert f'label="{status}"' in compliance
