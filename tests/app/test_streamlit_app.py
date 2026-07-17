from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_core_sections() -> None:
    app = AppTest.from_file("app/streamlit_app.py", default_timeout=45).run()

    assert not app.exception
    assert app.title[0].value == "Better Router Adaptive"
    visible_text = "\n".join(
        [
            *(item.value for item in app.markdown),
            *(item.value for item in app.warning),
            *(item.value for item in app.info),
        ]
    )
    assert "Rodolfo Fritz" in visible_text
    assert "Benjamín Cerda" in visible_text
    assert "Felipe Friz" in visible_text
    assert "demostración sintética" in visible_text.lower()
    assert any(button.label == "Simular enrutamiento" for button in app.button)
