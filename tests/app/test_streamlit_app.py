from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_streamlit_app_runs_as_a_script_from_repository_root() -> None:
    result = subprocess.run(
        [sys.executable, "app/streamlit_app.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"


def test_streamlit_app_renders_core_sections() -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

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
