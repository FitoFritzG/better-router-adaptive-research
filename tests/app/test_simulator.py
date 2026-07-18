from __future__ import annotations

from pathlib import Path

import pytest

from app.simulator import (
    DemoBundle,
    SimulationError,
    display_arm,
    simulate_prompt,
    train_demo_bundle,
)
from better_router_adaptive.features import OUTCOME_COLUMNS


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def bundle(repo_root: Path) -> DemoBundle:
    return train_demo_bundle(repo_root)


def test_train_demo_bundle_uses_four_synthetic_arms(bundle: DemoBundle) -> None:
    assert bundle.arms == ("arm-fast", "arm-balanced", "arm-reasoning", "arm-premium")
    assert bundle.task_groups == ("coding", "mathematics", "reasoning", "general")
    assert set(bundle.scaler.columns).isdisjoint(OUTCOME_COLUMNS)


def test_simulate_prompt_is_deterministic(bundle: DemoBundle) -> None:
    first = simulate_prompt(
        bundle,
        prompt_text="Explica un algoritmo de búsqueda y su complejidad.",
        task_group="coding",
    )
    second = simulate_prompt(
        bundle,
        prompt_text="Explica un algoritmo de búsqueda y su complejidad.",
        task_group="coding",
    )

    assert first == second
    assert first.xgboost_arm in bundle.arms
    assert first.linucb_arm in bundle.arms
    assert first.prompt_char_count > 0
    assert first.prompt_word_count > 0
    assert first.prompt_avg_word_length > 0


@pytest.mark.parametrize("prompt", ["", "   ", "hola", "x" * 2001])
def test_simulate_prompt_validates_length(bundle: DemoBundle, prompt: str) -> None:
    with pytest.raises(SimulationError, match="between 5 and 2000"):
        simulate_prompt(bundle, prompt_text=prompt, task_group="general")


def test_simulate_prompt_rejects_unknown_group(bundle: DemoBundle) -> None:
    with pytest.raises(SimulationError, match="task group"):
        simulate_prompt(
            bundle,
            prompt_text="Consulta válida para el simulador.",
            task_group="unknown",
        )


def test_display_arm_uses_didactic_names() -> None:
    assert display_arm("arm-fast") == "Rápido"
    assert display_arm("arm-balanced") == "Equilibrado"
    assert display_arm("arm-reasoning") == "Razonamiento"
    assert display_arm("arm-premium") == "Premium"
    assert display_arm("custom-arm") == "custom-arm"
