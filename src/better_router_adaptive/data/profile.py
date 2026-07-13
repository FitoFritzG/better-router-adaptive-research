"""Small, deterministic dataset-profile figures for documentation and audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def generate_profile_assets(
    frame: pd.DataFrame,
    directory: Path,
    *,
    evidence_label: str,
) -> tuple[Path, ...]:
    """Generate three figures and a machine-readable summary."""

    directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    task_counts = frame.groupby("task_group")["prompt_id"].nunique().sort_index()
    task_path = directory / "distribucion_tareas.svg"
    task_counts.plot(kind="bar")
    plt.title(f"Prompts por grupo de tarea — {evidence_label}")
    plt.xlabel("Grupo de tarea")
    plt.ylabel("Prompts únicos")
    _save_figure(task_path)
    outputs.append(task_path)

    model_summary = (
        frame.groupby("model_id", as_index=False)
        .agg(quality=("quality", "mean"), cost_usd=("cost_usd", "mean"))
        .sort_values("model_id")
    )
    quality_cost_path = directory / "calidad_costo_modelos.svg"
    plt.scatter(model_summary["cost_usd"], model_summary["quality"])
    for record in model_summary.to_dict(orient="records"):
        model_id = str(record["model_id"])
        cost_value = float(cast(float, record["cost_usd"]))
        quality_value = float(cast(float, record["quality"]))
        plt.annotate(model_id, (cost_value, quality_value))
    plt.title(f"Calidad frente a costo — {evidence_label}")
    plt.xlabel("Costo medio estimado [USD]")
    plt.ylabel("Calidad media normalizada")
    _save_figure(quality_cost_path)
    outputs.append(quality_cost_path)

    latency_path = directory / "latencia_modelos.svg"
    latency = frame.groupby("model_id")["latency_ms"].mean().dropna().sort_index()
    if latency.empty:
        plt.figure()
        plt.text(0.5, 0.5, "Latencia no disponible", ha="center", va="center")
        plt.axis("off")
    else:
        latency.plot(kind="bar")
        plt.xlabel("Brazo de modelo")
        plt.ylabel("Latencia media [ms]")
    plt.title(f"Latencia por modelo — {evidence_label}")
    _save_figure(latency_path)
    outputs.append(latency_path)

    summary = {
        "evidence_label": evidence_label,
        "rows": len(frame),
        "prompts": int(frame["prompt_id"].nunique()),
        "models": int(frame["model_id"].nunique()),
        "task_groups": int(frame["task_group"].nunique()),
        "datasets": sorted(frame["dataset"].astype(str).unique().tolist()),
    }
    summary_path = directory / "resumen_perfil.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs.append(summary_path)
    return tuple(outputs)
