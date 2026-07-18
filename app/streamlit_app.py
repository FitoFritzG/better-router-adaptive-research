"""Public Streamlit dashboard for the Better Router Adaptive research project."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import streamlit as st

try:
    from app.data_access import PublicResults, PublicResultsError, load_public_results
    from app.simulator import (
        DemoBundle,
        SimulationError,
        display_arm,
        simulate_prompt,
        train_demo_bundle,
    )
except ModuleNotFoundError as exc:
    if exc.name != "app":
        raise
    from data_access import PublicResults, PublicResultsError, load_public_results
    from simulator import (
        DemoBundle,
        SimulationError,
        display_arm,
        simulate_prompt,
        train_demo_bundle,
    )

ROOT: Final = Path(__file__).resolve().parents[1]
_POLICY_LABELS: Final = {
    "better-rules-proxy": "Better Rules Proxy",
    "xgboost": "XGBoost",
    "linucb": "LinUCB",
    "oracle": "Oracle offline",
    "fixed:gpt-4-1106-preview": "Brazo fijo: GPT-4",
    "fixed:mistralai/mistral-7b-chat": "Brazo fijo: Mistral 7B",
    "fixed:mistralai/mixtral-8x7b-chat": "Brazo fijo: Mixtral 8x7B",
    "fixed:zero-one-ai/Yi-34B-Chat": "Brazo fijo: Yi-34B",
}
_TASK_LABELS: Final = {
    "coding": "Programación",
    "mathematics": "Matemáticas",
    "reasoning": "Razonamiento",
    "general": "General",
}
_METRIC_LABELS: Final = {
    "mean_utility": "Utilidad media",
    "mean_quality": "Calidad media",
    "mean_cost_usd": "Costo medio (USD)",
    "error_rate": "Tasa de error",
}


@st.cache_data(show_spinner=False)
def _cached_results() -> PublicResults:
    return load_public_results(ROOT)


@st.cache_resource(show_spinner="Entrenando demostración sintética...")
def _cached_demo_bundle() -> DemoBundle:
    return train_demo_bundle(ROOT)


def _policy_label(policy: str) -> str:
    return _POLICY_LABELS.get(policy, policy)


def _policy_metric(summary: pd.DataFrame, policy: str, column: str) -> float | None:
    matches = summary.loc[summary["policy"] == policy, column]
    if matches.empty:
        return None
    return float(matches.iloc[0])


def _render_summary(results: PublicResults) -> None:
    st.subheader("Problema de ingeniería")
    st.write(
        "Un router multi-LLM debe seleccionar un modelo antes de conocer la respuesta. "
        "El estudio compara políticas que equilibran calidad, costo, latencia y errores."
    )

    left, middle, right = st.columns(3)
    left.metric(
        "Prompts evaluados",
        f"{int(results.summary['prompts'].max()):,}".replace(",", "."),
    )
    middle.metric("Algoritmos principales", "2", help="XGBoost y LinUCB")
    right.metric("Semillas experimentales", str(results.per_seed["seed"].nunique()))

    st.subheader("Metodología")
    st.markdown(
        """
- **Dataset:** RouterBench, descargado desde su fuente oficial y verificado con SHA-256.
- **Procesamiento:** conversión, validación, limpieza, normalización y
  características preinferencia.
- **Algoritmos:** XGBoost supervisado y LinUCB como bandit contextual.
- **Comparadores:** Better Rules Proxy, brazos fijos y Oracle offline.
- **Métricas:** utilidad, calidad, costo, error, regret e intervalos de confianza del 95 %.
        """
    )

    st.subheader("Resultado científico principal")
    st.info(
        "XGBoost y LinUCB no superaron significativamente a Better Rules Proxy con las "
        "características actuales. El Oracle offline sí mostró una brecha de utilidad cercana "
        "a +0,068, lo que demuestra que existe margen para mejorar la selección por consulta."
    )


def _render_results(results: PublicResults) -> None:
    st.subheader("Comparación agregada")

    baseline = _policy_metric(results.summary, "better-rules-proxy", "mean_utility")
    xgboost = _policy_metric(results.summary, "xgboost", "mean_utility")
    linucb = _policy_metric(results.summary, "linucb", "mean_utility")
    oracle = _policy_metric(results.summary, "oracle", "mean_utility")

    columns = st.columns(4)
    columns[0].metric("Baseline", "N/D" if baseline is None else f"{baseline:.4f}")
    columns[1].metric(
        "XGBoost",
        "N/D" if xgboost is None else f"{xgboost:.4f}",
        None if xgboost is None or baseline is None else f"{xgboost - baseline:+.4f}",
    )
    columns[2].metric(
        "LinUCB",
        "N/D" if linucb is None else f"{linucb:.4f}",
        None if linucb is None or baseline is None else f"{linucb - baseline:+.4f}",
    )
    columns[3].metric(
        "Oracle",
        "N/D" if oracle is None else f"{oracle:.4f}",
        None if oracle is None or baseline is None else f"{oracle - baseline:+.4f}",
    )

    policies = results.summary["policy"].astype(str).tolist()
    default_policies = [
        policy
        for policy in ("better-rules-proxy", "xgboost", "linucb", "oracle")
        if policy in policies
    ]
    selected = st.multiselect(
        "Políticas visibles",
        options=policies,
        default=default_policies,
        format_func=_policy_label,
    )
    if not selected:
        st.warning("Selecciona al menos una política para visualizar los resultados.")
    else:
        filtered = results.summary.loc[results.summary["policy"].isin(selected)].copy()
        filtered["Política"] = filtered["policy"].map(_policy_label)
        chart = filtered.set_index("Política")[["mean_utility"]].rename(
            columns={"mean_utility": "Utilidad media"}
        )
        st.bar_chart(chart)

        table = filtered.loc[
            :,
            [
                "Política",
                "prompts",
                "mean_utility",
                "ci95_low",
                "ci95_high",
                "diff_vs_baseline",
                "diff_ci95_low",
                "diff_ci95_high",
            ],
        ].rename(
            columns={
                "prompts": "Prompts",
                "mean_utility": "Utilidad",
                "ci95_low": "IC 95 % inferior",
                "ci95_high": "IC 95 % superior",
                "diff_vs_baseline": "Diferencia vs. baseline",
                "diff_ci95_low": "IC diferencia inferior",
                "diff_ci95_high": "IC diferencia superior",
            }
        )
        st.dataframe(table, hide_index=True, use_container_width=True)

    st.subheader("Variación entre semillas")
    metric = st.selectbox(
        "Métrica",
        options=list(_METRIC_LABELS),
        format_func=lambda value: _METRIC_LABELS[value],
    )
    seed_policies = selected or default_policies
    seed_frame = results.per_seed.loc[results.per_seed["policy"].isin(seed_policies)].copy()
    if seed_frame.empty:
        st.warning("No hay resultados por semilla para las políticas seleccionadas.")
    else:
        pivot = seed_frame.pivot(index="seed", columns="policy", values=metric).rename(
            columns=_policy_label
        )
        st.line_chart(pivot)
        st.caption(
            "Cada punto corresponde a una ejecución completa con una semilla distinta; "
            "las tablas públicas no contienen prompts ni respuestas de modelos."
        )


def _render_simulator() -> None:
    st.subheader("Simulador de enrutamiento")
    st.warning(
        "Demostración sintética: no es un resultado de RouterBench ni una decisión de "
        "producción. Los routers se entrenan con 12 prompts originales creados para pruebas."
    )
    st.write(
        "La consulta no sale del proceso de Streamlit, no se guarda y no se envía a ningún "
        "proveedor de inteligencia artificial."
    )

    with st.form("routing-simulator"):
        task_group = st.selectbox(
            "Tipo de consulta",
            options=list(_TASK_LABELS),
            format_func=lambda value: _TASK_LABELS[value],
        )
        prompt_text = st.text_area(
            "Consulta",
            placeholder="Ejemplo: explica cómo funciona un controlador PID.",
            max_chars=2_000,
            height=140,
        )
        submitted = st.form_submit_button("Simular enrutamiento", type="primary")

    if submitted:
        try:
            bundle = _cached_demo_bundle()
            result = simulate_prompt(
                bundle,
                prompt_text=prompt_text,
                task_group=task_group,
            )
        except SimulationError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # pragma: no cover - Streamlit boundary
            st.error(f"No fue posible iniciar la demostración: {exc}")
            return

        xgboost_column, linucb_column = st.columns(2)
        xgboost_column.metric("Selección de XGBoost", display_arm(result.xgboost_arm))
        linucb_column.metric("Selección de LinUCB", display_arm(result.linucb_arm))

        with st.expander("Características preinferencia utilizadas"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Caracteres": result.prompt_char_count,
                            "Palabras": result.prompt_word_count,
                            "Longitud media de palabra": result.prompt_avg_word_length,
                            "Grupo": _TASK_LABELS[task_group],
                        }
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "No se usan calidad, costo, latencia, tokens ni respuestas del modelo como "
                "características, porque esos valores solo existen después de la inferencia."
            )


def main() -> None:
    st.set_page_config(
        page_title="Better Router Adaptive",
        page_icon="🧭",
        layout="wide",
    )
    st.title("Better Router Adaptive")
    st.markdown(
        "**Bonus académico — dashboard interactivo**  \n"
        "Rodolfo Fritz · Benjamín Cerda · Felipe Friz  \n"
        "Universidad del Bío-Bío — Ingeniería Civil en Automatización"
    )

    try:
        results = _cached_results()
    except PublicResultsError as exc:
        st.error(f"No fue posible cargar los resultados públicos: {exc}")
        st.stop()

    summary_tab, results_tab, simulator_tab = st.tabs(["Resumen", "Resultados", "Simulador"])
    with summary_tab:
        _render_summary(results)
    with results_tab:
        _render_results(results)
    with simulator_tab:
        _render_simulator()

    st.divider()
    st.caption(
        "Proyecto reproducible y sin APIs externas. Código MIT; RouterBench no se redistribuye "
        "porque su tarjeta no declara una licencia explícita para los datos."
    )


if __name__ == "__main__":
    main()
