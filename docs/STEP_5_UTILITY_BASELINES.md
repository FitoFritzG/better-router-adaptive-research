# Paso 5 — Función de utilidad, Better Rules Proxy y Oracle offline

## Alcance implementado

- función de utilidad bloqueada `U = 0.65·Q − 0.20·Cn − 0.10·Ln − 0.05·E`;
- normalización min-max de costo y latencia **anclada al split de train**;
- política determinista Better Rules Proxy (tabla `task_group → brazo`);
- Oracle offline como cota superior por prompt;
- políticas de brazo fijo como referencias adicionales;
- CLI del Paso 5 (`python -m better_router_adaptive.baselines`);
- resultados por política y split en CSV auditables;
- pruebas unitarias e integración, incluida una prueba de que valores
  extremos fuera de train no alteran la normalización.

## Función de utilidad

Para cada resultado `(prompt, modelo)`:

```text
U = 0.65·quality − 0.20·cost_norm − 0.10·latency_norm − 0.05·error
```

- `quality` ya está en `[0, 1]` por el contrato canónico.
- `cost_norm` y `latency_norm` usan min-max **calculado solo en train**; en
  validación y prueba los valores fuera de rango se recortan a `[0, 1]`. Así
  las particiones de evaluación nunca influyen en la escala con que se las
  evalúa.
- `error = 1 − success`.
- Si una columna (p. ej. latencia, ausente en RouterBench) es completamente
  nula en train, su término se **deshabilita de forma explícita** y queda
  registrado en `normalization_stats.json`. Nulos parciales se rechazan: no
  hay imputación silenciosa.

## Políticas de referencia

| Política | Descripción | Información que usa |
|---|---|---|
| `oracle` | Brazo con mayor utilidad realizada por prompt | Resultados observados (cota superior offline) |
| `better-rules-proxy` | Tabla fija `task_group → brazo` con mejor utilidad media en train | Solo agregados de train |
| `fixed:<brazo>` | Siempre el mismo brazo | Ninguna |

Los empates se resuelven por `model_id` ascendente para que toda selección
sea determinista y reproducible.

## Artefactos generados

```text
utility_dataset.csv.gz     # dataset canónico + términos normalizados + utilidad
normalization_stats.json   # anclas min-max de train (auditables)
better_rules_proxy.json    # tabla de reglas ajustada en train
baseline_results.csv       # métricas por política y split
step5_manifest.json        # pesos, anclas, políticas y procedencia
```

Métricas reportadas por política y split: `mean_utility`, `mean_quality`,
`mean_cost_usd`, `error_rate` y número de prompts.

## Evidencia versionada

Los archivos en `artifacts/examples/step5/` provienen del fixture sintético y
no constituyen resultados del estudio.

## Próximo paso

El Paso 6 entrenará los routers aprendidos: XGBoost (ajustado exclusivamente
en validación) y LinUCB disjunto con replay prequential, usando las
características del Paso 4 y la utilidad definida aquí.
