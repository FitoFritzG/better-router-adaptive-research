# Paso 6 — Routers aprendidos: XGBoost y LinUCB

## Alcance implementado

- router supervisado XGBoost: un regresor de utilidad por brazo, ruteo por
  argmax de la utilidad predicha;
- router LinUCB disjunto: bandit contextual con término de exploración UCB,
  entrenado por replay prequential (elegir → observar → actualizar) sobre un
  barajado con semilla de los prompts de train;
- escalado min-max de características **anclado a train** (reutiliza la
  disciplina del Paso 5);
- selección de hiperparámetros **exclusivamente en validación**;
- curva de regret acumulado del LinUCB frente al mejor brazo realizado;
- CLI del Paso 6 (`python -m better_router_adaptive.learn`);
- pruebas unitarias e integración (determinismo por semilla, dominancia del
  Oracle, invariantes del regret).

## Diseño de los algoritmos

### XGBoost (supervisado, offline)

Para cada brazo se entrena un `XGBRegressor` que predice la utilidad del
Paso 5 a partir de las características pre-inferencia del Paso 4. En
inferencia se elige el brazo con mayor utilidad predicha (empates: primer
brazo en orden alfabético). Grilla bloqueada: `max_depth ∈ {2,3}`,
`n_estimators ∈ {50,150}`, `learning_rate ∈ {0.1,0.3}` (8 combinaciones). La
ganadora es la de mayor utilidad realizada media en validación; test no
participa en la selección.

### LinUCB disjunto (bandit contextual, online)

Cada brazo mantiene `A_a = λI + Σ x xᵀ` y `b_a = Σ r x` con intercepto en el
contexto. En el replay elige por `θᵀx + α·√(xᵀA⁻¹x)` y solo observa la
recompensa del brazo elegido — RouterBench permite este replay porque todos
los brazos fueron ejecutados offline. `α ∈ {0.1, 0.5, 1.0, 2.0}` se elige en
validación con ruteo greedy (sin bono de exploración, sin actualización). El
regret por paso se mide contra el mejor brazo realizado del prompt.

## Artefactos generados

```text
router_results.csv    # mismas métricas y esquema que baseline_results.csv
linucb_regret.csv     # step, prompt_id, regret, cumulative_regret
xgboost_search.json   # grilla completa con puntajes de validación y elegida
linucb_search.json    # alphas con puntajes de validación y elegida
step6_manifest.json   # semilla, brazos, características y procedencia
```

`router_results.csv` comparte esquema con `baseline_results.csv` del Paso 5,
de modo que el Paso 7 puede concatenarlos directamente para la comparación
final.

## Evidencia versionada

Los archivos en `artifacts/examples/step6/` provienen del fixture sintético y
no constituyen resultados del estudio.

## Próximo paso

El Paso 7 ejecutará las cinco semillas bloqueadas, calculará intervalos de
confianza por bootstrap pareado remuestreando por `prompt_id`, y generará las
figuras y tablas del informe IEEE y del póster desde artefactos.
