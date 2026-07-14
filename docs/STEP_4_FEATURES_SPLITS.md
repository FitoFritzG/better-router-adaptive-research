# Paso 4 — Características sin fuga y particiones agrupadas

## Alcance implementado

- características por prompt calculadas **solo con información disponible antes de la inferencia**;
- lista explícita de columnas de resultado prohibidas (`OUTCOME_COLUMNS`) y guardia `LeakageError`;
- partición `70/15/15` con `prompt_id` como unidad de asignación (nunca por filas);
- estratificación por `task_group` con asignación determinista por semilla;
- verificación de integridad: particiones disjuntas, exhaustivas y no vacías;
- CLI del Paso 4 (`python -m better_router_adaptive.prepare`);
- manifiesto JSON con semilla, proporciones objetivo y conteos realizados;
- pruebas unitarias e integración, incluidas pruebas automáticas contra fuga.

## Características

Cada prompt produce una fila con:

| Columna | Origen | Justificación |
|---|---|---|
| `prompt_char_count` | `prompt_text` | Longitud como aproximación de complejidad |
| `prompt_word_count` | `prompt_text` | Igual que la anterior, robusta a espacios |
| `prompt_avg_word_length` | `prompt_text` | Densidad léxica simple |
| `task_group_<grupo>` | `task_group` | Codificación one-hot de la categoría de tarea |

Columnas excluidas deliberadamente por ser resultados de la inferencia:
`model_id`, `quality`, `input_tokens`, `output_tokens`, `cost_usd`, `latency_ms`, `success`.
`input_tokens` se excluye de forma conservadora porque el conteo depende del
tokenizador de cada modelo, es decir, de la decisión de enrutamiento que se
quiere aprender.

## Partición

- La unidad de asignación es `prompt_id`: las cuatro filas (brazos) de un
  prompt caen siempre en la misma partición, lo que impide compartir
  resultados del mismo prompt entre entrenamiento y evaluación.
- Dentro de cada `task_group` los prompts se ordenan, se barajan con
  `numpy.random.default_rng(seed)` y se reparten con el método de restos
  mayores, garantizando al menos un prompt por partición y por grupo.
- Un grupo con menos de 3 prompts detiene el proceso con `SplitError`.
- La semilla debe pertenecer a las semillas bloqueadas del protocolo
  (`42, 123, 2026, 31415, 271828`); cualquier otra es rechazada.

## Artefactos generados

```text
prompt_features.csv.gz            # características + split por prompt
split_assignment.csv              # prompt_id, task_group, split
routerbench_canonical_split.csv.gz# dataset canónico + columna split
step4_manifest.json               # semilla, proporciones y conteos auditables
```

## Evidencia versionada

Los archivos en `artifacts/examples/step4/` provienen del fixture sintético y
no constituyen resultados del estudio.

## Próximo paso

El Paso 5 implementará la función de utilidad
`U = 0.65·Q − 0.20·Cn − 0.10·Ln − 0.05·E`, la línea base Better Rules Proxy y
el Oracle offline sobre las particiones generadas aquí.
