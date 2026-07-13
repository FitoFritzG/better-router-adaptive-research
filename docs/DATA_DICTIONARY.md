# Diccionario de datos — esquema canónico v1.0.0

## Unidad de observación

Una fila representa el resultado observado de un modelo candidato para una consulta. La clave primaria compuesta es:

```text
(prompt_id, model_id)
```

## Variables

| Variable | Tipo lógico | Admite nulo | Uso | Regla de validación |
|---|---|---:|---|---|
| `prompt_id` | texto | No | Identidad de consulta | No vacío y estable |
| `prompt_text` | texto | No | Entrada del router | Consistente dentro de cada `prompt_id` |
| `dataset` | texto | No | Procedencia del benchmark | Consistente dentro del prompt |
| `task_group` | categoría | No | Estratificación | `coding`, `mathematics`, `reasoning`, `general` |
| `model_id` | texto | No | Brazo de decisión | Un resultado por prompt |
| `quality` | real | No | Variable objetivo | Finita, `0 ≤ quality ≤ 1` |
| `input_tokens` | entero | Sí | Costo/caracterización | No negativo |
| `output_tokens` | entero | Sí | Costo/caracterización | No negativo |
| `cost_usd` | real | Sí | Penalización económica | Finito y no negativo |
| `latency_ms` | real | Sí | Penalización temporal | Finita y no negativa |
| `success` | booleano | No | Error de ejecución | `True` o `False` |
| `data_origin` | texto | No | Auditoría | Identificador explícito de procedencia |

## Campos no conservados

El esquema no almacena la respuesta textual de cada modelo. Esto reduce volumen, riesgos de licencia y exposición de contenido que no es necesario para el enrutamiento offline.

## Ausencias conocidas

RouterBench proporciona calidad y costo, pero la versión seleccionada puede no contener latencia ni conteos de tokens. Estos valores permanecen nulos; no se imputan silenciosamente en el Paso 3.

## Integridad

- Los duplicados exactos se eliminan y se contabilizan.
- Dos filas distintas con la misma clave generan error.
- Un `prompt_id` debe mantener el mismo texto, dataset y grupo de tarea.
- El experimento primario conserva únicamente prompts con los cuatro brazos seleccionados.
