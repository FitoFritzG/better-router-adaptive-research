# Paso 7 — Evaluación multi-semilla con bootstrap pareado

## Alcance implementado

- ejecución completa de los Pasos 4–6 en memoria para cada una de las cinco
  semillas bloqueadas (`42, 123, 2026, 31415, 271828`);
- evaluación de las ocho políticas en el split de prueba de cada semilla:
  `oracle`, `better-rules-proxy`, `xgboost`, `linucb` y los cuatro brazos fijos;
- bootstrap pareado remuestreando **por `prompt_id`**: cada remuestra evalúa
  todas las políticas sobre exactamente los mismos prompts;
- tablas y figuras generadas desde artefactos, nunca escritas a mano;
- CLI del Paso 7 (`python -m better_router_adaptive.evaluate`).

## Selección de brazos (real, definida antes de ver resultados de políticas)

El artefacto RouterBench verificado contiene 11 modelos. Se seleccionaron
cuatro niveles de costo con calidad creciente, excluyendo modelos dominados
(peor calidad media a costo igual o mayor):

| Rol | Modelo | Calidad media | Costo medio [USD] |
|---|---|---|---|
| Rápido | `mistralai/mistral-7b-chat` | 0.306 | 0.00005 |
| Equilibrado | `mistralai/mixtral-8x7b-chat` | 0.547 | 0.00013 |
| Razonamiento | `zero-one-ai/Yi-34B-Chat` | 0.647 | 0.00019 |
| Premium | `gpt-4-1106-preview` | 0.781 | 0.00329 |

La latencia del artefacto es 100 % nula, por lo que el término de latencia de
la utilidad queda **deshabilitado explícitamente** (véase el Paso 5); los
pesos restantes no se renormalizan.

## Metodología del bootstrap

1. Para cada semilla, cada política produce una utilidad realizada por prompt
   del split de prueba.
2. Las utilidades se agregan por `prompt_id` (promedio sobre las semillas en
   las que el prompt cayó en prueba).
3. Se remuestrean los `prompt_id` con reemplazo (2000 remuestras por defecto)
   y en cada remuestra se calcula la media de **todas** las políticas sobre
   los mismos prompts — eso hace el intervalo pareado.
4. Se reportan IC 95 % percentiles para la media de cada política y para su
   diferencia contra `better-rules-proxy`.

Una diferencia contra la línea base cuyo IC 95 % no cruza cero se considera
evidencia de mejora (o deterioro) bajo la función de utilidad bloqueada.

## Artefactos generados

```text
evaluation_per_seed.csv        # métricas por semilla y política (test)
evaluation_summary.csv         # medias, IC 95 % y diferencias vs. baseline
linucb_regret_curves.csv       # regret acumulado por semilla
figures/comparacion_politicas.svg
figures/regret_linucb.svg
figures/frontera_calidad_costo.svg
step7_manifest.json
```

## Reproducir con datos reales

```bash
# 1. Descargar y verificar (SHA-256 fijado)
python -m better_router_adaptive.data.download

# 2. Convertir el pickle verificado (ejecutar en entorno desechable)
python -m better_router_adaptive.data.convert \
  --input data/raw/routerbench_0shot.pkl \
  --output data/interim/routerbench_0shot_canonical.csv.gz \
  --expected-sha256 ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d

# 3. Limpiar restringiendo a los cuatro brazos del experimento
python -m better_router_adaptive.data.pipeline \
  --input data/interim/routerbench_0shot_canonical.csv.gz \
  --output-directory data/processed/step3-real \
  --model-arm "mistralai/mistral-7b-chat" \
  --model-arm "mistralai/mixtral-8x7b-chat" \
  --model-arm "zero-one-ai/Yi-34B-Chat" \
  --model-arm "gpt-4-1106-preview" \
  --restrict-to-arms \
  --evidence-label "RouterBench 0-shot — datos reales"

# 4. Evaluación multi-semilla con bootstrap pareado
python -m better_router_adaptive.evaluate \
  --input data/processed/step3-real/routerbench_canonical_clean.csv.gz \
  --output-directory artifacts/runs/step7-real \
  --bootstrap-samples 2000 \
  --evidence-label "RouterBench 0-shot — datos reales"
```

Los datos derivados de RouterBench viven en `data/` (ignorado por Git) y no
se redistribuyen; solo se versionan agregados y figuras.

## Resultados verificados

| Política | Utilidad media | Diferencia vs. baseline | IC 95 % de la diferencia |
|---|---:|---:|---:|
| Oracle | 0,564086 | +0,068004 | [0,065643; 0,070490] |
| XGBoost | 0,496082 | +0,000000 | [-0,000548; 0,000576] |
| Better Rules Proxy | 0,496082 | 0 | [0; 0] |
| LinUCB | 0,495940 | -0,000142 | [-0,000475; 0,000189] |

Los routers aprendidos no muestran una mejora significativa sobre la línea base. Los archivos entregados fueron reproducidos exactamente con `xgboost==3.3.0`.

## Estabilidad de la ejecución

Cada semilla se procesa en un proceso independiente. El aislamiento evita bloqueos observados cuando múltiples ciclos XGBoost completos compartían el mismo proceso y acumulaban estado nativo. Los resultados temporales de cada proceso se agregan después en el proceso principal.

## Amenazas a la validez

- la latencia es nula en todo el artefacto;
- la selección de cuatro brazos fue exploratoria y usó estadísticas globales;
- Better Rules Proxy converge a GPT-4 bajo la escala de costo elegida;
- las características no incluyen embeddings ni semántica profunda;
- los resultados son offline y no miden drift, cuotas o fallos reales de proveedores.
