# Revisión técnica — Bonus Streamlit

## Estado

**APPROVED_FOR_DEPLOYMENT_AUTHORIZATION**

La revisión se realizó contra la especificación `docs/superpowers/specs/2026-07-17-streamlit-dashboard-bonus-design.md` y el plan `docs/superpowers/plans/2026-07-17-streamlit-dashboard-bonus.md`.

## Alcance revisado

- dashboard Streamlit con resumen, resultados y simulador;
- carga validada de resultados agregados públicos;
- entrenamiento didáctico de XGBoost y LinUCB con el fixture sintético original;
- pruebas unitarias y smoke test de Streamlit;
- configuración para Streamlit Community Cloud;
- documentación de despliegue y privacidad;
- aislamiento entre la entrega obligatoria y el bonus.

## Revisión de arquitectura

### Carga de resultados

`app/data_access.py` concentra la lectura y validación de los dos CSV públicos. La capa de presentación no accede directamente al sistema de archivos ni inventa valores cuando falta un artefacto.

Se validan:

- existencia de archivos;
- sintaxis CSV;
- columnas obligatorias;
- presencia de al menos una fila;
- orden determinista de políticas y semillas.

### Simulador

`app/simulator.py` no reimplementa los algoritmos. Utiliza:

- `FeatureScaler`;
- `XGBoostRouter`;
- `LinUCBRouter`;
- `build_prompt_features`;
- `compute_normalization_stats`;
- `compute_utility`.

El entrenamiento se limita a `tests/fixtures/routerbench_sample.csv`, que contiene datos sintéticos originales. La interfaz distingue de manera visible este demo de los resultados reales de RouterBench.

### Presentación

`app/streamlit_app.py` se limita a:

- mostrar contexto metodológico;
- representar resultados agregados;
- filtrar políticas y métricas;
- ejecutar el simulador bajo una acción explícita del usuario;
- comunicar límites científicos y de privacidad.

La aplicación conserva los datos científicos como fuente de verdad en `artifacts/public/step7-real/`.

## Revisión de seguridad y privacidad

No se encontraron:

- solicitudes HTTP;
- uso de `requests`, `httpx` o SDK de proveedores;
- variables de API;
- lectura de `st.secrets`;
- escritura de prompts en disco;
- bases de datos o telemetría propia;
- datasets de RouterBench fila por fila;
- ejecución de código ingresado por el usuario.

El prompt se conserva únicamente en memoria durante la ejecución de Streamlit y se transforma en características preinferencia.

## Frontera científica

La interfaz presenta dos categorías diferentes:

1. **Resultados experimentales reales:** métricas agregadas de cinco semillas obtenidas desde los CSV públicos.
2. **Demostración sintética:** selecciones de XGBoost y LinUCB entrenados con 12 prompts de prueba.

El simulador no se presenta como:

- inferencia sobre RouterBench;
- comparación de proveedores reales;
- predicción de calidad;
- política preparada para producción.

## Hallazgos corregidos durante la revisión

1. `tests/app/__init__.py` ocultaba el paquete raíz `app`; fue eliminado.
2. La carga de CSV necesitaba validación explícita de columnas y archivos vacíos; se añadieron pruebas.
3. El simulador necesitaba validación de longitud y grupos; se añadieron errores explícitos.
4. Ruff detectó formato y expresiones regulares ambiguas; fueron corregidas con la configuración del proyecto.
5. El tipado del bonus y el tipado obligatorio interferían por sus dependencias distintas; la CI ahora usa alcances separados sin rebajar controles.
6. La versión de Streamlit quedó fijada en `1.59.2` para reproducibilidad.

## Pruebas

El job `bonus-app` verifica:

- carga de resultados;
- entrenamiento y determinismo del simulador;
- validación de entradas;
- arranque de la interfaz mediante `streamlit.testing.v1.AppTest`;
- Ruff lint;
- Ruff format;
- mypy estricto sobre `src`, `tests` y `app`.

La matriz original continúa verificando la entrega obligatoria en Python 3.12 y 3.13.

## Riesgos residuales

- Streamlit Community Cloud puede suspender aplicaciones inactivas y reconstruirlas al acceder.
- El tiempo del primer arranque incluye la carga de dependencias y el entrenamiento del fixture sintético.
- El enlace público requiere autorización manual del propietario del repositorio.
- Cambios futuros en resultados o dependencias deben pasar nuevamente la CI completa.

## Veredicto

El bonus cumple el objetivo de incorporar una interfaz gráfica funcional, documentada y verificable. Está listo para fusionarse después de que el commit final pase la matriz completa y, posteriormente, para la autorización mínima de despliegue en Streamlit Community Cloud.
