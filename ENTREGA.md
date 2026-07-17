# Entrega académica — Better Router Adaptive

## Integrantes

1. Rodolfo Fritz
2. Benjamín Cerda
3. Felipe Friz

## Estado

**CÓDIGO_LISTO_PARA_EVALUACIÓN**

Este repositorio público contiene el código fuente, configuraciones, pruebas, documentación metodológica, resultados agregados e instrucciones necesarias para evaluar la solución. El póster científico en PDF se entrega por separado y no se genera desde GitHub.

## Alcance obligatorio

La solución principal implementa y compara dos algoritmos de inteligencia artificial:

1. **XGBoost**: regresión supervisada de utilidad para seleccionar un modelo por consulta.
2. **LinUCB**: bandit contextual para selección secuencial de modelos.

Se utilizan Better Rules Proxy, brazos fijos y Oracle offline como referencias experimentales.

La evaluación obligatoria no depende de EvoCascade ni de una interfaz gráfica. Cualquier trabajo de bonus posterior se desarrollará en una rama nueva, después de cerrar esta versión.

## Archivos principales

- Guía para el profesor: `EVALUACION_PROFESOR.md`.
- Código fuente: `src/better_router_adaptive/`.
- Configuraciones: `config/`.
- Pruebas unitarias y de integración: `tests/`.
- Resultados agregados: `artifacts/public/step7-real/`.
- Metodología, análisis y reproducibilidad: `docs/`.
- Dependencias: `pyproject.toml` y `requirements.txt`.
- Verificador de estructura: `scripts/verify_professor_submission.py`.
- Póster científico: archivo PDF entregado por separado.

## Comando recomendado de evaluación

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
make professor-check
```

En Windows sin `make`:

```powershell
python -m pytest -q --cov=better_router_adaptive --cov-branch --cov-report=term-missing --cov-fail-under=85
ruff check .
ruff format --check .
mypy src tests
python -m build
python scripts/verify_professor_submission.py
```

## Resultado científico principal

XGBoost y LinUCB no superan significativamente a Better Rules Proxy con las características pre-inferencia utilizadas. El Oracle offline mejora la utilidad en aproximadamente `+0,068`, demostrando que existe margen para un router por consulta, aunque las señales actuales todavía no permiten capturarlo.

Este resultado es válido porque cuantifica la diferencia entre políticas aprendidas, una línea base determinista y la mejor selección offline disponible en el benchmark.

## Cumplimiento de la pauta

| Requisito | Evidencia |
|---|---|
| Procesamiento del dataset | Descarga verificada, conversión, limpieza, normalización y características |
| Dos algoritmos de IA | XGBoost y LinUCB |
| Dos o más métricas | Utilidad, calidad, costo, error, regret e IC 95 % |
| Comparación experimental | Baseline, XGBoost, LinUCB, brazos fijos y Oracle |
| Análisis de resultados | `docs/RESULTS.md` |
| Decisiones metodológicas justificadas | `docs/METHODOLOGY.md` y documentos por etapa |
| Programación modular | Paquete `src/better_router_adaptive/` |
| Código ejecutable | CI, pruebas, build y smoke tests |
| Póster PDF | Entregado por separado |

## Restricción de datos

El dataset original y los datos procesados fila por fila no se distribuyen públicamente porque la tarjeta de RouterBench no declara una licencia explícita de redistribución. El repositorio conserva scripts, procedencia, checksums, configuraciones y resultados agregados para permitir su regeneración desde la fuente oficial.

## Bonus

El bonus queda fuera de este cierre. Se desarrollará posteriormente en una rama separada para no alterar la versión que evaluará el profesor.
