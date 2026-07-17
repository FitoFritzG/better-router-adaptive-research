# Entrega académica final — Better Router Adaptive

## Integrantes

1. Rodolfo Fritz
2. Benjamín Cerda
3. Felipe Friz

## Estado

**ENTREGA_FINAL_LISTA_PARA_EVALUACIÓN**

El repositorio contiene el póster científico, código fuente, configuraciones, pruebas, documentación metodológica, resultados agregados y archivos necesarios para ejecutar y evaluar la solución.

## Alcance obligatorio

La solución principal implementa y compara dos algoritmos de inteligencia artificial:

1. **XGBoost**: regresión supervisada de utilidad para seleccionar un modelo por consulta.
2. **LinUCB**: bandit contextual para selección secuencial de modelos.

Better Rules Proxy, brazos fijos y Oracle offline se utilizan como referencias experimentales. `EvoCascade-Ideal` se conserva como iniciativa adicional y se presenta explícitamente como una cota experimental dependiente de un verificador perfecto simulado.

## Archivos principales

- Póster científico PDF: `paper/poster/poster_better_router.pdf`.
- Paquete final: `delivery/Better_Router_Adaptive_Entrega_Final.zip`.
- Checksums: `delivery/SHA256SUMS.txt`.
- Guía para el profesor: `EVALUACION_PROFESOR.md`.
- Código fuente: `src/better_router_adaptive/`.
- Configuraciones: `config/`.
- Pruebas unitarias y de integración: `tests/`.
- Resultados agregados: `artifacts/public/step7-real/` y `artifacts/public/evocascade-ideal-verifier/`.
- Metodología, análisis y reproducibilidad: `docs/`.
- Dependencias: `pyproject.toml` y `requirements.txt`.
- Verificador de estructura: `scripts/verify_professor_submission.py`.

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

EvoCascade-Ideal obtiene una mejora offline bajo un verificador perfecto simulado. Este resultado se documenta como bonus exploratorio y no como desempeño directamente desplegable.

## Cumplimiento de la pauta

| Requisito | Evidencia |
|---|---|
| Procesamiento del dataset | Descarga verificada, conversión, limpieza, normalización y características |
| Dos algoritmos de IA | XGBoost y LinUCB |
| Dos o más métricas | Utilidad, calidad, costo, error, regret e IC 95 % |
| Comparación experimental | Baseline, XGBoost, LinUCB, brazos fijos y Oracle |
| Análisis de resultados | `docs/RESULTS.md` y `LATEST_STUDY.md` |
| Decisiones metodológicas justificadas | `docs/METHODOLOGY.md` y documentos por etapa |
| Programación modular | Paquete `src/better_router_adaptive/` |
| Código ejecutable | CI, pruebas, build y smoke tests |
| Póster PDF | `paper/poster/poster_better_router.pdf` |
| Bonus | Hiperparámetros, evaluación multi-semilla y EvoCascade-Ideal |

## Restricción de datos

El dataset original y los datos procesados fila por fila no se distribuyen públicamente porque la tarjeta de RouterBench no declara una licencia explícita de redistribución. El repositorio conserva scripts, procedencia, checksums, configuraciones y resultados agregados para permitir su regeneración desde la fuente oficial.

## Cierre

GitHub Actions genera el PDF y el ZIP final, verifica su integridad mediante SHA-256 y elimina las ramas remotas distintas de `main`. El paquete excluye entornos virtuales, cachés, secretos, pickles y datasets sin permiso de redistribución.
