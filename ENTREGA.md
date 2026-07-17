# Entrega académica final - Better Router Adaptive

## Integrantes

1. Rodolfo Fritz
2. Benjamín Cerda
3. Felipe Friz

Universidad del Bío-Bío - Ingeniería Civil en Automatización.

## Entregables obligatorios

| Requisito | Archivo o directorio | Estado |
|---|---|---|
| Póster científico PDF | `paper/poster/poster_evocascade_ideal.pdf` | Generado y verificado automáticamente |
| Código fuente Python | `src/better_router_adaptive/` | Incluido, modular y comentado |
| Archivos de ejecución | `pyproject.toml`, `requirements.txt`, `config/`, `Makefile` | Incluidos |
| Procesamiento de datos | `src/better_router_adaptive/data/` | Incluido |
| Dos algoritmos de IA | XGBoost y LinUCB | Incluidos y comparados |
| Métricas cuantitativas | utilidad, calidad, costo, error, regret e IC 95 % | Incluidas |
| Análisis de resultados | `docs/RESULTS.md`, `LATEST_STUDY.md` | Incluido |
| Evidencia y pruebas | `tests/`, `.github/workflows/ci.yml` | Incluida |

## Bonus documentado

- análisis y selección de hiperparámetros de XGBoost y LinUCB;
- extensión adicional EvoCascade-Ideal optimizada mediante sep-CMA-ES;
- evaluación multi-semilla y bootstrap pareado;
- repositorio público con CI en Python 3.12 y 3.13.

## Resultado científico defendible

XGBoost y LinUCB no superan significativamente a Better Rules Proxy con las características pre-inferencia actuales. EvoCascade-Ideal mejora la utilidad en el experimento offline, pero emplea un verificador perfecto simulado con ground truth; por tanto, representa una cota superior experimental y no desempeño directamente desplegable.

## Reproducción

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q --cov=better_router_adaptive --cov-branch --cov-fail-under=85
```

La adquisición y transformación de RouterBench se ejecuta con los comandos documentados en `data/README.md` y `README.md`. El dataset procesado no se redistribuye porque su tarjeta no declara una licencia explícita.

## Paquete final

GitHub Actions genera y publica:

- `delivery/Better_Router_Adaptive_Entrega_Final.zip`;
- `delivery/SHA256SUMS.txt`;
- `paper/poster/poster_evocascade_ideal.pdf`.

El ZIP contiene el póster, código, configuración, resultados agregados, documentación, pruebas y archivos de instalación. Excluye entornos virtuales, cachés, secretos, pickles y datos fila por fila sin permiso de redistribución.
