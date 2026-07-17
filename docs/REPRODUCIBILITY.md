# Reproducibilidad

## Objetivo

Este documento describe cómo reconstruir el entorno, verificar el código y reproducir la evaluación principal de XGBoost y LinUCB sin depender de servicios privados de Better Router.

## Entorno soportado

- Python 3.12 o 3.13.
- Linux, macOS o Windows.
- Dependencias declaradas en `pyproject.toml`.
- XGBoost fijado en la versión `3.3.0` para reproducir los artefactos publicados.

## Instalación limpia

```bash
git clone https://github.com/FitoFritzG/better-router-adaptive-research.git
cd better-router-adaptive-research
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Activación en Linux/macOS:

```bash
source .venv/bin/activate
```

Activación en Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## Verificación del repositorio

```bash
python scripts/verify_professor_submission.py
python -m pytest -q \
  --cov=better_router_adaptive \
  --cov-branch \
  --cov-report=term-missing \
  --cov-fail-under=85
ruff check .
ruff format --check .
mypy src tests
python -m build
```

## Datos

El artefacto de RouterBench se descarga desde su fuente oficial mediante:

```bash
python -m better_router_adaptive.data.download \
  --destination data/raw/routerbench_0shot.pkl
```

El descargador verifica la revisión, el tamaño y el SHA-256 documentados en el código y en `data/README.md`.

El archivo `pickle` debe convertirse en un entorno desechable y sin credenciales:

```bash
python -m better_router_adaptive.data.convert \
  --input data/raw/routerbench_0shot.pkl \
  --output data/interim/routerbench_0shot_canonical.csv.gz \
  --expected-sha256 ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d
```

Posteriormente se ejecutan validación, limpieza y preparación siguiendo las interfaces CLI:

```bash
python -m better_router_adaptive.data.pipeline --help
python -m better_router_adaptive.prepare --help
python -m better_router_adaptive.baselines --help
python -m better_router_adaptive.learn --help
python -m better_router_adaptive.evaluate --help
```

## Evaluación principal

La evaluación final usa:

- cuatro modelos candidatos;
- cinco semillas;
- división agrupada por `prompt_id`;
- normalización calculada solo con entrenamiento;
- selección de hiperparámetros con validación;
- test final sin reutilizar para ajuste;
- bootstrap pareado por prompt.

Comando de evaluación:

```bash
python -m better_router_adaptive.evaluate \
  --input data/processed/step3-real/routerbench_canonical_clean.csv.gz \
  --output-directory artifacts/runs/step7-real \
  --bootstrap-samples 2000 \
  --evidence-label "RouterBench 0-shot — datos reales"
```

## Evidencia pública

Debido a las restricciones de redistribución, el repositorio no contiene el dataset fila por fila. Para facilitar la revisión se incluyen resultados agregados y figuras en:

```text
artifacts/public/step7-real/
```

## Determinismo y recursos

- Cada semilla se evalúa en un proceso independiente.
- Los hilos de BLAS, OpenMP y XGBoost se limitan para evitar sobreasignación.
- XGBoost se ejecuta con una versión fijada.
- Las configuraciones y semillas se versionan.
- Las pruebas comprueban ausencia de fuga entre entrenamiento, validación y prueba.

## Alcance del bonus

La extensión EvoCascade-Ideal no es necesaria para reproducir la entrega obligatoria. Cualquier desarrollo adicional del bonus debe realizarse después del cierre, en una rama separada.
