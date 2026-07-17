# Streamlit Dashboard Bonus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir y documentar un dashboard Streamlit público con resultados interactivos y un simulador didáctico que ejecute las implementaciones reales de XGBoost y LinUCB sobre datos sintéticos originales.

**Architecture:** La lógica se separa en carga validada de resultados (`app/data_access.py`), entrenamiento/inferencia didáctica sin Streamlit (`app/simulator.py`) y presentación (`app/streamlit_app.py`). El dashboard consume únicamente resultados agregados públicos; el simulador entrena en memoria con el fixture sintético y nunca llama APIs externas.

**Tech Stack:** Python 3.12, Streamlit 1.x, pandas, NumPy, XGBoost, scikit-learn, pytest, Streamlit AppTest, Ruff y mypy.

## Global Constraints

- Mantener `main` como fuente de verdad de la entrega obligatoria; implementar en `bonus/streamlit-dashboard-v1`.
- No incluir RouterBench ni datos procesados fila por fila.
- No realizar llamadas HTTP ni usar API keys.
- El simulador debe estar rotulado como demostración sintética, no como resultado experimental.
- Archivo de despliegue: `app/streamlit_app.py`.
- Python de despliegue: 3.12.
- Longitud permitida de consulta: entre 5 y 2.000 caracteres después de `strip()`.
- Grupos permitidos: `coding`, `mathematics`, `reasoning`, `general`.
- La CI obligatoria existente no puede degradarse.

---

### Task 1: Carga validada de resultados públicos

**Files:**
- Create: `app/__init__.py`
- Create: `app/data_access.py`
- Create: `tests/app/__init__.py`
- Create: `tests/app/test_data_access.py`

**Interfaces:**
- Consumes: `artifacts/public/step7-real/evaluation_summary.csv`, `artifacts/public/step7-real/evaluation_per_seed.csv`.
- Produces: `PublicResults` y `load_public_results(root: Path) -> PublicResults`.

- [ ] **Step 1: Escribir pruebas que fallen**

```python
from pathlib import Path

import pandas as pd
import pytest

from app.data_access import PublicResultsError, load_public_results


def test_load_public_results_returns_sorted_copies(tmp_path: Path) -> None:
    root = _write_valid_results(tmp_path)
    results = load_public_results(root)
    assert list(results.summary["policy"]) == sorted(results.summary["policy"])
    assert set(results.per_seed["seed"]) == {42, 123}


def test_load_public_results_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PublicResultsError, match="evaluation_summary.csv"):
        load_public_results(tmp_path)


def test_load_public_results_rejects_missing_columns(tmp_path: Path) -> None:
    root = _write_valid_results(tmp_path)
    pd.DataFrame({"policy": ["xgboost"]}).to_csv(
        root / "artifacts/public/step7-real/evaluation_summary.csv", index=False
    )
    with pytest.raises(PublicResultsError, match="missing columns"):
        load_public_results(root)
```

- [ ] **Step 2: Ejecutar y confirmar rojo**

Run: `python -m pytest -q tests/app/test_data_access.py`

Expected: FAIL con `ModuleNotFoundError: No module named 'app.data_access'`.

- [ ] **Step 3: Implementar carga y validación mínima**

```python
@dataclass(frozen=True, slots=True)
class PublicResults:
    summary: pd.DataFrame
    per_seed: pd.DataFrame


class PublicResultsError(ValueError):
    pass


def load_public_results(root: Path) -> PublicResults:
    base = root / "artifacts/public/step7-real"
    summary = _read_required_csv(base / "evaluation_summary.csv", SUMMARY_COLUMNS)
    per_seed = _read_required_csv(base / "evaluation_per_seed.csv", PER_SEED_COLUMNS)
    return PublicResults(
        summary=summary.sort_values("policy", kind="mergesort").reset_index(drop=True),
        per_seed=per_seed.sort_values(["seed", "policy"], kind="mergesort").reset_index(drop=True),
    )
```

`_read_required_csv` debe envolver `FileNotFoundError`/`pd.errors.ParserError`, comprobar columnas y devolver una copia.

- [ ] **Step 4: Ejecutar pruebas y calidad**

Run:

```bash
python -m pytest -q tests/app/test_data_access.py
ruff check app/data_access.py tests/app/test_data_access.py
ruff format --check app/data_access.py tests/app/test_data_access.py
mypy app/data_access.py tests/app/test_data_access.py
```

Expected: todos PASS.

- [ ] **Step 5: Commit**

```bash
git add app/__init__.py app/data_access.py tests/app/__init__.py tests/app/test_data_access.py
git commit -m "feat: add validated public result loading"
```

---

### Task 2: Simulador didáctico con routers reales

**Files:**
- Create: `app/simulator.py`
- Create: `tests/app/test_simulator.py`

**Interfaces:**
- Consumes: `tests/fixtures/routerbench_sample.csv`, `config/experiment.yaml`, `build_prompt_features`, `FeatureScaler`, `XGBoostRouter`, `LinUCBRouter`, `compute_normalization_stats`, `compute_utility`.
- Produces: `DemoBundle`, `SimulationResult`, `train_demo_bundle(root: Path)`, `simulate_prompt(...)` y `display_arm(...)`.

- [ ] **Step 1: Escribir pruebas que fallen**

```python
def test_train_demo_bundle_uses_four_synthetic_arms(repo_root: Path) -> None:
    bundle = train_demo_bundle(repo_root)
    assert bundle.arms == ("arm-fast", "arm-balanced", "arm-reasoning", "arm-premium")


def test_simulate_prompt_is_deterministic(repo_root: Path) -> None:
    bundle = train_demo_bundle(repo_root)
    first = simulate_prompt(bundle, prompt_text="Explica un algoritmo de búsqueda.", task_group="coding")
    second = simulate_prompt(bundle, prompt_text="Explica un algoritmo de búsqueda.", task_group="coding")
    assert first == second


@pytest.mark.parametrize("prompt", ["", "   ", "hola", "x" * 2001])
def test_simulate_prompt_validates_length(repo_root: Path, prompt: str) -> None:
    with pytest.raises(SimulationError):
        simulate_prompt(train_demo_bundle(repo_root), prompt_text=prompt, task_group="general")


def test_simulate_prompt_rejects_unknown_group(repo_root: Path) -> None:
    with pytest.raises(SimulationError, match="task group"):
        simulate_prompt(train_demo_bundle(repo_root), prompt_text="Consulta válida", task_group="unknown")
```

- [ ] **Step 2: Ejecutar y confirmar rojo**

Run: `python -m pytest -q tests/app/test_simulator.py`

Expected: FAIL con `ModuleNotFoundError: No module named 'app.simulator'`.

- [ ] **Step 3: Implementar entrenamiento reproducible**

Implementar:

```python
@dataclass(slots=True)
class DemoBundle:
    task_groups: tuple[str, ...]
    arms: tuple[str, ...]
    scaler: FeatureScaler
    xgboost: XGBoostRouter
    linucb: LinUCBRouter


@dataclass(frozen=True, slots=True)
class SimulationResult:
    xgboost_arm: str
    linucb_arm: str
    prompt_char_count: int
    prompt_word_count: int
    prompt_avg_word_length: float
```

`train_demo_bundle` debe:

1. leer el fixture;
2. convertir `success` a bool de forma estricta;
3. cargar pesos desde `config/experiment.yaml`;
4. construir características por prompt;
5. calcular utilidad con normalización ajustada al fixture sintético;
6. pivotar utilidad a matriz `(n_prompts, n_arms)` conservando el orden de brazos;
7. ajustar `FeatureScaler` con `feature_column_names`;
8. entrenar `XGBoostRouter` con `XGBoostConfig(2, 50, 0.1)` y semilla 42;
9. entrenar `LinUCBRouter(alpha=0.5)` mediante `replay`.

`simulate_prompt` debe construir una fila canónica mínima, extraer características, escalar y llamar `route` en ambos routers.

- [ ] **Step 4: Ejecutar pruebas y calidad**

Run:

```bash
python -m pytest -q tests/app/test_simulator.py
ruff check app/simulator.py tests/app/test_simulator.py
ruff format --check app/simulator.py tests/app/test_simulator.py
mypy app/simulator.py tests/app/test_simulator.py
```

Expected: todos PASS y la prueba determinista no bloquea XGBoost.

- [ ] **Step 5: Commit**

```bash
git add app/simulator.py tests/app/test_simulator.py
git commit -m "feat: add synthetic XGBoost and LinUCB simulator"
```

---

### Task 3: Interfaz Streamlit y smoke test

**Files:**
- Create: `app/streamlit_app.py`
- Create: `app/requirements.txt`
- Create: `.streamlit/config.toml`
- Create: `tests/app/test_streamlit_app.py`

**Interfaces:**
- Consumes: `load_public_results`, `train_demo_bundle`, `simulate_prompt`, `display_arm`.
- Produces: aplicación ejecutable con `streamlit run app/streamlit_app.py`.

- [ ] **Step 1: Escribir smoke test que falle**

```python
streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_core_sections() -> None:
    app = AppTest.from_file("app/streamlit_app.py", default_timeout=30).run()
    assert not app.exception
    titles = [item.value for item in app.title]
    assert "Better Router Adaptive" in titles
    markdown = "\n".join(item.value for item in app.markdown)
    assert "demostración sintética" in markdown.lower()
    assert "Rodolfo Fritz" in markdown
```

- [ ] **Step 2: Ejecutar y confirmar rojo**

Run: `python -m pytest -q tests/app/test_streamlit_app.py`

Expected: SKIP sin extra `app`; con `python -m pip install -e ".[dev,app]"`, FAIL porque no existe `app/streamlit_app.py`.

- [ ] **Step 3: Implementar la aplicación**

La aplicación debe:

- llamar `st.set_page_config(page_title="Better Router Adaptive", page_icon="🧭", layout="wide")`;
- mostrar título, integrantes y estado del bonus;
- usar pestañas `Resumen`, `Resultados`, `Simulador`;
- cargar resultados mediante una función `@st.cache_data`;
- mostrar KPI de baseline, XGBoost, LinUCB y Oracle;
- permitir filtrar políticas mediante `st.multiselect`;
- graficar utilidad e intervalos con `st.bar_chart` y tabla formateada;
- mostrar resultados por semilla y métrica seleccionable;
- entrenar el demo mediante `@st.cache_resource`;
- validar el formulario y mostrar los dos brazos seleccionados;
- incluir la advertencia visible: “Demostración sintética: no es un resultado de RouterBench ni una decisión de producción”.

`app/requirements.txt`:

```text
-e .[app]
```

`.streamlit/config.toml`:

```toml
[theme]
base = "light"
primaryColor = "#2563EB"

[browser]
gatherUsageStats = false

[server]
headless = true
```

- [ ] **Step 4: Ejecutar smoke y pruebas del bonus**

Run:

```bash
python -m pip install -e ".[dev,app]"
python -m pytest -q tests/app
streamlit run app/streamlit_app.py --server.headless true --server.port 8501
```

Expected: tests PASS y servidor inicia sin excepción; detener tras confirmar health local.

- [ ] **Step 5: Commit**

```bash
git add app/streamlit_app.py app/requirements.txt .streamlit/config.toml tests/app/test_streamlit_app.py
git commit -m "feat: add interactive Streamlit research dashboard"
```

---

### Task 4: Documentación, README y CI del bonus

**Files:**
- Create: `docs/BONUS_STREAMLIT.md`
- Modify: `README.md`
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: aplicación terminada y ruta de entrada.
- Produces: instrucciones de despliegue y job `bonus-app`.

- [ ] **Step 1: Documentar ejecución y despliegue**

`docs/BONUS_STREAMLIT.md` debe incluir:

```bash
python -m pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

Y parámetros de Streamlit Community Cloud:

```text
Repository: FitoFritzG/better-router-adaptive-research
Branch: main
Main file path: app/streamlit_app.py
Python: 3.12
```

Debe explicar la frontera sintética, privacidad, ausencia de APIs y procedimiento de actualización.

- [ ] **Step 2: Actualizar metadatos**

- Subir versión a `0.4.0` en `pyproject.toml`.
- Añadir `/app`, `/.streamlit` y `/docs/BONUS_STREAMLIT.md` al sdist.
- Añadir al README una sección `Bonus: dashboard online` con comando local, enlace pendiente marcado como `Se añadirá después de autorizar el despliegue` y límites científicos.

- [ ] **Step 3: Añadir CI específica**

Agregar job independiente:

```yaml
  bonus-app:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: python -m pip install -e ".[dev,app]"
      - run: python -m pytest -q tests/app
      - run: ruff check app tests/app
      - run: ruff format --check app tests/app
      - run: mypy app tests/app
```

- [ ] **Step 4: Ejecutar compuerta completa**

Run:

```bash
python -m pytest -q
python -m pytest -q tests/app
ruff check .
ruff format --check .
mypy src tests app
python -m build
python scripts/verify_professor_submission.py
```

Expected: todos PASS; el verificador obligatorio no exige Streamlit ni altera el póster.

- [ ] **Step 5: Commit**

```bash
git add docs/BONUS_STREAMLIT.md README.md pyproject.toml .github/workflows/ci.yml
git commit -m "docs: document and validate Streamlit bonus"
```

---

### Task 5: Revisión, PR y preparación del despliegue

**Files:**
- Create: `docs/reviews/BONUS_STREAMLIT_REVIEW.md`
- Create: `docs/evidence/bonus-streamlit-final-gate.txt`

**Interfaces:**
- Consumes: diff completo del bonus.
- Produces: PR revisable y datos exactos para Streamlit Cloud.

- [ ] **Step 1: Revisar alcance y seguridad**

Registrar:

- no hay `requests`, `httpx`, `urllib.request` ni SDK de proveedores;
- no hay `st.secrets` ni variables de API;
- los prompts no se escriben a disco;
- el simulador usa `routerbench_sample.csv` y muestra advertencia;
- el dashboard usa únicamente resultados agregados.

- [ ] **Step 2: Ejecutar gate final y guardar evidencia**

Run:

```bash
python -m pytest -q
python -m pytest -q tests/app
ruff check .
ruff format --check .
mypy src tests app
python -m build
python scripts/verify_professor_submission.py
```

Guardar comandos, versiones y salidas resumidas en `docs/evidence/bonus-streamlit-final-gate.txt`.

- [ ] **Step 3: Abrir PR**

Título:

```text
bonus: add public Streamlit dashboard and routing simulator
```

El cuerpo debe separar resultados reales de simulación sintética y declarar que no se usan APIs externas.

- [ ] **Step 4: Validar GitHub Actions**

Esperar `quality (3.12)`, `quality (3.13)` y `bonus-app`. Revisar logs si cualquier control falla. No fusionar mientras exista un fallo o check pendiente.

- [ ] **Step 5: Fusionar y solicitar intervención mínima**

Después del merge, entregar estos campos:

```text
Repository: FitoFritzG/better-router-adaptive-research
Branch: main
Main file path: app/streamlit_app.py
Python: 3.12
```

Solicitar únicamente iniciar sesión con GitHub en Streamlit Community Cloud, autorizar el repositorio y pulsar Deploy. Tras recibir la URL, actualizar el README en un PR documental.
