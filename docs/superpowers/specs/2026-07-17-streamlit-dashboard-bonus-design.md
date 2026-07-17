# Diseño del bonus — Dashboard Streamlit y simulador académico

## Objetivo

Publicar una interfaz web de solo lectura que permita al profesor revisar los resultados del estudio y probar un simulador académico de XGBoost y LinUCB desde un enlace público, sin instalar dependencias, usar claves API ni descargar RouterBench.

## Alcance

La aplicación tendrá tres secciones dentro de una sola página Streamlit:

1. **Resumen del estudio**: problema de ingeniería, integrantes, dataset, función de utilidad, algoritmos y límites metodológicos.
2. **Resultados interactivos**: tabla, filtros y gráficos construidos exclusivamente desde `artifacts/public/step7-real/evaluation_summary.csv` y `evaluation_per_seed.csv`.
3. **Simulador académico**: el usuario escribe una consulta, selecciona uno de los cuatro grupos de tarea y observa qué brazo elegirían XGBoost y LinUCB.

La interfaz no ejecutará proveedores LLM, no enviará prompts a terceros y no consumirá créditos.

## Frontera científica del simulador

El repositorio público no redistribuye las filas de RouterBench ni modelos entrenados con ellas. Por esta razón, el simulador entrenará las implementaciones reales de `XGBoostRouter` y `LinUCBRouter` sobre el fixture original y sintético `tests/fixtures/routerbench_sample.csv`.

La interfaz mostrará permanentemente que:

- las selecciones del simulador son **demostraciones didácticas**;
- no corresponden a los resultados reales de RouterBench;
- no predicen calidad real ni representan una decisión de producción;
- los resultados científicos reales se muestran únicamente en la sección de resultados agregados.

## Arquitectura

### `app/data_access.py`

Responsable de resolver rutas desde la raíz del repositorio, cargar CSV públicos, validar columnas obligatorias y devolver copias ordenadas. No importa Streamlit.

Interfaces:

```python
@dataclass(frozen=True, slots=True)
class PublicResults:
    summary: pd.DataFrame
    per_seed: pd.DataFrame


def load_public_results(root: Path) -> PublicResults: ...
```

### `app/simulator.py`

Responsable de cargar el fixture sintético, calcular utilidad con los pesos del experimento, construir características preinferencia, entrenar ambos routers reales y simular una consulta.

Interfaces:

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


def train_demo_bundle(root: Path) -> DemoBundle: ...

def simulate_prompt(bundle: DemoBundle, *, prompt_text: str, task_group: str) -> SimulationResult: ...
```

### `app/streamlit_app.py`

Responsable únicamente de la presentación. Usa `st.cache_data` para resultados y `st.cache_resource` para el bundle de demostración. Presenta métricas, gráficos, filtros, advertencias y el formulario del simulador.

### Configuración y despliegue

- `.streamlit/config.toml`: tema claro, telemetría desactivada y modo headless.
- `app/requirements.txt`: instala el proyecto con el extra `app` mediante `-e .[app]`.
- Archivo de entrada para Streamlit Community Cloud: `app/streamlit_app.py`.
- Python: 3.12.

## Flujo de datos

```text
CSV agregados públicos ──> validación ──> tablas y gráficos Streamlit

Fixture sintético ──> utilidad ──> características ──> scaler
                 ├──> XGBoostRouter ──┐
                 └──> LinUCBRouter ───┼──> consulta del usuario ──> selección didáctica
                                      ┘
```

## Experiencia de usuario

La página inicia con el estado del proyecto y una advertencia de alcance. El profesor puede filtrar políticas y métricas, consultar intervalos de confianza y comparar resultados por semilla. En el simulador, debe seleccionar `coding`, `mathematics`, `reasoning` o `general`, escribir entre 5 y 2.000 caracteres y pulsar **Simular enrutamiento**.

Las selecciones usarán nombres didácticos para los brazos sintéticos:

- `arm-fast` → Rápido
- `arm-balanced` → Equilibrado
- `arm-reasoning` → Razonamiento
- `arm-premium` → Premium

## Manejo de errores

- CSV ausente o con columnas incorrectas: mensaje visible y detención segura de la sección.
- Prompt vacío, demasiado corto o superior a 2.000 caracteres: validación antes del entrenamiento o inferencia.
- Grupo de tarea desconocido: error explícito.
- Fallo al entrenar el demo: mensaje visible sin afectar la sección de resultados.

No se silencian errores mediante valores inventados.

## Pruebas

1. `tests/app/test_data_access.py`: carga válida, archivo ausente y columnas faltantes.
2. `tests/app/test_simulator.py`: entrenamiento, determinismo, validación de prompt, grupos desconocidos y ausencia de columnas de resultado en las características.
3. `tests/app/test_streamlit_app.py`: smoke test con `streamlit.testing.v1.AppTest` y verificación de títulos, advertencia científica y formulario.
4. Job CI `bonus-app`: instala `.[dev,app]`, ejecuta `tests/app/`, Ruff, formato y mypy sobre `app` y pruebas del bonus.

## Criterios de aceptación

- La aplicación inicia con `streamlit run app/streamlit_app.py`.
- Los resultados se leen desde artefactos agregados versionados, no desde valores copiados en el código.
- El simulador utiliza las clases reales XGBoost y LinUCB del proyecto.
- Ningún prompt sale del proceso local del servidor.
- No existen secretos, llamadas HTTP ni dependencia del dataset restringido.
- La interfaz muestra los tres integrantes.
- La CI obligatoria previa continúa pasando.
- El bonus tiene documentación de despliegue paso a paso.

## Fuera de alcance

- Llamadas a modelos reales.
- Autenticación de usuarios.
- Carga de datasets arbitrarios.
- Reentrenamiento sobre RouterBench desde la interfaz.
- Persistencia de prompts.
- Despliegue en Hugging Face Spaces.
