# Guía de evaluación del código

## Proyecto

**Better Router Adaptive Research**

Integrantes:

- Rodolfo Fritz
- Benjamín Cerda
- Felipe Friz

## Alcance obligatorio evaluable

La solución principal implementa y compara dos algoritmos de inteligencia artificial:

1. **XGBoost**, utilizado como regresor supervisado de utilidad para cada modelo candidato.
2. **LinUCB**, utilizado como bandit contextual para seleccionar modelos de manera secuencial.

La comparación incluye además:

- Better Rules Proxy como línea base determinista;
- brazos fijos como referencias;
- Oracle offline como cota superior no desplegable.

La evaluación obligatoria no requiere ejecutar EvoCascade, una interfaz gráfica ni funciones de bonus.

## Requisitos

- Python 3.12 o 3.13.
- Git.
- Aproximadamente 2 GB libres para dependencias, datos temporales y resultados.

## Instalación

```bash
git clone https://github.com/FitoFritzG/better-router-adaptive-research.git
cd better-router-adaptive-research
python -m venv .venv
```

Linux o macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Verificación rápida

```bash
python scripts/verify_professor_submission.py
python -m pytest -q
```

## Verificación completa

En Linux/macOS con `make`:

```bash
make professor-check
```

En cualquier sistema:

```bash
python -m pytest -q \
  --cov=better_router_adaptive \
  --cov-branch \
  --cov-report=term-missing \
  --cov-fail-under=85
ruff check .
ruff format --check .
mypy src tests
python -m build
python scripts/verify_professor_submission.py
```

## Flujo de la solución

1. Descargar RouterBench desde la fuente oficial y verificar tamaño y SHA-256.
2. Convertir el artefacto a CSV no ejecutable.
3. Validar esquema, limpiar datos y conservar prompts con cuatro modelos completos.
4. Crear características pre-inferencia evitando fuga de información.
5. Dividir por `prompt_id` en entrenamiento, validación y prueba.
6. Calcular una utilidad conjunta de calidad, costo, latencia y error.
7. Entrenar XGBoost y LinUCB usando únicamente entrenamiento y validación.
8. Evaluar en test mediante utilidad, calidad, costo, error, regret e intervalos de confianza.

## Ejecución por etapas

```bash
python -m better_router_adaptive.data.download --help
python -m better_router_adaptive.data.convert --help
python -m better_router_adaptive.data.pipeline --help
python -m better_router_adaptive.prepare --help
python -m better_router_adaptive.baselines --help
python -m better_router_adaptive.learn --help
python -m better_router_adaptive.evaluate --help
```

Las instrucciones completas de descarga y reproducción están en `README.md`, `data/README.md` y `docs/REPRODUCIBILITY.md`.

## Evidencia versionada

Para permitir una revisión sin redistribuir el dataset de terceros, el repositorio incluye:

- resultados agregados de cinco semillas;
- intervalos de confianza mediante bootstrap pareado;
- figuras de comparación;
- configuraciones experimentales;
- pruebas unitarias y de integración;
- documentación metodológica;
- hashes y procedencia del dataset.

Los resultados obligatorios se encuentran en:

```text
artifacts/public/step7-real/
```

## Restricción del dataset

El dataset original y los datos procesados fila por fila no se incluyen en el repositorio público porque la tarjeta de RouterBench no declara una licencia explícita de redistribución. Los scripts permiten regenerarlos desde la fuente oficial.

## Póster

El póster científico se entrega como archivo PDF separado. Este repositorio no genera, modifica ni empaqueta nuevamente ese PDF.

## Bonus

El bonus no forma parte de esta versión cerrada. Se desarrollará después en una rama nueva y separada.
