# Bonus — Dashboard Streamlit

## Objetivo

El bonus presenta los resultados de Better Router Adaptive mediante una interfaz web y permite ejecutar un simulador académico de XGBoost y LinUCB desde el navegador.

La aplicación no llama modelos externos, no solicita claves API y no guarda las consultas ingresadas.

## Alcance de la interfaz

### Resumen

- problema de ingeniería;
- integrantes;
- dataset y procesamiento;
- función de utilidad;
- algoritmos y comparadores;
- conclusión científica principal.

### Resultados interactivos

- utilidad media e intervalos de confianza;
- diferencia frente a Better Rules Proxy;
- filtros por política;
- comparación entre semillas;
- utilidad, calidad, costo y tasa de error.

Los valores se leen directamente desde:

```text
artifacts/public/step7-real/evaluation_summary.csv
artifacts/public/step7-real/evaluation_per_seed.csv
```

### Simulador académico

El usuario escribe una consulta y selecciona uno de cuatro grupos:

- programación;
- matemáticas;
- razonamiento;
- general.

La interfaz ejecuta las implementaciones reales de `XGBoostRouter` y `LinUCBRouter`, pero las entrena con `tests/fixtures/routerbench_sample.csv`, un fixture sintético original de 12 prompts y cuatro brazos anónimos.

Por lo tanto, la selección mostrada es una **demostración sintética**. No corresponde a RouterBench, no predice la calidad de proveedores reales y no debe usarse como una decisión de producción.

## Privacidad y seguridad

- No se realizan solicitudes HTTP.
- No se usan SDK de proveedores LLM.
- No existe configuración de secretos.
- Las consultas no se escriben en archivos ni bases de datos.
- RouterBench y sus filas procesadas no están presentes en el despliegue.
- La aplicación solo consume resultados agregados públicos y el fixture sintético original.

## Ejecución local

Desde la raíz del repositorio:

```bash
python -m venv .venv
```

Linux o macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

La dirección local predeterminada es:

```text
http://localhost:8501
```

## Pruebas

```bash
python -m pip install -e ".[dev,app]"
python -m pytest -q tests/app
ruff check app tests/app
ruff format --check app tests/app
mypy app tests/app
```

La CI mantiene además todas las pruebas de la entrega obligatoria.

## Despliegue en Streamlit Community Cloud

Parámetros:

```text
Repository: FitoFritzG/better-router-adaptive-research
Branch: main
Main file path: app/streamlit_app.py
Python: 3.12
```

Procedimiento:

1. Iniciar sesión en Streamlit Community Cloud mediante GitHub.
2. Autorizar acceso al repositorio público.
3. Crear una aplicación nueva.
4. Seleccionar el repositorio y la rama `main`.
5. Indicar `app/streamlit_app.py` como archivo principal.
6. Abrir la configuración avanzada y seleccionar Python 3.12.
7. Elegir un subdominio disponible y pulsar **Deploy**.

No se deben añadir secretos en la configuración del despliegue.

## Actualización

Después de fusionar cambios en `main`, Streamlit Community Cloud reconstruye la aplicación desde GitHub. Antes de fusionar una actualización deben pasar:

```bash
python -m pytest -q
python -m pytest -q tests/app
ruff check .
ruff format --check .
mypy src tests app
python -m build
python scripts/verify_professor_submission.py
```

## Archivos del bonus

```text
app/__init__.py
app/data_access.py
app/simulator.py
app/streamlit_app.py
app/requirements.txt
.streamlit/config.toml
tests/app/
docs/BONUS_STREAMLIT.md
```
