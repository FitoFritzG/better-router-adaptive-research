# Revisión del Paso 3 — Conversión, esquema canónico y limpieza

## Veredicto

- **Cumplimiento del alcance:** APROBADO.
- **Calidad de código:** APROBADA.
- **Seguridad y privacidad:** APROBADA CON LIMITACIÓN DOCUMENTADA SOBRE PICKLE.
- **Reproducibilidad:** APROBADA PARA EL FIXTURE SINTÉTICO.
- **Ejecución del artefacto real:** PENDIENTE EN UN ENTORNO DESECHABLE CON ACCESO A LA DESCARGA.

## Alcance revisado

- conversión de RouterBench ancho a formato largo;
- clave primaria `(prompt_id, model_id)`;
- exclusión de respuestas textuales de modelos;
- checksum obligatorio antes de abrir un pickle;
- proceso hijo con entorno sin secretos y sin `shell`;
- publicación atómica de CSV comprimido;
- validación de rangos, tipos y consistencia por prompt;
- eliminación auditada de duplicados exactos y grupos incompletos;
- rechazo de duplicados contradictorios;
- reportes JSON/CSV y manifiesto de esquema;
- gráficos SVG rotulados como demostración sintética;
- README principal completamente documentado en español;
- pruebas unitarias e integración.

## Evidencia de verificación local

```text
51 pruebas aprobadas
92 % de cobertura total con ramas
Ruff: PASS
Ruff format: PASS
mypy estricto: PASS
compileall: PASS
build wheel/sdist: PASS
CLI convert: PASS
CLI pipeline: PASS
JSON/SVG: PASS
escaneo de secretos: PASS
```

## Hallazgos corregidos

1. El primer caso de limpieza mezclaba un duplicado contradictorio con una prueba de filas inválidas. Se separaron ambos comportamientos.
2. La cobertura inicial del nuevo conjunto era 81 %. Se añadieron pruebas de errores, CLI, proceso hijo y ausencia de latencia hasta alcanzar 92 % total.
3. Ruff detectó siete problemas de formato y colecciones duplicadas; fueron corregidos.
4. Mypy detectó tipos incompletos en pandas y monkeypatching; se agregaron stubs y se ajustaron las pruebas.
5. Los gráficos PNG se sustituyeron por SVG para mantener artefactos versionables, legibles y renderizables por GitHub.
6. El README deja explícito que los gráficos actuales no son resultados experimentales.

## Evaluación de seguridad

La conversión verifica el SHA-256 y ejecuta la carga en un proceso hijo sin secretos heredados. Esto reduce el riesgo operativo, pero no convierte pickle en un formato seguro ni crea un sandbox. El artefacto real debe procesarse en una máquina o contenedor desechable, sin credenciales ni conexión con Better Router o SIMPA.

## Limitaciones residuales

- El entorno de esta sesión no pudo descargar el archivo oficial de 99,6 MB, por lo que no se afirma que se haya ejecutado la conversión real.
- RouterBench no garantiza latencia ni tokens en el artefacto seleccionado; estos campos permanecen nulos.
- La selección final de cuatro modelos se realizará después de inspeccionar el CSV real convertido.
- Las particiones por prompt y las características sin fuga pertenecen al Paso 4.
