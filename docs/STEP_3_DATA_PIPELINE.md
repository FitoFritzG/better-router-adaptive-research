# Paso 3 — Conversión, esquema canónico y limpieza

## Alcance implementado

- conversión del formato ancho de RouterBench a formato largo;
- verificación obligatoria del checksum antes de abrir un pickle;
- carga en proceso hijo con entorno sin secretos;
- exportación a CSV comprimido no ejecutable;
- exclusión de respuestas textuales;
- esquema canónico versionado;
- limpieza determinista;
- reportes JSON y CSV;
- gráficos de perfil;
- CLI de conversión y CLI de limpieza;
- pruebas unitarias e integración.

## Frontera de seguridad

El proceso hijo reduce la exposición accidental, pero no es un sandbox. La ejecución con el pickle real debe hacerse en infraestructura desechable y aislada.

## Evidencia versionada

Los archivos en `artifacts/examples/step3/` y `docs/assets/` se generan desde el fixture sintético. No constituyen resultados del estudio.

## Próximo paso

El Paso 4 creará características disponibles antes de la inferencia y particiones `70/15/15` agrupadas por `prompt_id`, evitando fuga entre entrenamiento, validación y prueba.
