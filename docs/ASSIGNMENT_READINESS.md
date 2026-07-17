# Estado de la tarea universitaria

## Veredicto

**READY_WITH_FINAL_PACKAGING**: el contenido técnico satisface la pauta, pero la entrega académica todavía requiere empaquetado final y revisión visual del póster corregido.

## Cumplimiento

| Requisito | Estado | Evidencia |
|---|---|---|
| Máximo 3 integrantes | Cumple | Rodolfo Fritz, Benjamín Cerda y Felipe Friz |
| Procesamiento de datos | Cumple | adquisición, checksum, conversión, esquema y limpieza |
| Al menos 2 algoritmos de IA | Cumple | XGBoost y LinUCB; EvoCascade-Ideal es una extensión adicional |
| Al menos 2 métricas cuantitativas | Cumple ampliamente | utilidad, calidad, costo, error, regret e IC 95 % |
| Comparación experimental | Cumple | baseline, brazos fijos, XGBoost, LinUCB, Oracle y cascada |
| Fortalezas, limitaciones y mejoras | Cumple | `docs/RESULTS.md` y `LATEST_STUDY.md` |
| Metodología justificada | Cumple | split agrupado, normalización en train, tuning en validation y test final |
| Código Python modular y comentado | Cumple | paquete `better_router_adaptive` y pruebas automatizadas |
| Póster PDF | Cumple con corrección | versión metodológicamente corregida preparada |
| Archivos para ejecutar | Cumple con condición | código/configuración incluidos; dataset se descarga por licencia |
| Bono por hiperparámetros/iniciativa | Cumple | tuning de XGBoost/LinUCB y extensión sep-CMA-ES |

## Estado del repositorio

El repositorio está en estado fuerte para una tarea universitaria:

- CI en Python 3.12 y 3.13;
- cobertura mínima automatizada;
- lint, formato, tipado estricto y build;
- procedencia y hashes de datos;
- resultados agregados reproducibles;
- documentación en español;
- tres autores formalizados;
- extensión EvoCascade identificada explícitamente como cota experimental.

## Pendientes antes de entregar

1. Exportar y comprobar visualmente el póster final.
2. Crear el ZIP con el nombre solicitado por la pauta.
3. Ejecutar la verificación desde una carpeta limpia.
4. No incluir `.venv`, cachés, secretos ni datos sin permiso de redistribución.
5. Incluir instrucciones para descargar el dataset y reproducir los resultados.
6. Preparar el cuestionario individual de defensa.

El informe IEEE es valioso para Better AI y una futura publicación, pero no sustituye los entregables obligatorios de póster, código y archivos de ejecución.
