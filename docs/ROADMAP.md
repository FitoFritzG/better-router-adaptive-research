# Hoja de ruta de investigación

## Fase 1 — Base reproducible ✅

- Paquete Python, configuración inmutable, pruebas, cobertura, CI y citación.

## Fase 2 — Dataset y procedencia ✅

- Fuente oficial de RouterBench, licencia, revisión, tamaño y SHA-256.
- Fixture sintético y política de no redistribución.

## Fase 3 — Pipeline de datos ✅

- Conversión controlada del pickle verificado a CSV comprimido.
- Formato largo `(prompt_id, model_id)`.
- Validación, limpieza, reportes, manifiesto y gráficos.

## Fase 4 — Características y particiones ✅

- Variables conocidas antes de inferencia.
- División `70/15/15` agrupada por prompt.
- Pruebas automáticas contra fuga de información.

## Fase 5 — Utilidad y referencias ✅

- Función de utilidad, Better Rules Proxy y Oracle offline.

## Fase 6 — Algoritmos ✅

- XGBoost con ajuste exclusivo en validación.
- LinUCB disjunto con replay prequential.

## Fase 7 — Evaluación y publicación 🔄

- Métricas y bootstrap pareado ✅ (`python -m better_router_adaptive.evaluate`).
- Póster científico ✅ (`paper/poster/build_poster.py`, generado desde artefactos).
- Pendiente: informe IEEE y artículo Better AI.

La integración con Better Router en producción queda fuera de la fase académica y exigirá modo sombra y revisión de seguridad.

## Estado actualizado del proyecto académico

- [x] Pasos 4–7 recibidos y revisados.
- [x] Resultados reales reproducidos exactamente.
- [x] Bloqueo nativo de evaluación corregido mediante aislamiento por semilla.
- [x] Resultados agregados y póster preparados para publicación.
- [x] Autoría de Rodolfo Fritz, Benjamín Cerda y Felipe Friz incorporada.
- [ ] Actualizar el informe IEEE final con resultados, discusión y firmas académicas.
