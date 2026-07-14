# Resultados experimentales y análisis crítico

## Autores

Rodolfo Fritz, Benjamín Cerda y Felipe Friz.

## Ejecución evaluada

- Dataset: RouterBench 0-shot.
- Prompts completos: 36.497.
- Filas canónicas: 145.988.
- Brazos: 4.
- Semillas: 42, 123, 2026, 31415 y 271828.
- División: 70/15/15 agrupada por `prompt_id` y estratificada por grupo de tarea.
- Bootstrap: 2.000 remuestras pareadas por `prompt_id`.
- Prompts únicos agrupados en test: 20.263.
- XGBoost reproducido con versión 3.3.0.

## Resultado principal

Las políticas aprendidas no entregan una mejora estadísticamente defendible sobre Better Rules Proxy:

- XGBoost: diferencia `+0,00000012`, IC 95 % `[-0,000548; 0,000576]`.
- LinUCB: diferencia `-0,000142`, IC 95 % `[-0,000475; 0,000189]`.

Ambos intervalos cruzan cero. Por lo tanto, no se rechaza la hipótesis nula de igualdad frente a la línea base bajo la función de utilidad y las señales disponibles.

El Oracle obtiene una mejora de `+0,068004`, con IC 95 % `[0,065643; 0,070490]`. Esta diferencia es consistente y demuestra que sí existe heterogeneidad útil entre modelos a nivel de consulta.

## Por qué la línea base colapsa a GPT-4

Better Rules Proxy selecciona GPT-4 en todas las categorías. Las causas observadas son:

1. GPT-4 posee la mayor calidad media.
2. La normalización min-max del costo se ajusta con el máximo del split de entrenamiento, que contiene una cola de costos altos.
3. La penalización relativa del costo típico de GPT-4 queda reducida frente al peso de calidad.
4. Las características disponibles —longitud superficial y grupo de tarea— no distinguen suficientemente qué prompts pueden resolverse con modelos económicos.

XGBoost aprende esencialmente esa misma política. LinUCB explora más, pero no dispone de contexto suficiente para recuperar la brecha del Oracle.

## Reproducibilidad verificada

Se ejecutó nuevamente la evaluación completa después de revisar el aporte recibido. Los archivos regenerados coincidieron exactamente con los entregados:

- `evaluation_summary.csv`: diferencia numérica máxima `0,0`.
- `evaluation_per_seed.csv`: diferencia numérica máxima `0,0`.
- Campos no numéricos: coincidencia exacta.

La evaluación de cada semilla fue aislada en un proceso independiente para evitar acumulación de estado nativo de XGBoost y OpenMP.

## Limitaciones

### Latencia ausente

`latency_ms` es nulo en el 100 % de las filas del artefacto experimental. El estudio no puede afirmar mejoras empíricas de latencia. El término correspondiente se deshabilita explícitamente.

### Selección exploratoria de brazos

Los cuatro modelos se eligieron tras observar promedios globales de calidad y costo en el artefacto. Esta decisión es razonable para una prueba de factibilidad, pero introduce riesgo de sesgo de selección. La investigación debe presentarse como exploratoria.

### Características limitadas

No se utilizaron embeddings semánticos, complejidad aprendida, dominio fino, incertidumbre ni historial del usuario. Estas señales podrían acercar las políticas al Oracle.

### Función de utilidad fija

Los pesos `0,65/0,20/0,10/0,05` representan una preferencia particular. Se requiere análisis de sensibilidad con perfiles orientados a calidad, costo y velocidad.

### Benchmark offline

RouterBench permite evaluar todas las decisiones contrafactuales porque contiene resultados de cada brazo. Una puesta en producción tendría retroalimentación parcial, cambios de precio, fallos, límites de cuota y drift.

## Trabajo futuro

1. Incorporar embeddings del prompt y características semánticas.
2. Repetir el análisis con normalización robusta de costo, por ejemplo percentiles o `log1p`.
3. Evaluar distintos perfiles de utilidad.
4. Separar un conjunto de modelos sin inspeccionar el test.
5. Ejecutar modo sombra en Better Router sin afectar decisiones reales.
6. Incorporar latencias observadas y tasas de fallo de proveedores.
