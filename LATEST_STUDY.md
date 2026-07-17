# Último estudio: EvoCascade con verificador ideal

Esta extensión documenta el último análisis del equipo sobre una política de cascada entrenada con **sep-CMA-ES**. Se integra como una investigación exploratoria separada del resultado principal de XGBoost y LinUCB.

## Resultado observado

En RouterBench 0-shot, `EvoCascade-Ideal` obtiene una utilidad media de **0,52819**, frente a **0,49608** de Better Rules Proxy. La diferencia pareada es **+0,03211**, con IC 95 % **[+0,03023; +0,03409]**.

También aumenta la calidad media de aproximadamente `0,783` a `0,825` y reduce el costo medio de aproximadamente USD `0,00329` a USD `0,00201`.

## Interpretación correcta

El resultado utiliza un **verificador perfecto simulado**: la cascada escala cuando la calidad verdadera del primer modelo es exactamente cero o cuando la llamada falla. Esa calidad es una etiqueta del benchmark conocida después de evaluar la respuesta; no es una señal disponible automáticamente en producción.

Por tanto, la cifra anterior es una **cota superior experimental del valor potencial de una cascada con verificación**, no evidencia de que exista actualmente un router desplegable con esa mejora.

## Ablaciones decisivas

Promedio de cinco semillas, usando las mismas acciones elegidas por la política:

| Escenario | Utilidad | Diferencia vs. baseline |
|---|---:|---:|
| Better Rules Proxy | 0,49615 | 0 |
| EvoCascade con verificador ideal | 0,52846 | +0,03230 |
| Nunca escalar | 0,43113 | -0,06503 |
| Escalar siempre | 0,49568 | -0,00048 |

La política selecciona una acción de cascada en **83,34 %** de los prompts, pero el verificador ideal escala realmente en **24,84 %**. Sin detección perfecta del fallo, los primeros brazos escogidos rinden por debajo de la baseline.

## Fortalezas

- amplía el routing desde una única llamada hacia decisiones secuenciales;
- optimiza directamente una utilidad no diferenciable mediante sep-CMA-ES;
- evalúa cinco semillas y bootstrap pareado;
- contabiliza el costo de las dos llamadas cuando existe escalada;
- formula una dirección aplicable a Better Router: ejecutar barato y escalar selectivamente.

## Limitaciones

- el verificador usa ground truth y no es desplegable;
- falta una curva de sensibilidad/especificidad del verificador;
- RouterBench no aporta latencia en el artefacto utilizado;
- los cuatro brazos fueron seleccionados de manera exploratoria;
- el experimento offline no cubre cambios de precio, fallos de proveedor ni tráfico real.

## Conclusión defendible

> La verificación post-inferencia puede capturar parte de la brecha entre una política fija y el Oracle, pero su utilidad depende críticamente de la precisión, costo y latencia de un verificador real.

No debe afirmarse que EvoCascade ya supera a sistemas comerciales, Sakana Fugu o al routing de producción de Better Router.

Los resultados agregados y las ablaciones se encuentran en [`artifacts/public/evocascade-ideal-verifier/`](artifacts/public/evocascade-ideal-verifier/). El estado académico está documentado en [`docs/ASSIGNMENT_READINESS.md`](docs/ASSIGNMENT_READINESS.md).
