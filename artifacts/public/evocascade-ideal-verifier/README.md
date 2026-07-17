# EvoCascade-Ideal - artefactos públicos

Esta carpeta contiene resultados agregados y ablaciones. No contiene el dataset fila por fila.

## Resultado principal

El estudio entregado registra una mejora idealizada de utilidad de aproximadamente `+0.0321` frente a Better Rules Proxy, con IC 95 % `[+0.0302; +0.0341]`.

## Advertencia metodológica

La política `evocascade-ideal-verifier` usa un verificador perfecto simulado con la etiqueta de calidad del benchmark. El resultado es una cota superior y no debe presentarse como desempeño directamente desplegable.

## Ablación

- verificador ideal: utilidad `0.52846`;
- nunca escalar: `0.43113`;
- escalar siempre: `0.49568`;
- baseline: `0.49615`.

La interpretación completa se encuentra en `LATEST_STUDY.md`.
