# Revisión técnica de la entrega colaborativa

## Integrantes

- Rodolfo Fritz
- Benjamín Cerda
- Felipe Friz

## Alcance recibido

La entrega revisada incorporó avances sustanciales posteriores al Paso 3:

- características pre-inferencia y split por prompt;
- normalización y función de utilidad;
- Oracle, Better Rules Proxy y brazos fijos;
- XGBoost y LinUCB;
- evaluación multi-semilla con bootstrap pareado;
- artefactos reales de RouterBench;
- generador y PDF del póster científico.

## Veredicto

- Metodología general: **APROBADA CON LIMITACIONES DOCUMENTADAS**.
- Pruebas unitarias e integración: **APROBADAS**.
- Resultados agregados: **REPRODUCIDOS EXACTAMENTE**.
- Dataset procesado para repositorio público: **RECHAZADO POR POLÍTICA DE NO REDISTRIBUCIÓN**.
- Estabilidad de evaluación multi-semilla original: **CORREGIDA**.

## Hallazgos y correcciones

### 1. Bloqueo nativo en ejecuciones repetidas

La evaluación podía quedar bloqueada después de varios entrenamientos XGBoost por acumulación de estado nativo y sobreasignación de hilos BLAS/OpenMP.

Correcciones:

- `n_jobs=1` y límite explícito del pool nativo durante el entrenamiento;
- liberación de los `Booster` al cerrar cada router;
- valores conservadores de hilos nativos al cargar el paquete;
- aislamiento de cada semilla de evaluación en un proceso `spawn` independiente;
- pruebas de regresión para límites de hilos, liberación y aislamiento.

### 2. Reproducibilidad de XGBoost

Los resultados publicados se reproducen con `xgboost==3.3.0`. Una versión anterior produjo diferencias pequeñas. La dependencia quedó fijada.

### 3. Redistribución de datos

La entrega contenía una tabla procesada de RouterBench. El repositorio ya había adoptado una política conservadora porque la tarjeta del dataset no declara licencia explícita. La tabla fila por fila se excluye de GitHub; se publican únicamente scripts, hashes, agregados y figuras.

### 4. Documentación desactualizada

El README aún afirmaba que no existían resultados experimentales. Fue actualizado con métricas reales, interpretación negativa, limitaciones y los tres integrantes.

### 5. Autoría

La entrega mencionaba solo a dos participantes. Se añadieron Rodolfo Fritz, Benjamín Cerda y Felipe Friz en README, `AUTHORS.md`, `CITATION.cff`, metadatos del paquete, documentación y póster.

### 6. Privacidad en el póster

Se eliminó un correo personal del encabezado público del póster. El documento conserva institución, asignatura y año.

## Evidencia final esperada

- suite completa de pruebas;
- cobertura de ramas superior a 85 %;
- Ruff y formato aprobados;
- mypy estricto aprobado;
- paquete construible;
- ejecución final reproducida;
- CI remota en Python 3.12 y 3.13.
