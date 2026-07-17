# Corrección metodológica del póster EvoCascade

La versión final del póster debe usar el nombre **EvoCascade-Ideal** y mostrar de forma visible:

> El verificador es perfecto y simulado con la etiqueta del benchmark. La mejora es una cota superior experimental, no rendimiento directamente desplegable.

Cambios obligatorios respecto del borrador del grupo:

1. eliminar correos personales del encabezado público;
2. reemplazar «única política aprendida que supera la baseline» por una afirmación condicionada al verificador ideal;
3. evitar «fallo observable en producción», porque la escalada real se activa con `quality == 0`;
4. incluir la ablación: nunca escalar `U=0.431`, escalar siempre `U=0.496`;
5. concluir que el próximo problema es entrenar y evaluar un verificador real;
6. conservar las métricas principales, pero etiquetarlas como cota experimental.

La versión corregida fue renderizada y revisada visualmente en formato A0 horizontal. No presenta recortes ni superposiciones.
