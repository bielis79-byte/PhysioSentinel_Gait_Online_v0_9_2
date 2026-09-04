# PhysioSentinel Gait Online v0.9.2

## Objetivo principal
Evitar que la aplicación publique cadencia, asimetría temporal o doble apoyo cuando
las distintas partes del detector temporal describen una marcha matemáticamente
incompatible.

## Línea temporal anatómica única
La v0.9.2 construye una secuencia canónica:
IC izquierda → IC derecha → IC izquierda → IC derecha...

Los contactos proceden de los mismos ciclos IC→TO→IC utilizados para apoyo y
oscilación. Ya no se usa una alternancia impar/par como sustituto de lateralidad.

## Cadencia
Se generan tres estimaciones internas:
1. cadencia por pasos anatómicos L-R;
2. cadencia por ciclos ipsilaterales IC→IC, sin usar el filtro estadístico del CV;
3. cadencia compatible con el cierre físico apoyo izquierdo + apoyo derecho +
   doble apoyo observado.

La cadencia clínica solo se publica si las estimaciones disponibles concuerdan
dentro de un 10%. Si no, se muestra como no fiable/no calculable en lugar de
seleccionar silenciosamente una cifra.

## CV
Se conserva el algoritmo robusto de v0.9.1:
- CV izquierdo y derecho por separado;
- rechazo robusto de outliers;
- CV global ponderado;
- número de ciclos válido por lado.

Importante: el filtro de outliers del CV ya NO decide la cadencia.

## Asimetría temporal
Se calcula exclusivamente a partir de:
- IC izquierda → IC derecha
frente a
- IC derecha → IC izquierda
de la misma cadena anatómica.

Solo se publica si se supera el control global de coherencia temporal.

## Doble apoyo
El doble apoyo sigue siendo la intersección temporal directa de ambas máscaras de
apoyo, pero el control es más estricto:
- fase aérea <=5%;
- discrepancia con la ocupación de apoyo <=5 puntos porcentuales;
- coherencia temporal global superada.

Si falla cualquiera de estos controles, el doble apoyo se suprime.

## Métricas de control nuevas
- cadencia candidata por pasos anatómicos;
- cadencia candidata por ciclos ipsilaterales;
- cadencia candidata por cierre apoyo/doble apoyo;
- discrepancia pasos vs ciclos;
- discrepancia pasos vs cierre físico;
- discrepancia ciclos vs cierre físico;
- discordancia temporal máxima;
- flag de coherencia temporal global.

## Conservado
- modo multipersona v0.9.0;
- bloqueo manual de identidad;
- rechazo de frames ambiguos;
- CV robusto v0.9.1;
- exclusión automática de giro/transiciones;
- desfase tronco-pelvis normalizado a −180°…+180°.

## Alcance
Sigue siendo un análisis markerless 2D experimental. El control de coherencia
reduce falsa precisión, pero no convierte los eventos cinemáticos estimados en
mediciones equivalentes a plataforma de fuerzas o laboratorio 3D.
