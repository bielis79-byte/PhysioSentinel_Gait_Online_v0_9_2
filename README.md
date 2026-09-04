---
title: Sentinel Gait
emoji: 🚶
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# PhysioSentinel Gait Online v0.9.2

Versión centrada en mejorar la coherencia biomecánica de las métricas temporales 2D.

Principales cambios: exclusión automática de giro/transiciones, CV y asimetría derivados de contactos
I/D validados, doble apoyo restringido a ciclos válidos y controles internos que suprimen resultados
incoherentes en lugar de mostrarlos como fiables.

Las métricas markerless 2D son experimentales y deben integrarse con la observación clínica.


## v0.9.2 · Modo multipersona
Selección manual del paciente + seguimiento de identidad bloqueado + rechazo de frames ambiguos. Pensado para marcha con acompañante, supervisión estrecha o asistencia física.


## v0.9.2 · CV robusto por lado
CV izquierdo y derecho calculados por separado, rechazo robusto de ciclos atípicos, CV global ponderado y tamaño muestral explícito.

## v0.9.2 · Coherencia temporal fuerte
Cadencia desde una línea temporal anatómica L-R, control independiente por ciclos
ipsilaterales y cierre físico apoyo/doble apoyo. Las métricas temporalmente
incompatibles se suprimen en lugar de publicarse con falsa precisión.
