# Proyecto Integrador M5 **HammerZaldua**
Repositorio para el desarrollo de un modelo predictivo de riesgo crediticio utilizando técnicas de Machine Learning. Incluye un pipeline estructurado bajo principios de MLOps, abarcando desde la carga y análisis de datos hasta el entrenamiento, despliegue y monitoreo del modelo.
# Avance #1 - Proyecto Integrador M5
### Objetivo del Avance #1


# Avance #2 - Proyecto Integrador M5
Se identificó un posible problema de data leakage debido a métricas perfectas.
Se procedió a eliminar variables que contenían información derivada del comportamiento de pago,
logrando un modelo más realista y generalizable. un leakage fuerte de la variable Puntaje
Posible problema

Algunas de estas variables ( PUNTAJE saldo_principal huella_consulta capital_prestado) pueden:

reflejar estado posterior al crédito
correlacionarse indirectamente con el target
actuar como “proxy” del puntaje

👉 Es decir, leakage disfrazado
Se eliminó la variable "puntaje" debido a su alta correlación con la variable objetivo,
evitando así problemas de data leakage y garantizando un modelo más generalizable.

El modelo inicial presentó bajo recall en la clase minoritaria (morosos),
por lo que se ajustó el umbral de clasificación para priorizar la detección
de clientes en riesgo, mejorando significativamente el desempeño en dicha clase.

El modelo presenta dificultades para identificar la clase minoritaria debido al desbalance del dataset.
Se realizó ajuste de umbral de decisión, evidenciando la necesidad de modelos más robustos.

A pesar de utilizar modelos avanzados como XGBoost, el desempeño en la detección
de la clase minoritaria fue limitado, evidenciando la dificultad del problema
debido al alto desbalance y la baja separabilidad de las variables.

Se exploraron diferentes umbrales de decisión y técnicas de ponderación,
sin lograr mejoras significativas en el recall de la clase minoritaria.

Se evaluaron modelos Random Forest y XGBoost bajo diferentes umbrales de decisión.

Los resultados evidencian que XGBoost presenta un mejor desempeño en la detección de la clase minoritaria (clientes en riesgo), especialmente a umbrales más altos.

Se observa un trade-off entre recall y accuracy, donde al aumentar el threshold se mejora la detección de morosos, pero se reduce la precisión global del modelo.



Para fines de negocio, se recomienda utilizar XGBoost con un threshold de 0.5 o 0.7, priorizando la identificación de clientes con riesgo de incumplimiento.

en el model deploy: El modelo genera probabilidades de incumplimiento, las cuales son transformadas en decisiones mediante un umbral configurable, permitiendo adaptar el modelo a diferentes estrategias de riesgo.


para el model monitoring: Se implementó un sistema de monitoreo que evalúa periódicamente la estabilidad del modelo mediante métricas estadísticas como PSI, KS Test, Jensen-Shannon y Chi-cuadrado, permitiendo detectar cambios en la distribución de los datos que puedan afectar el desempeño del modelo.