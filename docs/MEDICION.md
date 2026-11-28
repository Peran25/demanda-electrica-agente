# Documento de Medición y Validación Experimental

## 1. Origen, Naturaleza y Particionado de los Datos
Los datos utilizados provienen íntegramente de **fuentes oficiales reales** recolectadas mediante scripts reproducibles (`src/data/`):
- **Demanda eléctrica real (`ESIOS`, indicador 1293):** Serie horaria en MW de la demanda peninsular española entre `2021-01-01` y `2024-12-31`, tras corregir la escala histórica y alinear en UTC. Se registran `8` horas ausentes correspondientes a `00:00` y `01:00 UTC` de los 4 domingos de cambio de hora (DST) en octubre entre `2021` y `2024`.
- **Meteorología (`Open-Meteo`):** Temperatura horaria a 2 m en Madrid, Barcelona, Valencia y Sevilla.
- **Calendario laboral (`Nager.Date`):** Festivos nacionales oficiales en España.

Tras la alineación UTC y la generación de retardos temporales (`lag_168h` requiere los primeros 7 días de 2021), el dataset supervisado (`data/processed/dataset_features.csv`) quedó particionado cronológicamente en:
- **Entrenamiento (`2021–2023`):** `26.106` registros horarios.
- **Evaluación / Test (`2024`):** `8.782` registros horarios (`8.517` horas no pico / `96,98 %` y `265` horas pico críticas / `3,02 %`).

---

## 2. Evaluación de Machine Learning (`eval/resultados/`)

### 2.1 Modelos Baseline (`baselines.json`)
Para establecer un suelo de referencia antes de entrenar modelos paramétricos, se evaluaron heurísticas temporales y un baseline trivial sobre el conjunto de test (`2024`):

#### Regresión (`baselines.json`)
| Baseline | MSE (MW²) | RMSE (MW) | MAE (MW) | MAPE (%) | R² |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Media móvil 24h (`media_movil_24h`) | 11.155.652,99 | 3.340,01 | 2.788,00 | 10,914 % | 0,3143 |
| Persistencia diaria (`persistencia_24h`) | 6.772.335,19 | 2.602,37 | 1.740,92 | 6,701 % | 0,5837 |
| Persistencia semanal (`persistencia_168h`) | 3.551.399,74 | 1.884,52 | 1.230,42 | 4,607 % | 0,7817 |

#### Clasificación (`baselines.json`)
| Baseline | Accuracy | Precision | Recall | F1-Score | Matriz de Confusión `[[TN, FP], [FN, TP]]` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Siempre no (`siempre_no`) | 0,9698 | 0,0000 | 0,0000 | 0,0000 | `[[8517, 0], [265, 0]]` |
| Persistencia semanal (`persistencia_168h`) | 0,9581 | 0,3057 | 0,3057 | 0,3057 | `[[8333, 184], [184, 81]]` |
| Persistencia diaria (`persistencia_24h`) | 0,9759 | 0,6000 | 0,6000 | 0,6000 | `[[8411, 106], [106, 159]]` |

### 2.2 Regresión de Demanda Horaria (`regresion.json`)
Comparativa sobre las `8.782` horas del año `2024`:

| Modelo | MSE (MW²) | RMSE (MW) | MAE (MW) | MAPE (%) | R² |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Ridge` (Lineal L2) | 760.588,23 | 872,12 | 664,54 | 2,491 % | 0,9532 |
| `RandomForest` | 173.047,89 | 415,99 | 302,96 | 1,123 % | 0,9894 |
| **`HistGradientBoosting`** | **125.786,66** | **354,66** | **265,51** | **0,989 %** | **0,9923** |

*Análisis:* `HistGradientBoosting` obtiene el mejor rendimiento en test (`RMSE = 354,66 MW`, `MAE = 265,51 MW`, `MAPE = 0,989 %`, `R² = 0,9923`), reduciendo el error respecto a `persistencia_168h` (`RMSE = 1.884,52 MW`, `R² = 0,7817`) y superando ampliamente a `Ridge` (`RMSE = 872,12 MW`, `R² = 0,9532`). Como precaución metodológica, debe revisarse en futuras iteraciones el peso de variables con dependencia temporal estrecha o posible fuga según el horizonte operativo (`demanda_lag_1h`, medias y desviaciones móviles y temperatura observada).

### 2.3 Clasificación de Hora Pico Crítica y Desplazamiento de Distribución (`clasificacion.json`)
El umbral de hora pico crítica se fijó en el percentil 95 de entrenamiento (`33.942,42 MW`). En el conjunto de test (`2024`), se observan `265` horas pico sobre `8.782` registros (`3,02 %`), frente al `5 %` teórico del percentil 95 en entrenamiento, lo que evidencia un **desplazamiento de distribución** interanual.

| Modelo | Accuracy | Precision | Recall | F1-Score | Matriz de Confusión `[[TN, FP], [FN, TP]]` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `LogisticRegression` | 0,9713 | 0,5130 | **0,9660** | 0,6702 | `[[8274, 243], [9, 256]]` |
| `RandomForest` | 0,9874 | 0,7265 | 0,9321 | 0,8165 | `[[8424, 93], [18, 247]]` |
| **`HistGradientBoosting`** | **0,9907** | **0,8188** | 0,8868 | **0,8514** | **`[[8465, 52], [30, 235]]`** |

*Análisis frente a baselines y desbalanceo:* Un baseline trivial `siempre_no` alcanza `0,9698` de accuracy pero `0,0` de precision, recall y F1 (no detecta ninguna de las `265` horas pico). `persistencia_168h` queda en `F1 = 0,3057` (`184` FP y `184` FN) y `persistencia_24h` en `F1 = 0,6000` (`106` FP y `106` FN). Entre los modelos entrenados, `HistGradientBoosting` logra el mayor `F1` (`0,8514`) con `81,88 %` de precision (`52` FP) y `88,68 %` de recall (`235` TP frente a `30` FN), equilibrando detección de picos y contención de falsas alarmas frente a `LogisticRegression` (`F1 = 0,6702`, `243` FP).

---

## 3. Evaluación Reproducible de RAG y Agente (`eval/resultados/bateria_agente.json`)

La batería (`eval/baterias/bateria_preguntas.json`) evalúa `6` escenarios sobre `gemma3:4b`:
- **Acierto en herramientas:** `5 / 6` (`83,33 %`).
- **Acierto en contenido:** `6 / 6` (`100,0 %`).
- **Latencia media:** `38,01 s`.

*Nota sobre escala:* Los valores numéricos registrados dentro de `eval/resultados/bateria_agente.json` corresponden a la ejecución previa al ajuste de escala histórica de demanda y quedan pendientes de regeneración tras el cierre de la medición.

---

## 4. Experimentos de Prompting y Parámetros (`eval/resultados/`)

### 4.1 Comparativa Zero-Shot vs. Chain-of-Thought (`comparativa_prompting.json`)
Ejecución reproducible (`python -m eval.run_prompting`) sobre los 6 casos de la batería:
- **Zero-Shot:** `83,33 %` (`5/6`) en acierto de herramientas, `83,33 %` (`5/6`) en acierto de contenido, `1,83` pasos medios y `37,24 s` de latencia media.
- **Chain-of-Thought (CoT):** `100,0 %` (`6/6`) en acierto de herramientas, `100,0 %` (`6/6`) en acierto de contenido, `2,50` pasos medios y `51,24 s` de latencia media.

### 4.2 Experimento de `temperature` y `seed` (`experimento_temperatura_seed.json`)
Ejecución reproducible (`python -m eval.run_experimento_params`) sobre la consulta compuesta de `2024-01-11 20:00`:
- `determinista_seed42` (`temperature = 0.0`, `seed = 42`): JSON válido, acción `predecir_demanda`, `28,87 s`.
- `determinista_seed99` (`temperature = 0.0`, `seed = 99`): JSON válido, acción `predecir_demanda`, `5,17 s`.
- `creativo_temp08_seed42` (`temperature = 0.8`, `seed = 42`): JSON válido, acción `predecir_demanda`, `4,93 s`.

El validador estructural mantiene validez JSON en todos los escenarios probados, mientras que `temperature = 0.0` conserva coherencia entre semillas.

---

## 5. Trazas de Referencia del Agente
- **Caso límite (`CLF_01`):** Consulta sobre `2024-01-11 20:00` mediante `clasificar_hora_pico`.
- **Caso multi-herramienta (`MULTI_01`):** Consulta compuesta sobre `2024-01-11 20:00` (`predecir_demanda`, `clasificar_hora_pico` y `consultar_normativa`).
- **Caso fuera de rango (`OOD_01`):** Consulta sobre marzo de `2026` resuelta sin invocar herramientas al quedar fuera del rango histórico `2021–2024`.
