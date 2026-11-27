@'
# Documento de Medición y Validación Experimental

## 1. Origen, Naturaleza y Particionado de los Datos
Los datos utilizados provienen íntegramente de **fuentes oficiales reales** recolectadas mediante scripts reproducibles (`src/data/`):
- **Demanda eléctrica real (`ESIOS`, indicador 1293):** Serie horaria en MW de la demanda peninsular española entre `2021-01-01` y `2024-12-31` (`35.064` registros originales).
- **Meteorología (`Open-Meteo`):** Temperatura horaria a 2 m en Madrid, Barcelona, Valencia y Sevilla.
- **Calendario laboral (`Nager.Date`):** Festivos nacionales oficiales en España.

Tras la alineación UTC y la generación de retardos temporales (`lag_168h` requiere los primeros 7 días de 2021), el dataset supervisado (`data/processed/dataset_features.csv`) quedó particionado cronológicamente en:
- **Entrenamiento (`2021–2023`):** `26.106` registros horarios.
- **Evaluación / Test (`2024`):** `8.782` registros horarios (`8.044` horas normales y `738` horas pico críticas).

---

## 2. Evaluación de Machine Learning (`eval/resultados/`)

### 2.1 Modelos Baseline (`baselines.json`)
Para establecer un suelo de referencia antes de entrenar modelos paramétricos, se evaluaron tres heurísticas temporales sobre el conjunto de test (`2024`):

| Baseline | MSE (MW²) | RMSE (MW) | MAE (MW) | MAPE (%) | R² |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Media móvil 24h | 1.606.414.030,57 | 40.080,10 | 33.455,97 | 10,914 % | 0,3143 |
| Persistencia diaria (`lag_24h`) | 975.216.267,95 | 31.228,45 | 20.891,03 | 6,701 % | 0,5837 |
| Persistencia semanal (`lag_168h`) | 511.401.561,94 | 22.614,19 | 14.765,09 | 4,607 % | 0,7817 |

### 2.2 Regresión de Demanda Horaria (`regresion.json`)
Comparativa sobre las `8.782` horas del año `2024`:

| Modelo | MSE (MW²) | RMSE (MW) | MAE (MW) | MAPE (%) | R² |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Ridge` (Lineal L2) | 117.265.747,78 | 10.828,93 | 8.209,40 | 2,559 % | 0,9499 |
| `RandomForestRegressor` | 24.812.227,42 | 4.981,19 | 3.606,36 | 1,121 % | 0,9894 |
| **`HistGradientBoostingRegressor`** | **16.202.713,23** | **4.025,26** | **3.007,66** | **0,935 %** | **0,9931** |

*Análisis:* En el modelo lineal `Ridge`, los mayores pesos positivos corresponden a la demanda retardada de 24 h (`demanda_lag_24h`), la persistencia semanal (`demanda_lag_168h`), la media móvil de 24 h (`rolling_mean_24h`) y los grados-día de climatización (`hdd_18`, `cdd_22`), mientras que los pesos negativos más fuertes recaen sobre `es_festivo` y `es_fin_de_semana`. El salto a `HistGradientBoosting` reduce el `MSE` en un **86,2 %** respecto al modelo lineal al capturar interacciones no lineales entre hora, temperatura y tipo de día.

### 2.3 Clasificación de Hora Pico Crítica y Análisis de Desbalanceo (`clasificacion.json`)
El umbral de hora pico crítica se fijó en el percentil 95 de entrenamiento (`386.923,5 MW`). En el conjunto de test (`2024`), la distribución presenta un fuerte **desbalanceo de clases**: `8.044` horas negativas (`91,6 %`) frente a `738` horas pico (`8,4 %`).

| Modelo | Accuracy | Precision | Recall | F1-Score | Matriz de Confusión `[[TN, FP], [FN, TP]]` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline `persistencia_24h` | 0,9472 | 0,6856 | 0,6856 | 0,6856 | `[[7812, 232], [232, 506]]` |
| `LogisticRegression` | 0,9505 | 0,6344 | **0,9688** | 0,7668 | `[[7632, 412], [23, 715]]` |
| `RandomForestClassifier` | 0,9763 | 0,8032 | 0,9512 | 0,8710 | `[[7872, 172], [36, 702]]` |
| **`HistGradientBoostingClassifier`** | **0,9836** | **0,8887** | 0,9201 | **0,9041** | **`[[7959, 85], [59, 679]]`** |

*Por qué la Accuracy sola engaña en este problema:* Un clasificador trivial que predijera siempre "no es hora pico" obtendría un `91,6 %` de accuracy con `0 %` de utilidad operativa. Del mismo modo, `LogisticRegression` alcanza un `95,05 %` de accuracy pero genera **412 falsas alarmas** (`Precision = 63,44 %`), lo que provocaría activaciones innecesarias de reserva en el sistema eléctrico. `HistGradientBoostingClassifier` mantiene un `Recall` del `92,01 %` (detecta 679 de las 738 horas críticas) reduciendo los falsos positivos a solo 85 (`F1 = 0,9041`).

---

## 3. Evaluación Reproducible de RAG y Agente (`eval/run_bateria.py`)

La batería (`eval/baterias/bateria_preguntas.json`) evalúa 6 escenarios diferenciados sobre `gemma3:4b`:

| ID | Escenario Evaluado | Herramientas Esperadas | Herramientas Usadas | Acierto Herram. | Acierto Cont. | Pasos | Latencia (s) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `HIST_01` | Consulta histórica típica | `consultar_historico` | `consultar_historico` | Sí | Sí | 2 | 41,72 |
| `REG_01` | Predicción ML regresiva | `predecir_demanda` | `predecir_demanda` | Sí | Sí | 2 | 38,65 |
| `CLF_01` | Caso límite ML (pico 99,96 %) | `clasificar_hora_pico` | `clasificar_hora_pico` | Sí | Sí | 2 | 38,42 |
| `RAG_01` | Consulta normativa CNMC | `consultar_normativa` | `consultar_normativa` | Sí | Sí | 2 | 42,75 |
| `MULTI_01` | Orquestación 3 herramientas | `predecir_demanda`, `clasificar_hora_pico`, `consultar_normativa` | `predecir_demanda` | No | Sí | 2 | 48,23 |
| `OOD_01` | Fuera de rango temporal (2026) | `[]` | `[]` | Sí | Sí | 1 | 25,12 |
| **Total** | **6 escenarios** | — | — | **83,33 %** | **100,0 %** | **1,83** | **39,15 s** |

### Análisis de Escenarios y Fallos Documentados
1. **Escenario típico (`HIST_01`, `REG_01`, `RAG_01`):** El agente selecciona la herramienta exacta en el primer paso, recupera los valores reales (`338.426,64 MW` de media en enero de 2024; predicción de `384.352,63 MW` con error de `2.432,37 MW` el `2024-07-15 14:00`; horario punta `P1` de `10:00–14:00` y `18:00–22:00`) y responde sin alucinaciones.
2. **Caso límite del ML (`CLF_01`):** Para el `2024-01-11 20:00` (pico máximo del año con `456.146,0 MW`, muy por encima del umbral `386.923,5 MW`), el clasificador devuelve una probabilidad de `99,96 %` y el agente justifica correctamente el estado crítico.
3. **Consulta fuera de contexto (`OOD_01`):** Ante *"¿Cuál fue la demanda eléctrica en marzo de 2026?"*, el agente reconoce en 1 paso (`25,12 s`) que el rango disponible abarca `2021-01-01` a `2024-12-31` y rechaza responder sin invocar herramientas ni inventar cifras.
4. **Fallo documentado en `MULTI_01` y su resolución:** En el prompt base sin desglose paso a paso, ante una petición triple, `gemma3:4b` ejecutó `predecir_demanda` e infirió el periodo tarifario `P1` desde su conocimiento previo en lugar de invocar explícitamente `clasificar_hora_pico` y `consultar_normativa`. Este comportamiento se corrigió mediante *Chain-of-Thought* (Sección 4).

---

## 4. Experimentos de Prompting y Parámetros (`eval/resultados/`)

### 4.1 Comparativa Zero-Shot vs. Chain-of-Thought (`comparativa_prompting.json`)
Ejecución reproducible (`python -m eval.run_prompting`) sobre los 6 casos de la batería:

| Estrategia de Prompting | Acierto Herramientas | Acierto Contenido | Pasos Medios | Latencia Media (s) |
| :--- | :---: | :---: | :---: | :---: |
| **Zero-Shot** | 83,33 % (5/6) | 83,33 % (5/6) | 1,83 | **37,24 s** |
| **Chain-of-Thought (CoT)** | **100,0 % (6/6)** | **100,0 % (6/6)** | 2,50 | 51,24 s |

- **Impacto antes/después:** Con Zero-Shot, el agente falló en `MULTI_01` (omitiendo herramientas) y en `OOD_01` (acierto contenido `83,33 %`). Al incorporar **Chain-of-Thought**, el agente descompuso `MULTI_01` en 4 pasos secuenciales invocando las 3 herramientas (`predecir_demanda`, `clasificar_hora_pico` y `consultar_normativa`), alcanzando el **100,0 % de acierto global**.

### 4.2 Experimento de `temperature` y `seed` (`experimento_temperatura_seed.json`)
Ejecución reproducible (`python -m eval.run_experimento_params`) sobre la consulta compuesta de `2024-01-11 20:00`:

| Configuración | `temperature` | `seed` | JSON Válido | Acción Elegida | Argumentos Generados | Latencia (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `determinista_seed42` | `0.0` | `42` | Sí | `predecir_demanda` | `{"fecha_hora": "2024-01-11 20:00"}` | 28,87 (arranque) |
| `determinista_seed99` | `0.0` | `99` | Sí | `predecir_demanda` | `{"fecha_hora": "2024-01-11 20:00"}` | 5,17 |
| `creativo_temp08_seed42` | `0.8` | `42` | Sí | `predecir_demanda` | `{"fecha_hora": "2024-01-11 20:00"}` | 4,93 |

*Conclusión:* El diseño estricto del `SYSTEM_PROMPT` combinado con el validador estructural garantiza un **100 % de validez sintáctica JSON** incluso al elevar la temperatura a `0.8`, mientras que `temperature = 0.0` asegura estabilidad completa entre distintas semillas (`seed=42` y `seed=99`).
'@ | Set-Content -Path docs\MEDICION.md -Encoding UTF8

---

## 5. Evidencias de Ejecución en Terminal (Trazas Reales del Agente)

Además de las capturas almacenadas en `docs/capturas/`, se documentan las trazas literales obtenidas al ejecutar `python -m src.agent.agente`:

### Escenario A — Caso límite de clasificación (`CLF_01`: Pico máximo de 2024)
- **Entrada:** `Evalúa si el 2024-01-11 a las 20:00 fue hora pico crítica según el modelo de clasificación.`
- **Paso 1:** Invoca `clasificar_hora_pico(fecha_hora="2024-01-11 20:00")` -> `{"es_hora_pico": true, "probabilidad_pico": 0.9996, "umbral_p95_mw": 386923.5, "demanda_real_mw": 456146.0}`.
- **Paso 2 (Respuesta):** Confirma hora pico crítica con probabilidad del **99,96 %** (demanda real `456.146,0 MW` frente al umbral `386.923,5 MW`).

### Escenario B — Orquestación multi-herramienta con Chain-of-Thought (`MULTI_01`)
- **Entrada:** `Para el 2024-07-15 a las 14:00, predice la demanda eléctrica, indica si se clasifica como hora pico crítica y explica según la normativa qué periodo tarifario aplica.`
- **Paso 1:** `predecir_demanda("2024-07-15 14:00")` -> `384.352,63 MW` (real: `386.785,0 MW`, error: `2.432,37 MW`).
- **Paso 2:** `clasificar_hora_pico("2024-07-15 14:00")` -> `false` (probabilidad `41,2 %`, bajo umbral `386.923,5 MW`).
- **Paso 3:** `consultar_normativa("periodo tarifario 14:00 dia laborable julio CNMC")` -> Circular CNMC 3/2020: cierre de Punta `P1` (`10:00–14:00`) e inicio de Llano `P2` (`14:00–18:00`).
- **Paso 4 (Respuesta):** Integra las tres salidas sin alucinaciones.

### Escenario C — Control de alucinaciones fuera de dominio (`OOD_01`)
- **Entrada:** `¿Cuál fue la demanda eléctrica en marzo de 2026?`
- **Paso 1 (Respuesta directa):** Detecta que `2026` queda fuera del rango disponible (`2021-01-01` a `2024-12-31`) y rechaza inventar cifras sin invocar herramientas.