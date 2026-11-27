# Informe de Evaluación y Métricas del Sistema

## 1. Evaluación de Modelos de Machine Learning (Test: Año 2024)

### 1.1 Regresión de Demanda Horaria (MW)
| Modelo | MAE (MW) | RMSE (MW) | MAPE (%) | R² |
| :--- | :---: | :---: | :---: | :---: |
| Baseline Persistencia (`lag_168h`) | ~14.820 | ~21.450 | ~4,65 % | ~0,842 |
| `HistGradientBoostingRegressor` (Seleccionado) | **~2.980** | **~4.025** | **~0,94 %** | **~0,994** |

### 1.2 Clasificación de Hora Pico Crítica (Umbral P95 = 386.923,5 MW)
| Modelo | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| `HistGradientBoostingClassifier` (Seleccionado) | **~0,991** | **~0,912** | **~0,896** | **~0,904** |

*(Nota: Las métricas detalladas por modelo y matrices de confusión completas se encuentran registradas en `eval/resultados/baselines.json`, `eval/resultados/regresion.json` y `eval/resultados/clasificacion.json`).*

---

## 2. Evaluación End-to-End del Agente (`eval/run_bateria.py`)

Evaluación ejecutada sobre `gemma3:4b` en CPU con una batería de 6 casos que cubren consulta histórica, predicción regresiva, clasificación binaria, recuperación documental RAG, orquestación multi-herramienta y detección de consultas fuera de dominio (`eval/resultados/bateria_agente.json`):

| ID | Categoría | Herramientas Esperadas | Herramientas Usadas | Acierto Herram. | Acierto Cont. | Pasos | Latencia (s) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `HIST_01` | Histórico | `consultar_historico` | `consultar_historico` | Sí | Sí | 2 | 41,72 |
| `REG_01` | Regresión | `predecir_demanda` | `predecir_demanda` | Sí | Sí | 2 | 38,65 |
| `CLF_01` | Clasificación | `clasificar_hora_pico` | `clasificar_hora_pico` | Sí | Sí | 2 | 38,42 |
| `RAG_01` | Normativa RAG | `consultar_normativa` | `consultar_normativa` | Sí | Sí | 2 | 42,75 |
| `MULTI_01` | Multi-herramienta | `predecir_demanda`, `clasificar_hora_pico`, `consultar_normativa` | `predecir_demanda` | No | Sí | 2 | 48,23 |
| `OOD_01` | Fuera de rango | `[]` | `[]` | Sí | Sí | 1 | 25,12 |
| **Global** | **6 casos** | — | — | **83,33 %** | **100,0 %** | **1,83** | **39,15 s** |

### Análisis de Resultados de la Batería
- **Cero alucinaciones en datos fuera de rango (`OOD_01`):** Ante una consulta sobre marzo de 2026, el agente detectó inmediatamente el límite temporal del sistema (`2021-01-01` al `2024-12-31`) en 1 solo paso (`25,12 s`) sin invocar herramientas innecesarias ni inventar cifras.
- **Comportamiento en consultas compuestas (`MULTI_01`):** En el modo estándar, el modelo resolvió `predecir_demanda` y dedujo el resto de la respuesta sin invocar las otras dos herramientas explícitas, logrando el 100 % de acierto en el contenido final pero penalizando la métrica estricta de cobertura de herramientas.

---

## 3. Comparativa de Estrategias de Prompting (`eval/run_prompting.py`)

Se comparó el rendimiento de **Zero-Shot** frente a **Chain-of-Thought (CoT)** sobre la misma batería de 6 casos (`eval/resultados/comparativa_prompting.json`):

| Estrategia | Acierto Herramientas | Acierto Contenido | Pasos Medios | Latencia Media (s) |
| :--- | :---: | :---: | :---: | :---: |
| **Zero-Shot** | 83,33 % (5/6) | 83,33 % (5/6) | 1,83 | **37,24 s** |
| **Chain-of-Thought (CoT)** | **100,0 % (6/6)** | **100,0 % (6/6)** | 2,50 | 51,24 s |

### Conclusiones de la Comparativa
1. **Orquestación completa en Chain-of-Thought:** En `MULTI_01`, CoT descompuso la petición en subobjetivos y ejecutó secuencialmente `predecir_demanda`, `clasificar_hora_pico` y `consultar_normativa` en 4 pasos, alcanzando el **100 % de acierto tanto en herramientas como en contenido**.
2. **Compromiso precisión-latencia:** CoT incrementa la latencia media en `+14,0 s` (+37,6 %) respecto a Zero-Shot debido a la generación explícita del razonamiento y a los pasos adicionales en tareas multi-herramienta (`88,7 s` en `MULTI_01`), lo que confirma su idoneidad para consultas analíticas complejas frente a Zero-Shot para consultas simples de un solo paso.

---

## 4. Verificación Automatizada (`pytest`)
La suite de pruebas unitarias y de integración (`tests/test_pipeline_y_agente.py`) verifica en `5,09 s` (`6/6 PASSED`) la integridad de los datasets procesados, las funciones de métricas ML, la existencia de modelos serializados, la ejecución de las herramientas locales y la robustez del validador JSON.