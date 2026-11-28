@'
# Documento de Decisiones Técnicas y Justificación

Este documento recoge las decisiones de ingeniería adoptadas en el diseño del sistema, justificando cada elección frente a las alternativas descartadas.

---

## 1. Adquisición y Estandarización Temporal de Datos
- **Fuentes reales combinadas:** En lugar de datos sintéticos, se integraron tres APIs oficiales:
  1. **ESIOS (REE, indicador 1293):** Demanda real peninsular horaria (`2021–2024`, 35.064 horas).
  2. **Open-Meteo Archive API:** Temperatura horaria a 2 metros en Madrid, Barcelona, Valencia y Sevilla, agregadas en `temp_media` para representar el gradiente térmico peninsular sin sobreajustar a una única estación meteorológica.
  3. **Nager.Date API:** Calendario oficial de festivos nacionales en España (`es_festivo`), determinante en la caída industrial de la demanda.
- **Estandarización en UTC y localización segura (`Europe/Madrid`):** Los cambios de horario de verano/invierno generan horas duplicadas (octubre) e inexistentes (marzo). Todas las series se unificaron primero en `dt_utc` para garantizar continuidad horaria estricta, y en las herramientas puntuales del agente se aplicó `tz_localize("Europe/Madrid", ambiguous=True, nonexistent="shift_forward")` para evitar excepciones en horas de transición DST.

---

## 2. Ingeniería de Características y Selección de Modelos de ML
- **Particionado cronológico estricto (`train`: 2021–2023, 26.106 filas | `test`: 2024, 8.782 filas):** Se descartó la validación cruzada aleatoria (`K-Fold` estándar) porque en series temporales introduce fuga de datos del futuro hacia el pasado (*data leakage*).
- **Representación no lineal y cíclica:**
  - *Grados-día (`hdd_18` y `cdd_22`):* La relación entre temperatura y consumo eléctrico tiene forma de U (aumenta tanto con frío extremo por calefacción como con calor extremo por aire acondicionado). Incorporar `max(0, 18 - T)` y `max(0, T - 22)` permite capturar ambos regímenes.
  - *Codificación trigonométrica (`sin`/`cos`):* Evita discontinuidades artificiales entre las 23:00 y las 00:00 o entre diciembre y enero.
  - *Autorregresión (`lag_24h`, `lag_48h`, `lag_168h` y `rolling_mean_24h`):* Captura la inercia diaria y el patrón semanal exacto de 168 horas.
- **Comparativa de modelos de regresión:** Se evaluaron modelos lineales regularizados (`Ridge`) frente a ensambles basados en árboles (`RandomForest` y `HistGradientBoosting`). `Ridge` quedó limitado a `RMSE = 10.828,93 MW` (`R² = 0,9499`) al no poder modelar interacciones multiplicativas (ej. temperatura alta en día laborable vs. festivo). `HistGradientBoostingRegressor` redujo el error en un **62,8 %** (`RMSE = 4.025,26 MW`, `R² = 0,9931`) y fue seleccionado por su precisión y velocidad de inferencia.
- **Comparativa de modelos de clasificación (Umbral P95 = `386.923,5 MW`):** Dado que solo el 5 % superior del entrenamiento constituye hora pico crítica, `LogisticRegression` generó 412 falsos positivos (`Precision = 0,6344`, `F1 = 0,7668`). `HistGradientBoostingClassifier` equilibró sensibilidad y precisión (`Precision = 0,8887`, `Recall = 0,9201`, `F1 = 0,9041`), reduciendo los falsos positivos en un **79,4 %**.

---

## 3. Diseño del Sistema RAG (Chunking, Embeddings y Qdrant)
- **Base vectorial local (`Qdrant` en modo persistente `qdrant_db/`):** Se eligió Qdrant en disco frente a soluciones en memoria pura (que obligan a reindexar en cada arranque) o servidores externos (Chroma/Pinecone en la nube), cumpliendo el requisito de ejecución 100 % local y reproducible.
- **Modelo de embeddings (`nomic-embed-text`, 768 dimensiones):** Frente a modelos multilingües ligeros de 384 dimensiones, `nomic-embed-text` ofrece una ventana de contexto amplia (8.192 tokens) y alta resolución semántica en español técnico-regulatorio, ejecutándose nativamente en Ollama sin dependencias de PyTorch/CUDA.
- **Estrategia de chunking documentado:** Los 5 documentos del corpus (`data/corpus/`) se estructuraron por secciones temáticas homogéneas (periodos tarifarios CNMC Circular 3/2020, protocolos de punta REE, impacto meteorológico, integración renovable y diccionario de variables) y se fragmentaron respetando límites de párrafo con solapamiento controlado (`chunk_size` ~500 caracteres, `overlap` ~80 caracteres) mediante similitud coseno. Esto evita cortar tablas horarias a la mitad y garantiza que cada fragmento recupere una regla normativa completa.

---

## 4. Arquitectura del Agente, Parámetros del LLM y Reparto de Herramientas
- **Modelo de lenguaje (`gemma3:4b` en Ollama):** Seleccionado por su equilibrio entre capacidad de seguimiento de instrucciones estructuradas (JSON) y viabilidad de ejecución en CPU local sin GPU dedicada.
- **Hiperparámetros de inferencia (`temperature = 0.0`, `seed = 42`):** En sistemas agénticos que emiten llamadas a funciones JSON, una temperatura alta introduce variabilidad léxica innecesaria en nombres de argumentos. Fijar `temperature = 0.0` maximiza la reproducibilidad y el determinismo.
- **Reparto granular de las 4 herramientas:** Se separaron explícitamente `consultar_historico` (datos observados), `predecir_demanda` (regresión continua), `clasificar_hora_pico` (decisión binaria de riesgo) y `consultar_normativa` (conocimiento no estructurado RAG) siguiendo el principio de responsabilidad única. De este modo, cada herramienta tiene un esquema de entrada mínimo (1 a 3 parámetros), minimizando errores de parametrización por parte de un LLM de 4B parámetros.
'@ | Set-Content -Path docs\DECISIONES.md -Encoding UTF8