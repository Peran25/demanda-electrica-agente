# Decisiones de Arquitectura y Diseño

## 1. Adquisición y Estandarización Temporal de Datos
- **Fuentes heterogéneas:** Se combinaron tres fuentes oficiales mediante APIs REST:
  1. **ESIOS (REE, indicador 1293):** Demanda real peninsular horaria entre el 1 de enero de 2021 y el 31 de diciembre de 2024 (35.064 registros horarios).
  2. **Open-Meteo Archive API:** Temperatura horaria a 2 metros en cuatro nodos representativos de la península (Madrid, Barcelona, Valencia y Sevilla), promediada en `temp_media`.
  3. **Nager.Date API:** Calendario de festivos nacionales en España (`es_festivo`).
- **Estandarización UTC y gestión de cambios de hora (DST):** Para evitar duplicidades o huecos en los cambios de horario de verano/invierno en `Europe/Madrid`, todas las series se alinearon primero en índice UTC (`dt_utc`) antes de proyectar las variables de calendario locales (`dt_madrid`).
- **Localización escalar segura:** En las herramientas del agente, las consultas puntuales utilizan `tz_localize("Europe/Madrid", ambiguous=True, nonexistent="shift_forward")` para garantizar robustez incluso en horas de transición DST.

## 2. Ingeniería de Características y Modelos de Machine Learning
- **Prevención de fuga de datos (Data Leakage):** El particionado temporal se realizó de forma estrictamente cronológica: entrenamiento (`2021–2023`, 26.280 horas) y evaluación (`2024`, 8.784 horas).
- **Variables predictoras:**
  - *Calendario cíclico:* codificación trigonométrica (`sin`/`cos`) de hora del día, día de la semana y mes, junto con indicadores de fin de semana y festivo nacional.
  - *Meteorología no lineal:* temperatura media y grados-día de calefacción (`hdd_18 = max(0, 18 - T)`) y refrigeración (`cdd_22 = max(0, T - 22)`), capturando el efecto en U de la climatización.
  - *Autocorrelación temporal:* retardos (`lag_24h`, `lag_48h`, `lag_168h`) y media móvil de 24 horas desplazada (`rolling_mean_24h`).
- **Selección de modelos:**
  - **Regresión:** Se comparó persistencia semanal (`Naive lag_168h`), `Ridge` y `HistGradientBoostingRegressor`. Se seleccionó `HistGradientBoostingRegressor` por capturar interacciones no lineales entre temperatura, hora y calendario sin requerir escalado adicional.
  - **Clasificación de hora pico crítica:** Definida como demanda superior al percentil 95 del conjunto de entrenamiento (`386.923,50 MW`). Se seleccionó `HistGradientBoostingClassifier` frente a `DummyClassifier` y `LogisticRegression` por su superior equilibrio precisión-cobertura (`F1 = 0,904`).

## 3. Sistema RAG Local (Qdrant + Ollama)
- **Base vectorial local:** Se eligió **Qdrant** en modo persistente en disco (`qdrant_db/`) por no requerir contenedores externos ni servicios en la nube, garantizando ejecución 100 % local y reproducible.
- **Embeddings y chunking:** Se indexó el corpus regulatorio y operativo (`data/corpus/`, 5 documentos en Markdown sobre periodos horarios CNMC 3/2020, generación renovable, gestión de picos REE, olas de frío/calor y metodología de variables) mediante fragmentación semántica con solapamiento y el modelo `nomic-embed-text` (768 dimensiones, similitud coseno).

## 4. Arquitectura del Agente ReAct y Estrategia de Prompting
- **Modelo de lenguaje local:** `gemma3:4b` ejecutado mediante Ollama con `temperature=0.0` para maximizar el determinismo en la generación de llamadas estructuradas.
- **Bucle ReAct con validación estricta:** El agente opera en un ciclo iterativo (`máximo 6 pasos`) donde cada salida del LLM pasa por un validador JSON (`src/agent/validador.py`) que extrae bloques JSON, verifica que la acción pertenezca al catálogo autorizado (`consultar_historico`, `predecir_demanda`, `clasificar_hora_pico`, `consultar_normativa`, `responder`) y reinyecta mensajes de corrección si el formato es inválido.
- **Elección de prompting:** Aunque Zero-Shot ofrece menor latencia en consultas simples, **Chain-of-Thought (CoT)** demostró ser indispensable en consultas compuestas (`MULTI_01`), elevando la selección exacta de herramientas del **83,33 % al 100,0 %**.