# Agente Inteligente de Predicción y Análisis de Demanda Eléctrica Peninsular

Sistema agéntico local (Ollama) que combina **aprendizaje automático**, **RAG** y un **agente con herramientas** para analizar la demanda eléctrica de la España peninsular (`2021–2024`): predicción horaria, detección de picos críticos y consulta de normativa sobre periodos tarifarios.

Trabajo de fin de curso. Cuenta para *Modelos de Inteligencia Artificial* (LLM, agente, RAG) y *Sistemas de Aprendizaje Automático* (ML clásico y su medición).

## Qué hace

| Componente | Qué resuelve | Dónde está |
|---|---|---|
| **Regresión** | Predice la demanda horaria (MWh) comparando varios modelos frente a baselines ingenuos | `src/ml/regresion.py`, `src/ml/baselines.py` |
| **Clasificación** | Detecta horas de pico crítico (clase desbalanceada) | `src/ml/clasificacion.py` |
| **RAG** | Responde sobre normativa y periodos tarifarios con búsqueda semántica en Qdrant | `src/rag/` |
| **Agente** | Bucle ReAct con 4 herramientas (ML + RAG) y validación de la salida en JSON | `src/agent/` |
| **Evaluación** | Baterías de preguntas, métricas de ML y experimentos de prompting, todo reproducible | `eval/` |

El ML forma parte del flujo del agente: el agente invoca los modelos entrenados como herramientas, no como un paso aparte.

## Documentación entregable

| Documento | Ubicación |
|---|---|
| Informe del proyecto (PDF) | `docs/Alcantarilla_Peran_Jaime_FinEvalIA.pdf` (fuente en `docs/INFORME_PROYECTO.md`) |
| Documento de decisiones | `docs/DECISIONES.md` |
| Documento de medición | `docs/MEDICION.md` |
| Capturas de los escenarios | `docs/capturas/` |

## Requisitos

- Python 3.10 o superior
- [Ollama](https://ollama.com) instalado y en ejecución
- Token personal de ESIOS (Red Eléctrica de España)
- Conexión a internet solo para la descarga inicial de datos y de los modelos

## Montaje desde cero

### 1. Clonar e instalar

```bash
git clone https://github.com/Peran25/demanda-electrica-agente.git
cd demanda-electrica-agente
python -m venv .venv
```

Activar el entorno: `source .venv/bin/activate` (Linux/macOS) o `.venv\Scripts\Activate.ps1` (Windows PowerShell). Después:

```bash
pip install -r requirements.txt
```

### 2. Configurar las claves

1. Pide un token de ESIOS escribiendo a `consultasios@ree.es`.
2. Copia la plantilla y rellénala:

```bash
cp .env.example .env        # Windows: Copy-Item .env.example .env
```

```dotenv
ESIOS_TOKEN=tu_token_aqui
```

El archivo `.env` está en `.gitignore`. **No lo subas nunca.**

### 3. Descargar los modelos de Ollama

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

### 4. Regenerar los datos

Los datos crudos (`data/raw/`) **no están en el repositorio**, porque proceden de la API de Red Eléctrica y se regeneran con el script de descarga. Los datos ya procesados sí están en `data/processed/`, así que puedes saltar al paso 6 si solo quieres probar el agente.

```bash
python -m src.data.descarga --fuente esios --indicador 1293
python -m src.data.descarga --fuente openmeteo
python -m src.data.descarga --fuente festivos
```

El script guarda cada tramo mensual en `data/raw/` y **no repite peticiones** de lo ya descargado, respetando las condiciones de uso de la API de REE. Los metadatos de la descarga (fecha y parámetros) quedan en `data/raw/metadata_descarga.json`.

### 5. Limpieza y variables

```bash
python -m src.data.limpieza
python -m src.data.features
```

Genera `data/processed/demanda_limpia.csv` y `data/processed/dataset_features.csv`.

### 6. Entrenar los modelos

```bash
python -m src.ml.baselines
python -m src.ml.regresion
python -m src.ml.clasificacion
```

Guarda los modelos en `models/` (`.joblib`, no versionados) y los resultados en `eval/resultados/`.

### 7. Indexar el corpus del RAG (una sola vez)

```bash
python -m src.rag.indexar
```

Divide los documentos de `data/corpus/` en fragmentos, calcula los embeddings con `nomic-embed-text` y los guarda en una base Qdrant local (`qdrant_db/`, no versionada). La base se persiste en disco: no hace falta reindexar en cada arranque.

### 8. Lanzar el agente

```bash
python -m src.agent.agente
```

### 9. Reproducir la evaluación

```bash
python -m eval.run_bateria             # batería de preguntas del agente
python -m eval.run_prompting           # comparativa de prompting
python -m eval.run_experimento_params  # experimento de temperature y seed
pytest tests/                          # tests automáticos
```

Los resultados se guardan en `eval/resultados/`. Todas las semillas y parámetros se definen en `src/config.py`.

## Estructura del repositorio

```text
demanda-electrica-agente/
├── data/
│   ├── corpus/                  # Documentación normativa del RAG + LICENCIAS.md
│   ├── processed/               # Datasets limpios y matriz de variables (2021-2024)
│   └── raw/                     # Descargas brutas (se regeneran; no versionado)
├── docs/
│   ├── DECISIONES.md            # Justificación técnica de cada decisión
│   ├── MEDICION.md              # Datos, métricas, escenarios y experimentos
│   ├── INFORME_PROYECTO.md      # Informe (fuente del PDF)
│   ├── Alcantarilla_Peran_Jaime_FinEvalIA.pdf
│   └── capturas/                # Capturas de los escenarios evaluados
├── eval/
│   ├── baterias/                # Batería JSON de casos de prueba
│   ├── resultados/              # Métricas ML, batería del agente y experimentos
│   ├── metricas_ml.py           # MAE, RMSE, MAPE, R², matriz de confusión, P/R/F1
│   ├── run_bateria.py           # Evaluación completa del agente
│   ├── run_prompting.py         # Comparativa de prompting
│   └── run_experimento_params.py # Experimento de temperature y seed
├── models/                      # Modelos entrenados (.joblib; se regeneran)
├── src/
│   ├── agent/                   # Bucle ReAct, prompts, validador JSON y 4 herramientas
│   ├── data/                    # Descarga, limpieza/alineación temporal y variables
│   ├── ml/                      # Baselines, regresión y clasificación
│   ├── rag/                     # Chunking, indexación y búsqueda en Qdrant
│   └── config.py                # Rutas, semillas, hiperparámetros y configuración de Ollama
├── tests/
│   └── test_pipeline_y_agente.py
├── .env.example                 # Plantilla de claves (sin valores)
└── requirements.txt
```

## Datos y fuentes

| Fuente | Uso | Acceso |
|---|---|---|
| ESIOS (Red Eléctrica de España), indicador 1293 | Demanda real (variable objetivo) | API con token personal |
| Open-Meteo | Temperatura horaria de Madrid, Barcelona, Valencia y Sevilla | Sin clave |
| Nager.Date | Festivos nacionales | Sin clave |
| CNMC y Red Eléctrica (`data/corpus/`) | Corpus del RAG | Ver `data/corpus/LICENCIAS.md` |

Datos de Red Eléctrica de España (ESIOS), usados con atribución a la fuente. El año 2020 se excluye por el patrón anómalo de consumo durante la pandemia.

## Limitaciones conocidas

- **Temperatura real frente a previsión.** El entrenamiento usa temperatura observada del propio día. En un uso real solo existiría la previsión, así que el error en producción sería algo mayor que el medido.
- **Validación cronológica.** La división entre entrenamiento y prueba respeta el orden temporal, nunca se mezcla al azar.
- **Errores más altos esperados** en festivos, puentes, cambios de hora y olas de calor o frío extremas.

Los resultados, los fallos documentados y su causa están en `docs/MEDICION.md`.

