# Agente Inteligente de Predicción y Análisis de Demanda Eléctrica Peninsular

Sistema integral de ciencia de datos, aprendizaje automático, recuperación aumentada por generación (RAG) y agentes conversacionales locales con **Ollama** para el análisis histórico, predicción horaria, detección de picos críticos y consulta normativa de la demanda eléctrica peninsular en España (`2021–2024`).

## Arquitectura del Proyecto

```text
demanda-electrica-agente/
├── data/
│   ├── corpus/                  # Documentación normativa y operativa (CNMC, REE)
│   ├── processed/               # Datasets limpios y matriz de features (2021-2024)
│   └── raw/                     # Descargas brutas de ESIOS, Open-Meteo y Nager.Date
├── docs/
│   ├── DECISIONES.md            # Justificación técnica de arquitectura, ML, RAG y agente
│   └── MEDICION.md              # Resultados experimentales y comparativa de prompting
├── eval/
│   ├── baterias/                # Batería JSON de casos de prueba del agente
│   ├── resultados/              # Métricas ML, batería del agente y comparativa CoT
│   ├── metricas_ml.py           # Cálculo de MAE, RMSE, MAPE, R², F1 y matriz de confusión
│   ├── run_bateria.py           # Evaluación end-to-end del agente ReAct
│   └── run_prompting.py         # Comparativa Zero-Shot vs Chain-of-Thought
├── models/                      # Modelos entrenados serializados (.joblib)
├── src/
│   ├── agent/                   # Bucle ReAct, prompts, validador JSON y 4 herramientas
│   ├── data/                    # Descarga de APIs y limpieza/alineación temporal UTC
│   ├── features/                # Ingeniería de variables meteorológicas, calendario y lags
│   ├── ml/                      # Entrenamiento de modelos de regresión y clasificación
│   ├── rag/                     # Indexación y búsqueda semántica local en Qdrant
│   └── config.py                # Rutas, hiperparámetros y configuración de Ollama
├── tests/
│   └── test_pipeline_y_agente.py # Suite de verificación automática con pytest
└── requirements.txt