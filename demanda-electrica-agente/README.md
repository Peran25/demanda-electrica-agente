# Predicción de demanda eléctrica con agente (RAG + ML + prompting)

Sistema agéntico que predice la demanda eléctrica de España del día siguiente y responde
consultas usando un modelo de ML y un RAG sobre normativa eléctrica.

## Montaje desde cero

1. Clonar el repositorio y crear el entorno:
   ```
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Copiar `.env.example` a `.env` y rellenar `ESIOS_TOKEN` y `AEMET_KEY`
   (token ESIOS: consultasios@ree.es; key AEMET: opendata.aemet.es).
3. Instalar Ollama y descargar los modelos:
   ```
   ollama pull gemma3:4b
   ollama pull nomic-embed-text
   ```
4. Descargar datos: `python -m src.data.descarga`
5. Entrenar modelos: `python -m src.ml.regresion` y `python -m src.ml.clasificacion`
6. Indexar el corpus (una vez): `python -m src.rag.indexar`
7. Lanzar el agente: `python -m src.agent.agente`
8. Reproducir la evaluación: `python eval/run_bateria.py` y `python eval/run_prompting.py`

## Estructura
Ver `docs/` para el informe, el documento de decisiones y el de medición.

## Fuentes de datos
ESIOS y REData (Red Eléctrica), Open-Meteo, calendario de festivos. Citar la fuente según licencia.
