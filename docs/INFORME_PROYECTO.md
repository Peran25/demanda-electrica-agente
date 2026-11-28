# Sistema Agéntico Integral para la Predicción, Detección de Picos y Consulta Normativa de la Demanda Eléctrica Peninsular

**Asignaturas:** Modelos de Inteligencia Artificial (LLM / Agentes / RAG) y Sistemas de Aprendizaje Automático (Machine Learning Clásico y Medición)  
**Repositorio:** `https://github.com/Peran25/demanda-electrica-agente`  
**Stack Tecnológico:** Python 3.14, Scikit-Learn, Qdrant (local), Ollama (`gemma3:4b` + `nomic-embed-text`), Pytest

---

## 1. Resumen Ejecutivo y Problema Resuelto
La gestión del sistema eléctrico peninsular español exige anticipar la demanda horaria (MW), detectar eventos extremos de sobrecarga (horas pico críticas) y aplicar correctamente los periodos tarifarios regulados por la CNMC y Red Eléctrica de España (REE). 

Este proyecto implementa un **sistema agéntico 100 % local** que integra tres pilares de Inteligencia Artificial en un único flujo conversacional verificable:
1. **Pipeline de Datos y Aprendizaje Automático (ML Clásico):** Adquisición desde fuentes oficiales reales (`2021–2024`), ingeniería de variables meteorológicas/calendario/autorregresivas y modelos supervisados de regresión (`RMSE = 4.025,26 MW`, `R² = 0,9931`) y clasificación de horas pico (`F1 = 0,9041`).
2. **Sistema RAG Local:** Base de datos vectorial persistente en **Qdrant** indexada con embeddings locales (`nomic-embed-text`) sobre documentación regulatoria de CNMC y REE.
3. **Agente ReAct con Control Estructural:** Orquestado sobre `gemma3:4b` mediante Ollama, con validación estricta de salidas JSON, control de alucinaciones fuera de rango y razonamiento *Chain-of-Thought* que alcanza el **100 % de acierto** en selección de herramientas y contenido.

---

## 2. Arquitectura del Sistema

```text
+-----------------------------------------------------------------------+
|                        USUARIO (Consulta Natural)                     |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|             AGENTE ReAct (Ollama: gemma3:4b, temp=0.0)                |
|  [System Prompt + Few-Shot/CoT] <---> [Validador JSON Estructural]    |
+-------+-------------------+-------------------+-------------------+---+
        |                   |                   |                   |
        v                   v                   v                   v
+---------------+   +---------------+   +---------------+   +---------------+
| consultar_    |   | predecir_     |   | clasificar_   |   | consultar_    |
| historico     |   | demanda       |   | hora_pico     |   | normativa     |
+-------+-------+   +-------+-------+   +-------+-------+   +-------+-------+
        |                   |                   |                   |
        v                   v                   v                   v
+---------------+   +---------------+   +---------------+   +---------------+
| demanda_      |   | HistGradient  |   | HistGradient  |   | Qdrant Local  |
| limpia.csv    |   | BoostingReg.  |   | BoostingClf.  |   | nomic-embed   |
| (35.064 h)    |   | (R²=0,9931)   |   | (F1=0,9041)   |   | (Corpus REE)  |
+---------------+   +---------------+   +---------------+   +---------------+