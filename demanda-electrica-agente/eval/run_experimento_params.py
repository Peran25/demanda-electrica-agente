"""Experimento comparativo de hiperparámetros de inferencia (temperature y seed).

Uso:
    python -m eval.run_experimento_params
"""
import json
import time
import ollama

import src.agent.prompts as prompts_mod
from src.agent.validador import validar_salida_llm
from src.config import LLM_MODEL, ROOT

RUTA_SALIDA = ROOT / "eval" / "resultados" / "experimento_temperatura_seed.json"

PREGUNTA_PRUEBA = (
    "Para el 2024-01-11 20:00, predice la demanda eléctrica, comprueba si es "
    "hora pico crítica e indica qué periodo tarifario corresponde."
)


def _obtener_system_prompt() -> str:
    for nombre in ("SYSTEM_PROMPT", "SISTEMA_AGENTE", "PROMPT_SISTEMA", "REACT_SYSTEM_PROMPT"):
        if hasattr(prompts_mod, nombre):
            return getattr(prompts_mod, nombre)
    for k, v in vars(prompts_mod).items():
        if not k.startswith("_") and isinstance(v, str) and len(v) > 100:
            return v
    raise RuntimeError("No se encontró el prompt de sistema en src/agent/prompts.py")


def ejecutar_experimento() -> dict:
    sys_prompt = _obtener_system_prompt()
    configuraciones = [
        {"nombre": "determinista_seed42", "temperature": 0.0, "seed": 42},
        {"nombre": "determinista_seed99", "temperature": 0.0, "seed": 99},
        {"nombre": "creativo_temp08_seed42", "temperature": 0.8, "seed": 42},
    ]

    resultados = []
    for cfg in configuraciones:
        print(f"Probando {cfg['nombre']} (temp={cfg['temperature']}, seed={cfg['seed']}) ...")
        t0 = time.perf_counter()
        resp = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": PREGUNTA_PRUEBA},
            ],
            options={"temperature": cfg["temperature"], "seed": cfg["seed"]},
        )
        latencia = round(time.perf_counter() - t0, 2)
        crudo = resp["message"]["content"]
        validacion = validar_salida_llm(crudo)

        resultados.append(
            {
                "configuracion": cfg["nombre"],
                "temperature": cfg["temperature"],
                "seed": cfg["seed"],
                "latencia_s": latencia,
                "json_valido": validacion.es_valida,
                "accion_elegida": validacion.accion,
                "argumentos": validacion.argumentos,
                "salida_cruda": crudo,
            }
        )

    salida = {
        "modelo": LLM_MODEL,
        "pregunta": PREGUNTA_PRUEBA,
        "experimentos": resultados,
    }
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    RUTA_SALIDA.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resultados guardados en {RUTA_SALIDA}")
    return salida


if __name__ == "__main__":
    ejecutar_experimento()
