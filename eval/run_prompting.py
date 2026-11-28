"""Experimentos de prompting: seed/temperature, antes/después."""
"""Comparativa de estrategias de prompting (Zero-Shot vs Chain-of-Thought).

Uso (desde la raíz del proyecto):
    python -m eval.run_prompting
"""
import json
import time

from src.agent.agente import AgenteDemandaElectrica
from src.agent.prompts import PROMPT_COT, PROMPT_ZERO_SHOT
from src.config import LLM_MODEL, ROOT

RUTA_BATERIA = ROOT / "eval" / "baterias" / "bateria_preguntas.json"
RUTA_SALIDA = ROOT / "eval" / "resultados" / "comparativa_prompting.json"


def ejecutar_comparativa_prompting() -> dict:
    casos = json.loads(RUTA_BATERIA.read_text(encoding="utf-8"))
    agente = AgenteDemandaElectrica()
    estrategias = {"zero_shot": PROMPT_ZERO_SHOT, "chain_of_thought": PROMPT_COT}
    resultados = {}

    for nombre, plantilla in estrategias.items():
        print(f"\n--- Evaluando estrategia: {nombre} ---")
        registros = []
        for caso in casos:
            prompt_formateado = plantilla.format(pregunta=caso["pregunta"])
            t0 = time.perf_counter()
            res = agente.consultar(prompt_formateado, verbose=False)
            latencia = round(time.perf_counter() - t0, 2)

            esperadas = set(caso["herramientas_esperadas"])
            usadas = set(res["herramientas_usadas"])
            tool_ok = esperadas.issubset(usadas) if esperadas else len(usadas) == 0
            claves_ok = all(k.lower() in res["respuesta"].lower() for k in caso["palabras_clave"])

            registros.append(
                {
                    "id": caso["id"],
                    "acierto_herramientas": tool_ok,
                    "acierto_contenido": claves_ok,
                    "pasos": res["pasos"],
                    "latencia_s": latencia,
                }
            )

        n = len(registros)
        resultados[nombre] = {
            "acierto_herramientas": round(sum(r["acierto_herramientas"] for r in registros) / n, 4),
            "acierto_contenido": round(sum(r["acierto_contenido"] for r in registros) / n, 4),
            "pasos_medios": round(sum(r["pasos"] for r in registros) / n, 2),
            "latencia_media_s": round(sum(r["latencia_s"] for r in registros) / n, 2),
            "detalle": registros,
        }

    salida = {"modelo": LLM_MODEL, "estrategias": resultados}
    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    RUTA_SALIDA.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nComparativa guardada en {RUTA_SALIDA}")
    return salida


if __name__ == "__main__":
    ejecutar_comparativa_prompting()
