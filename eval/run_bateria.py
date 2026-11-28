"""Lanza la batería de preguntas y guarda resultados en eval/resultados/."""
"""Ejecuta la batería de evaluación end-to-end sobre el agente.

Uso (desde la raíz del proyecto):
    python -m eval.run_bateria
"""
import json
import time

from src.agent.agente import AgenteDemandaElectrica
from src.config import LLM_MODEL, ROOT

RUTA_BATERIA = ROOT / "eval" / "baterias" / "bateria_preguntas.json"
RUTA_SALIDA = ROOT / "eval" / "resultados" / "bateria_agente.json"


def ejecutar_bateria() -> dict:
    casos = json.loads(RUTA_BATERIA.read_text(encoding="utf-8"))
    agente = AgenteDemandaElectrica()
    detalles = []

    for caso in casos:
        print(f"Evaluando [{caso['id']}] {caso['pregunta']} ...")
        t0 = time.perf_counter()
        res = agente.consultar(caso["pregunta"], verbose=False)
        latencia_s = round(time.perf_counter() - t0, 2)

        esperadas = set(caso["herramientas_esperadas"])
        usadas = set(res["herramientas_usadas"])
        tool_ok = esperadas.issubset(usadas) if esperadas else len(usadas) == 0

        resp_lower = res["respuesta"].lower()
        claves_ok = all(k.lower() in resp_lower for k in caso["palabras_clave"])

        detalles.append(
            {
                "id": caso["id"],
                "categoria": caso["categoria"],
                "pregunta": caso["pregunta"],
                "herramientas_esperadas": caso["herramientas_esperadas"],
                "herramientas_usadas": res["herramientas_usadas"],
                "acierto_herramientas": tool_ok,
                "acierto_contenido": claves_ok,
                "pasos": res["pasos"],
                "latencia_s": latencia_s,
                "respuesta": res["respuesta"],
            }
        )

    total = len(detalles)
    resumen = {
        "modelo": LLM_MODEL,
        "total_casos": total,
        "tasa_acierto_herramientas": round(sum(d["acierto_herramientas"] for d in detalles) / total, 4),
        "tasa_acierto_contenido": round(sum(d["acierto_contenido"] for d in detalles) / total, 4),
        "latencia_media_s": round(sum(d["latencia_s"] for d in detalles) / total, 2),
        "casos": detalles,
    }

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    RUTA_SALIDA.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResumen guardado en {RUTA_SALIDA}:")
    print(f"  Acierto herramientas: {resumen['tasa_acierto_herramientas']*100:.1f}%")
    print(f"  Acierto contenido:    {resumen['tasa_acierto_contenido']*100:.1f}%")
    print(f"  Latencia media:       {resumen['latencia_media_s']} s")
    return resumen


if __name__ == "__main__":
    ejecutar_bateria()
