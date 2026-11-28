"""Flujo del agente: orquesta herramientas y validador."""
"""Agente conversacional sobre demanda eléctrica peninsular (Ollama + ML + RAG).

Uso interactivo (desde la raíz del proyecto):
    python -m src.agent.agente
"""
import json
import ollama

from src.agent.herramientas import CATALOGO_HERRAMIENTAS
from src.agent.prompts import SYSTEM_PROMPT_REACT
from src.agent.validador import validar_respuesta_final, validar_salida_llm
from src.config import LLM_MODEL, LLM_SEED, TEMPERATURE


class AgenteDemandaElectrica:
    def __init__(
        self,
        modelo: str = LLM_MODEL,
        temperature: float = TEMPERATURE,
        seed: int = LLM_SEED,
        max_pasos: int = 5,
    ):
        self.modelo = modelo
        self.temperature = temperature
        self.seed = seed
        self.max_pasos = max_pasos

    def _llamar_llm(self, mensajes: list[dict]) -> str:
        resp = ollama.chat(
            model=self.modelo,
            messages=mensajes,
            options={
                "temperature": self.temperature,
                "seed": self.seed,
            },
        )
        return resp["message"]["content"]

    def consultar(self, pregunta: str, verbose: bool = False) -> dict:
        """Ejecuta el ciclo ReAct completo para responder a una pregunta."""
        mensajes = [
            {"role": "system", "content": SYSTEM_PROMPT_REACT},
            {"role": "user", "content": pregunta},
        ]
        traza = []
        herramientas_usadas = []

        for paso in range(1, self.max_pasos + 1):
            salida_raw = self._llamar_llm(mensajes)
            validacion = validar_salida_llm(salida_raw)

            if not validacion.es_valida:
                if verbose:
                    print(f"[Paso {paso}] Error de formato LLM: {validacion.error}")
                mensajes.append({"role": "assistant", "content": salida_raw})
                mensajes.append(
                    {
                        "role": "user",
                        "content": (
                            f"ERROR DE FORMATO: {validacion.error} "
                            "Corrige tu salida y responde ÚNICAMENTE con un JSON válido."
                        ),
                    }
                )
                continue

            if validacion.accion == "responder":
                ok, motivo = validar_respuesta_final(validacion.respuesta, herramientas_usadas)
                if ok:
                    return {
                        "pregunta": pregunta,
                        "respuesta": validacion.respuesta,
                        "herramientas_usadas": herramientas_usadas,
                        "traza": traza,
                        "pasos": paso,
                    }
                mensajes.append({"role": "assistant", "content": salida_raw})
                mensajes.append({"role": "user", "content": f"Corrige tu respuesta: {motivo}"})
                continue

            # Ejecutar herramienta local
            nombre_tool = validacion.accion
            args = validacion.argumentos
            if verbose:
                print(f"[Paso {paso}] Ejecutando {nombre_tool}({args}) ...")

            fn = CATALOGO_HERRAMIENTAS[nombre_tool]
            try:
                resultado = fn(**args)
            except TypeError as e:
                resultado = {"error": f"Argumentos incorrectos para {nombre_tool}: {e}"}
            except Exception as e:
                resultado = {"error": f"Fallo al ejecutar {nombre_tool}: {e}"}

            herramientas_usadas.append(nombre_tool)
            traza.append({"paso": paso, "herramienta": nombre_tool, "argumentos": args, "resultado": resultado})

            mensajes.append({"role": "assistant", "content": salida_raw})
            mensajes.append(
                {
                    "role": "user",
                    "content": (
                        f"OBSERVACIÓN de '{nombre_tool}':\n"
                        f"{json.dumps(resultado, ensure_ascii=False)}\n\n"
                        "Si ya tienes suficiente información, devuelve el JSON con 'accion': 'responder'. "
                        "Si necesitas otro dato, invoca otra herramienta."
                    ),
                }
            )

        return {
            "pregunta": pregunta,
            "respuesta": "No se pudo completar una respuesta estructurada dentro del límite de pasos.",
            "herramientas_usadas": herramientas_usadas,
            "traza": traza,
            "pasos": self.max_pasos,
        }


def main():
    agente = AgenteDemandaElectrica()
    print(f"Agente listo ({LLM_MODEL}). Escribe 'salir' para terminar.\n")
    while True:
        try:
            q = input("Tú > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"salir", "exit", "quit"}:
            break
        res = agente.consultar(q, verbose=True)
        print(f"\nAgente > {res['respuesta']}\n")
        print(f"  [Herramientas: {res['herramientas_usadas']} | Pasos: {res['pasos']}]\n")


if __name__ == "__main__":
    main()