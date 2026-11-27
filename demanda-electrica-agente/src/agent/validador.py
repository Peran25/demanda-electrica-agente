"""Validador de JSON (diagnostico, pasos, ...) con reescritura."""
"""Validación de acciones JSON y respuestas finales del agente."""
import json
import re
from dataclasses import dataclass

from src.agent.herramientas import CATALOGO_HERRAMIENTAS


@dataclass
class AccionValidada:
    es_valida: bool
    accion: str
    argumentos: dict
    respuesta: str
    error: str = ""


def extraer_json(texto: str) -> dict:
    """Extrae y parsea un objeto JSON incluso si viene envuelto en ```json ... ```."""
    limpio = texto.strip()
    bloque = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", limpio, flags=re.DOTALL)
    if bloque:
        limpio = bloque.group(1)
    else:
        inicio = limpio.find("{")
        fin = limpio.rfind("}")
        if inicio != -1 and fin != -1 and fin > inicio:
            limpio = limpio[inicio : fin + 1]
    return json.loads(limpio, strict=False)


def validar_salida_llm(texto_crudo: str) -> AccionValidada:
    """Comprueba que la salida del LLM cumpla el esquema esperado."""
    try:
        data = extraer_json(texto_crudo)
    except Exception as e:
        return AccionValidada(
            es_valida=False,
            accion="",
            argumentos={},
            respuesta="",
            error=f"Salida no es JSON válido ({e}). Debes responder SOLO con JSON.",
        )

    if not isinstance(data, dict) or "accion" not in data:
        return AccionValidada(
            es_valida=False,
            accion="",
            argumentos={},
            respuesta="",
            error="Falta la clave obligatoria 'accion' en el JSON.",
        )

    accion = str(data["accion"]).strip()

    if accion == "responder":
        respuesta = str(data.get("respuesta", "")).strip()
        if len(respuesta) < 10:
            return AccionValidada(
                es_valida=False,
                accion="responder",
                argumentos={},
                respuesta="",
                error="La clave 'respuesta' está vacía o es demasiado corta.",
            )
        return AccionValidada(es_valida=True, accion="responder", argumentos={}, respuesta=respuesta)

    if accion not in CATALOGO_HERRAMIENTAS:
        return AccionValidada(
            es_valida=False,
            accion=accion,
            argumentos={},
            respuesta="",
            error=f"Herramienta desconocida '{accion}'. Válidas: {list(CATALOGO_HERRAMIENTAS.keys())}.",
        )

    argumentos = data.get("argumentos", {})
    if not isinstance(argumentos, dict):
        return AccionValidada(
            es_valida=False,
            accion=accion,
            argumentos={},
            respuesta="",
            error="La clave 'argumentos' debe ser un objeto JSON (diccionario).",
        )

    return AccionValidada(es_valida=True, accion=accion, argumentos=argumentos, respuesta="")


def validar_respuesta_final(respuesta: str, herramientas_usadas: list[str]) -> tuple[bool, str]:
    """Verifica criterios mínimos de calidad antes de entregar la respuesta al usuario."""
    if not respuesta or not respuesta.strip():
        return False, "Respuesta vacía."
    if "TODO" in respuesta or "<tu respuesta" in respuesta:
        return False, "La respuesta contiene plantillas sin rellenar."
    return True, "OK"