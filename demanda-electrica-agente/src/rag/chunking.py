"""Estrategias de chunking comparadas."""
"""Carga y troceado (chunking) de documentos del corpus en data/corpus/."""
import re
from pathlib import Path

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, CORPUS_DIR

ARCHIVOS_EXCLUIDOS = {"LICENCIAS.md", ".gitkeep"}
EXTENSIONES_VALIDAS = {".md", ".txt"}


def _limpiar_texto(texto: str) -> str:
    texto = texto.replace("\r\n", "\n")
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def trocear_texto(
    texto: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Divide texto en bloques de ~chunk_size caracteres con solapamiento."""
    texto = _limpiar_texto(texto)
    if not texto:
        return []
    if len(texto) <= chunk_size:
        return [texto]

    chunks = []
    inicio = 0
    n = len(texto)

    while inicio < n:
        fin = min(inicio + chunk_size, n)
        if fin < n:
            # Intentar cortar en salto de párrafo o punto final para no partir frases
            corte_parrafo = texto.rfind("\n\n", inicio + chunk_size // 2, fin)
            corte_punto = texto.rfind(". ", inicio + chunk_size // 2, fin)
            if corte_parrafo != -1:
                fin = corte_parrafo + 2
            elif corte_punto != -1:
                fin = corte_punto + 2

        fragmento = texto[inicio:fin].strip()
        if fragmento:
            chunks.append(fragmento)

        if fin >= n:
            break
        inicio = max(fin - overlap, inicio + 1)

    return chunks


def cargar_y_trocear_corpus(corpus_dir: Path = CORPUS_DIR) -> list[dict]:
    """Recorre data/corpus/ y devuelve una lista de chunks con metadatos."""
    documentos = []
    for ruta in sorted(corpus_dir.rglob("*")):
        if not ruta.is_file():
            continue
        if ruta.name in ARCHIVOS_EXCLUIDOS or ruta.suffix.lower() not in EXTENSIONES_VALIDAS:
            continue

        contenido = ruta.read_text(encoding="utf-8")
        fragmentos = trocear_texto(contenido)
        for idx, frag in enumerate(fragmentos):
            documentos.append(
                {
                    "id_global": len(documentos),
                    "fuente": ruta.relative_to(corpus_dir).as_posix(),
                    "chunk_idx": idx,
                    "texto": frag,
                }
            )
    return documentos


if __name__ == "__main__":
    chunks = cargar_y_trocear_corpus()
    print(f"Total de chunks generados: {len(chunks)}")
    if chunks:
        print("Ejemplo chunk 0:", chunks[0])
