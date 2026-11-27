"""Consulta al RAG: devuelve fragmentos y distancia."""
"""Consulta semántica sobre la colección Qdrant local.
Uso (desde la raíz del proyecto):
    python -m src.rag.buscar "periodos horarios punta llano valle"
"""
import sys
import ollama
from qdrant_client import QdrantClient

from src.config import COLECCION, EMBEDDING_MODEL, QDRANT_PATH, TOP_K


def buscar_normativa(consulta: str, top_k: int = TOP_K) -> list[dict]:
    """Devuelve los top_k fragmentos más relevantes de la base vectorial."""
    if not QDRANT_PATH.exists():
        raise FileNotFoundError(
            f"No existe la base Qdrant en {QDRANT_PATH}. Ejecuta primero: python -m src.rag.indexar"
        )

    resp = ollama.embed(model=EMBEDDING_MODEL, input=consulta)
    vector_consulta = resp["embeddings"][0]

    cliente = QdrantClient(path=str(QDRANT_PATH))
    try:
        resultados = cliente.query_points(
            collection_name=COLECCION,
            query=vector_consulta,
            limit=top_k,
        ).points
    finally:
        cliente.close()

    salida = []
    for r in resultados:
        payload = r.payload or {}
        salida.append(
            {
                "score": round(float(r.score), 4),
                "fuente": payload.get("fuente", "desconocida"),
                "chunk_idx": payload.get("chunk_idx", -1),
                "texto": payload.get("texto", ""),
            }
        )
    return salida


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "periodos horarios tarifa eléctrica"
    for hit in buscar_normativa(query):
        print(f"[{hit['score']}] {hit['fuente']} (chunk {hit['chunk_idx']}):\n{hit['texto']}\n---")