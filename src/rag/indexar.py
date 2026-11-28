"""Indexa el corpus en Qdrant con nomic-embed-text (se ejecuta UNA vez)."""
"""Indexación vectorial de data/corpus/ en Qdrant local usando Ollama.

Uso (desde la raíz del proyecto):
    python -m src.rag.indexar
"""
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import COLECCION, EMBEDDING_MODEL, QDRANT_PATH
from src.rag.chunking import cargar_y_trocear_corpus


def _obtener_embedding(texto: str) -> list[float]:
    resp = ollama.embed(model=EMBEDDING_MODEL, input=texto)
    return resp["embeddings"][0]


def indexar_corpus() -> int:
    chunks = cargar_y_trocear_corpus()
    if not chunks:
        raise SystemExit(
            "No se encontraron archivos .md o .txt válidos en data/corpus/ "
            "(recuerda que LICENCIAS.md se excluye automáticamente)."
        )

    print(f"Generando embeddings ({EMBEDDING_MODEL}) para {len(chunks)} fragmentos...")
    vectores = [_obtener_embedding(c["texto"]) for c in chunks]
    dim = len(vectores[0])

    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    cliente = QdrantClient(path=str(QDRANT_PATH))

    # Recrear la colección limpia para evitar duplicados al reindexar
    if cliente.collection_exists(COLECCION):
        cliente.delete_collection(COLECCION)

    cliente.create_collection(
        collection_name=COLECCION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    puntos = [
        PointStruct(
            id=c["id_global"],
            vector=vec,
            payload={
                "fuente": c["fuente"],
                "chunk_idx": c["chunk_idx"],
                "texto": c["texto"],
            },
        )
        for c, vec in zip(chunks, vectores)
    ]

    cliente.upsert(collection_name=COLECCION, points=puntos)
    cliente.close()
    print(f"Indexados {len(puntos)} chunks (dim={dim}) en '{COLECCION}' ({QDRANT_PATH}).")
    return len(puntos)


if __name__ == "__main__":
    indexar_corpus()