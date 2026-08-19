from embedder import embed_texts
from vector_store import query as vector_query


def retrieve(user_query: str, top_k: int = 3) -> list:
    """
    Given a user question, return the top_k most relevant chunks.
    Returns a list of dicts: { text, metadata, distance }
    """
    query_embedding = embed_texts([user_query])[0]
    results = vector_query(query_embedding, top_k=top_k)

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved.append({"text": doc, "metadata": meta, "distance": dist})

    return retrieved

from vector_store import get_collection
print(f"[debug] Collection count: {get_collection().count()}")

if __name__ == "__main__":
    query = "What programming languages does the candidate know?"
    results = retrieve(query, top_k=3)

    print(f"Query: {query}\n")
    for r in results:
        print(f"[{r['metadata']['candidate']} - {r['metadata']['section']}] (distance: {r['distance']:.4f})")
        print(f"  {r['text'][:150]}...\n")