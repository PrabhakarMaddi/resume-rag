import chromadb

DB_PATH = "chroma_db"
COLLECTION_NAME = "resumes"

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=DB_PATH)
        _collection = _client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def add_chunks(chunks: list, embeddings) -> None:
    """
    Add chunks + their embeddings to the vector store.
    chunks: list of dicts with 'text' and 'metadata'
    embeddings: numpy array, shape (num_chunks, embedding_dim)
    """
    collection = get_collection()

    ids = [f"{c['metadata']['candidate']}_{c['metadata']['section']}_{i}"
           for i, c in enumerate(chunks)]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),   # Chroma expects plain lists, not numpy arrays
        documents=documents,
        metadatas=metadatas,
    )
    print(f"Added {len(chunks)} chunks to collection '{COLLECTION_NAME}'.")


def query(query_embedding, top_k: int = 3) -> dict:
    """Query the vector store for the top_k most similar chunks."""
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
    )
    return results


def reset_collection():
    """Delete and recreate the collection (useful when re-ingesting from scratch)."""
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'.")
    except Exception:
        pass  # collection didn't exist yet, that's fine


if __name__ == "__main__":
    from loader import load_all_resumes
    from chunker import chunk_resume
    from embedder import embed_texts

    # Start clean each time we run this test
    reset_collection()

    resumes = load_all_resumes()
    all_chunks = []
    for name, text in resumes.items():
        all_chunks.extend(chunk_resume(text, candidate_name=name))

    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts)

    add_chunks(all_chunks, embeddings)

    # Sanity test: search for something skill-related
    test_query = "Education details"
    query_vec = embed_texts([test_query])[0]

    results = query(query_vec, top_k=3)

    print(f"\nQuery: '{test_query}'")
    print("Top matches:\n")
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        print(f"[{meta['candidate']} - {meta['section']}] (distance: {dist:.4f})")
        print(f"  {doc[:150]}...\n")