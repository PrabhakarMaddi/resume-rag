from sentence_transformers import SentenceTransformer
import numpy as np

# all-MiniLM-L6-v2: small, fast, good baseline. Downloads once (~80MB), then fully local.
MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model = None


def get_model():
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME} (first run downloads it locally)...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list) -> np.ndarray:
    """Embed a list of texts. Returns an array of shape (num_texts, embedding_dim)."""
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings


if __name__ == "__main__":
    from loader import load_all_resumes
    from chunker import chunk_resume

    resumes = load_all_resumes()
    all_chunks = []
    for name, text in resumes.items():
        all_chunks.extend(chunk_resume(text, candidate_name=name))

    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts)

    print(f"\nEmbedded {len(texts)} chunks.")
    print(f"Embedding shape: {embeddings.shape}")  # (num_chunks, embedding_dim)

    # Sanity check: cosine similarity between two Skills sections vs Skills vs Header
    from numpy.linalg import norm

    def cosine_sim(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    skills_idxs = [i for i, c in enumerate(all_chunks) if "Skills" in c["metadata"]["section"]]
    header_idxs = [i for i, c in enumerate(all_chunks) if c["metadata"]["section"] == "Header"]

    if len(skills_idxs) >= 2:
        sim_skills = cosine_sim(embeddings[skills_idxs[0]], embeddings[skills_idxs[1]])
        print(f"\nSimilarity (Skills vs Skills): {sim_skills:.4f}")

    if skills_idxs and header_idxs:
        sim_skills_header = cosine_sim(embeddings[skills_idxs[0]], embeddings[header_idxs[0]])
        print(f"Similarity (Skills vs Header): {sim_skills_header:.4f}")