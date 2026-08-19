import ollama
from retriever import retrieve

MODEL_NAME = "qwen2.5:3b"  

def build_prompt(user_query: str, retrieved_chunks: list) -> str:
    context = "\n\n".join(
        f"[{c['metadata']['candidate']} - {c['metadata']['section']}]\n{c['text']}"
        for c in retrieved_chunks
    )

    prompt = f"""You are an assistant answering questions about candidate resumes.
Use ONLY the context below to answer the question. If the answer is not
contained in the context, say "I don't have that information in the resumes."
Do not make up any information.

Context:
{context}

Question: {user_query}

Answer:"""
    return prompt


def generate_answer(user_query: str, top_k: int = 5) -> str:
    retrieved_chunks = retrieve(user_query, top_k=top_k)
        # TEMP DEBUG
    print("\n[debug] Retrieved chunks:")
    for c in retrieved_chunks:
        print(f"  [{c['metadata']['candidate']} - {c['metadata']['section']}] (distance: {c['distance']:.4f})")

    prompt = build_prompt(user_query, retrieved_chunks)

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


if __name__ == "__main__":
    while True:
        user_query = input("\nAsk a question about the resumes (or 'quit'): ")
        if user_query.lower() == "quit":
            break

        answer = generate_answer(user_query)
        print(f"\nAnswer:\n{answer}")