import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# === Конфигурация ===
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "BAAI/bge-m3"
EVAL_FILE = "eval_questions.json"
TOP_K = 3


def evaluate(vectorstore, questions: list) -> None:
    """Прогоняет evaluation set, считает hit_rate@k."""
    hits = 0
    reciprocal_ranks = []
    total = len(questions)

    print(f"\n{'=' * 70}")
    print(f"Evaluation: модель = {EMBEDDING_MODEL}, top_k = {TOP_K}")
    print('=' * 70)

    for i, item in enumerate(questions, start=1):
        question = item["question"]
        valid_pages = set(item["valid_pages"])

        results = vectorstore.similarity_search(question, k=TOP_K)
        found_pages = [doc.metadata.get("page") for doc in results]
        hit = any(p in valid_pages for p in found_pages)

        rr = 0.0
        for rank, page in enumerate(found_pages, start=1):
            if page in valid_pages:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        if hit:
            hits += 1

        status = "HIT " if hit else "MISS"
        print(f"\n[{i}] {status} | {question}")
        print(f"    Ожидаемые страницы: {sorted(valid_pages)}")
        print(f"    Найдено в top-{TOP_K}:  {found_pages}")
        print(f"    Reciprocal rank: {rr:.3f}")

    mrr = sum(reciprocal_ranks) / total
    print(f"\n{'=' * 70}")
    print(f"hit_rate@{TOP_K} = {hits}/{total} = {hits / total:.0%}")
    print(f"MRR@{TOP_K}      = {mrr:.3f}")
    print('=' * 70)


def main():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    with open(EVAL_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    evaluate(vectorstore, questions)


if __name__ == "__main__":
    main()