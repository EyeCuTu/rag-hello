from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

# === Конфигурация ===
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "claude-haiku-4-5"
TOP_K = 5

# === Промпт-шаблон ===
PROMPT = ChatPromptTemplate.from_template(
"""Ты помощник, который отвечает на вопросы строго на основе предоставленного контекста.

Контекст:
{context}

Вопрос: {question}

Правила:
- Отвечай только на основе контекста выше.
- Если в контексте нет ответа — честно скажи: "В предоставленных материалах нет ответа на этот вопрос."
- Не выдумывай и не додумывай.
- Отвечай на русском языке."""
)


def answer_question(question: str, vectorstore, llm) -> None:
    """Находит релевантные чанки и формирует ответ через LLM."""
    print(f"\n{'=' * 70}")
    print(f"Вопрос: {question}")
    print('=' * 70)

    # Retrieval
    results = vectorstore.similarity_search(question, k=TOP_K)
    pages = [doc.metadata.get('page', '?') for doc in results]
    print(f"Найдено чанков: {len(results)} (страницы: {pages})")

    # Augmentation + Generation
    context = "\n\n---\n\n".join(doc.page_content for doc in results)
    prompt = PROMPT.invoke({"context": context, "question": question})
    response = llm.invoke(prompt)

    print(f"\nОтвет:\n{response.content}")


def main():
    load_dotenv()

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    llm = ChatAnthropic(model=LLM_MODEL, temperature=0)

    print(f"В базе чанков: {vectorstore._collection.count()}")

    questions = [
        "Что такое двойной интеграл?",
        "Как вычислить площадь плоской фигуры с помощью двойного интеграла?",
        "Как Лейбниц придумал интегральное исчисление?",
    ]
    for q in questions:
        answer_question(q, vectorstore, llm)


if __name__ == "__main__":
    main()