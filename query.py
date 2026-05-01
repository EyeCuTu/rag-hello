from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "./chroma_db"

# Подключаемся к существующей базе
embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

print(f"В базе чанков: {vectorstore._collection.count()}")

# Поисковый запрос
query = "Что такое двойной интеграл?"
results = vectorstore.similarity_search(query, k=5)

print(f"\n--- Запрос: {query} ---")
for i, doc in enumerate(results):
    page = doc.metadata.get('page', '?')
    print(f"\n[Результат {i+1}] (страница {page})")
    print(doc.page_content[:400])
    print("-" * 60)