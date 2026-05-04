import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PDF_PATH = "data/integrals.pdf"
CHROMA_DIR = "./chroma_db"

# 1. Чистим старую базу, если есть
if Path(CHROMA_DIR).exists():
    shutil.rmtree(CHROMA_DIR)
    print(f"Старая база удалена.")

# 2. Загружаем PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print(f"Загружено страниц: {len(documents)}")

# 3. Режем на чанки
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(documents)
print(f"Получилось чанков: {len(chunks)}")

# 4. Эмбеддинги и запись в Chroma
embeddings = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-m3"
)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
)
print(f"Чанков в векторной базе: {vectorstore._collection.count()}")
print("Индексация завершена.")