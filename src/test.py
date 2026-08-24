from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path


VECTORSTORE_PATH = Path("data/vectorstore/faiss")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)
vectorstore = FAISS.load_local(str(VECTORSTORE_PATH), embeddings, allow_dangerous_deserialization = True)

query = "Какие финансовые результаты показал ВТБ за 2023 год?"

docs = vectorstore.similarity_search(query, k = 3)

for (i, doc) in enumerate(docs):
    print(f"--- Чанк {i+1} (из {doc.metadata.get('source', 'Unknown')}) ---")
    print(doc.page_content[:500])
    print("-" * 50)