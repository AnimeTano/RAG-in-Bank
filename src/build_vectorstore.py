import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_PATH = Path("data/raw")
VECTORSTORE_PATH = Path("data/vectorstore/faiss")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
print('Load documents')

documents = []


# Подгружаем pdf-ки
for file_path in DATA_PATH.glob("*.pdf"):
    year_match = re.search(r'(\d{4})', file_path.stem)
    year = int(year_match.group(1)) if (year_match) else None

    loader = PyPDFLoader(str(file_path))
    docs = loader.load()
    
    for doc in docs:
        doc.metadata["year"] = year   # ключевое поле
    documents.extend(docs)

for file_path in DATA_PATH.glob('*.txt'):
    print(f' - Загружаем: {file_path.name}')
    loader = TextLoader(str(file_path), encoding = 'utf-8')
    documents.extend(loader.load())

print(f"Loaded pages: {len(documents)}")

# Чанкование документов

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 450,
    length_function = len,
    separators = ["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(documents)
print(f'Amount of chunks: {len(chunks)}')

# Эмбеддинги
embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)
vectorstore = FAISS.from_documents(chunks, embeddings)

print(F'Save indexes to {VECTORSTORE_PATH}')
vectorstore.save_local(str(VECTORSTORE_PATH))
print('Index created and saved')