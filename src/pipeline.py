import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import requests
import json


VECTORSTORE_PATH = Path("data/vectorstore/faiss")
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
OLLAMA_URL = "http://localhost:11434/api/generate"

print("Загружаем индекс")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = FAISS.load_local(
    str(VECTORSTORE_PATH),
    embeddings,
    allow_dangerous_deserialization=True
)

print(f"Индекс загружен, чанков: {vectorstore.index.ntotal}")

def ask_question(query, k=3):
    print(f"Ищем релевантные чанки для: '{query}'...")

    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Ты - эксперт в банковской сфере и документации. 
Ответь на вопрос, используя информацию только из предоставленного контекста.
Если ответа в контексте нет, то необходимо сообщить: "Ответ не найден или отсутствует в документах".

Контекст:
{context}

Вопрос: {query}

Ответ:
"""
    
    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 256,
            "temperature": 0.1
        }
    }
    
    print("Отправляем запрос в Ollama")
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)

        if (response.status_code == 200):
            result = response.json()
            answer = result.get("response", "").strip()
            
            print(f"\n Вопрос: {query}\n")
            print(f"Ответ: {answer}\n")
            print("Источники:")

            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", "Неизвестно")
                page = doc.metadata.get("page", "?")
                print(f"   {i+1}. {source} (стр. {page})")
            return answer
        else:
            print(f"Ошибка: статус {response.status_code}")
            print(response.text)

            return None
    except Exception as e:
        print(f"Ошибка при запросе: {e}")

        return None

if (__name__ == "__main__"):
    question = 'Какова чистая прибыль ВТБ в 2023 год?'
    ask_question(question)