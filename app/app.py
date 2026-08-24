import os, sys, requests
from pathlib import Path
import streamlit as st 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VECTORSTORE_PATH = Path("data/vectorstore/faiss")
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
OLLAMA_URL = 'http://localhost:11434/api/generate'


@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        str(VECTORSTORE_PATH),
        embeddings,
        allow_dangerous_deserialization = True 
    )

    return vectorstore


def get_answer(query, k = 3):
    vectorstore = load_vectorstore()

    docs = vectorstore.similarity_search(query, k = k)
    context = '\n\n---\n\n'.join([doc.page_content for doc in docs])

    prompt = f"""
Ты - эксперт в банковской сфере и документации. 
Ответь на вопрос, используя информацию только из предоставленного контекста.
Если ответа в контексте нет, то необходимо сообщить: "Ответ не найден или отсутствует в документах".

Обрати особое внимание на год, указанный в вопросе. В контексте могут быть данные за разные годы. Выбери информацию именно за нужный год.

Контекст:
{context}

Вопрос: {query}

Ответ:
"""

    payload = {
        'model': 'mistral',
        'prompt': prompt,
        'stream': False,
        'options': {
            'num_predict': 256,
            'temperature': 0.1
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json = payload, timeout = 60)

        if (response.status_code == 200):
            result = response.json()
            answer = result.get("response", "").strip()

            return answer, docs
        else:
            return f"Ошибка подключения к Ollama (статус: {response.status_code})", []
    except Exception as e:
        return f"Error: {str(e)}", []


# Интерфейс
st.set_page_config(
    page_title = 'Анализ документов',
    layout = 'centered'
)

st.markdown("""
<style>
    .main {
        background-color: #f0f4f8;
    }
    .stTextInput > div > div > input {
        background-color: white;
    }
    .answer-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00337C;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

st.title("Rag-анализ документов")
st.markdown("*Система отвечает на вопросы по финансовой отчётности и договорам с указанием источников*")

try:
    test_req = requests.get("http://localhost:11434", timeout=2)
    if test_req.status_code != 200:
        st.warning("⚠️ Ollama не отвечает. Убедись, что сервер запущен (ollama serve).")
except:
    st.warning("⚠️ Не удалось подключиться к Ollama. Запусти сервер командой `ollama serve`.")

# Поле ввода
user_question = st.text_input("Введите вопрос по документам:", placeholder="Например: Какова чистая прибыль ВТБ в 2023 году?")

col1, col2 = st.columns([1, 5])
with col1:
    k_value = st.number_input("Количество чанков (k)", min_value=1, max_value=5, value=3, step=1)

if st.button("🔍 Найти ответ", type="primary", use_container_width=True):
    if user_question.strip():
        with st.spinner("Думаю... (может занять 10–30 секунд)"):
            answer, docs = get_answer(user_question, k=k_value)
        
        # Показываем ответ
        st.markdown("### 💬 Ответ")
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
        
        # Показываем источники
        if docs:
            st.markdown("### 📚 Источники")
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", "Неизвестно")
                page = doc.metadata.get("page", "?")
                with st.expander(f"Источник {i+1}: {source} (стр. {page})"):
                    st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
    else:
        st.error("Пожалуйста, введите вопрос.")

# Подвал
st.divider()
st.caption("Разработано с использованием Mistral 7B, FAISS и LangChain")