import os, sys, requests, re, json
from pathlib import Path
import streamlit as st 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


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

    with open('data/processed/chunks.json', 'r', encoding = 'utf-8') as f:
        chunks_metadata = json.load(f)

    texts = [chunk['text'] for chunk in chunks_metadata]
    tokenized_texts = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_texts)

    return vectorstore, bm25, chunks_metadata

def extract_year(text):
    match = re.search(r'\b(20\d{2})\b', text)

    return int(match.group(1)) if (match) else None 

def hybrid_search(question, vectorstore, bm25, chunks_metadata, k=3, alpha=0.5):
    faiss_docs = vectorstore.similarity_search(question, k=k*3)
    
    tokenized_q = question.split()
    bm25_scores = bm25.get_scores(tokenized_q)
    sorted_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    top_bm25_indices = sorted_indices[:k*3]

    bm25_docs = []
    for idx in top_bm25_indices:
        chunk = chunks_metadata[idx]
        doc = Document(page_content=chunk['text'], metadata=chunk['metadata'])
        bm25_docs.append(doc)
    
    seen_texts = set()
    combined = []
    
    for doc in faiss_docs:
        text = doc.page_content
        if (text not in seen_texts):
            seen_texts.add(text)
            combined.append(doc)
    
    for doc in bm25_docs:
        text = doc.page_content
        if (text not in seen_texts):
            seen_texts.add(text)
            combined.append(doc)
    
    return combined[:k]

def create_history(messages, max_turns = 5):
    '''
    Формирует историю диалога в промт
    '''

    history = messages[-max_turns * 2:] if (len(messages) < max_turns * 2) else messages
    string = ""

    for mes in history:
        if (mes['role'] == 'user'):
            string += f"Вопрос: {mes['content']}\n"
        else:
            string += f"Ответ: {mes['content']}\n"

    return string

def expand_query(question, history_messages):
    '''
    Необходимо дополнять короткие вопросы историей, для лучшего поиска
    '''
    keywords = ['прибыль', 'доход', 'активы', 'рентабельность']
    has_keyword = any(kw in question.lower() for kw in keywords)

    if (not has_keyword and history_messages):
        for mes in reversed(history_messages):
            if (mes['role'] == 'user'):
                last_question = mes['content']

                for kw in keywords:
                    if (kw in last_question.lower()):
                        return f"{kw} {question}"
                break
    return question

def get_answer(question, vectorstore, history_messages, bm25, chunks_metadata, k = 3):
    '''
    Генерация ответа с учетом истории
    '''

    expanded_question = expand_query(question, history_messages)

    docs = hybrid_search(expanded_question, vectorstore, bm25, chunks_metadata, k)
    context = '\n\n---\n\n'.join([doc.page_content for doc in docs])

    history = create_history(history_messages)

    if (history):
        new_block = f"Предыдущий диалог:\n {history}\n"
    else:
        new_block = ""

    prompt = f"""
Ты - эксперт в банковской сфере и документации. 
Ответь на вопрос, используя информацию только из предоставленного контекста.
Если ответа в контексте нет, то необходимо сообщить: "Ответ не найден или отсутствует в документах".

Обрати внимание: если текущий вопрос является уточнением (например: "а за 2024 год?"), используй
информацию из предыдушего диалога, чтобы понять, о чем идет речь.

{new_block}

Контекст:
{context}

Вопрос: {question}

Ответ:
"""

    payload = {
        'model': 'mistral',
        'prompt': prompt,
        'stream': False,
        'options': {
            'num_predict': 512,
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
    .main { background-color: #f0f4f8; }
    .stTextInput > div > div > input { background-color: white; }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        max-width: 80%;
    }
    .user-message {
        background-color: #d1e7ff;
        align-self: flex-end;
    }
    .assistant-message {
        background-color: #e9ecef;
        align-self: flex-start;
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        margin-top: 10px;
        border-left: 3px solid #00337C;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

st.title("ИИ-ассистент по банковским документам")
st.caption("Задавайте вопросы по финансовой отчетности и договорам.")

if ('messages' not in st.session_state): st.session_state.messages = []

for mes in st.session_state.messages:
    with st.chat_message(mes['role']):
        st.markdown(mes['content'])

if (prompt := st.chat_input("Введите запрос:")):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        with st.spinner("Ищу нужный ответ"):
            vectorstore, bm25, chunks_metadata = load_vectorstore()
            answer, docs = get_answer(
                prompt,
                vectorstore,
                st.session_state.messages[:-1],
                bm25,
                chunks_metadata,
                k = 3
            )
            st.markdown(answer)

            if (docs):
                with st.expander("Источники"):
                    for (i, doc) in enumerate(docs):
                        source = doc.metadata.get("source", 'Неизвестно')
                        page = doc.metadata.get('page', '?')
                        st.write(f"**{i + 1}. {source} (стр. {page})**")
                        st.text(doc.page_content[:300] + "..." if (len(doc.page_content) > 300) else doc.page_content)
                        st.divider()
            else:
                st.info("Источники не найдены")
        st.session_state.messages.append({'role': 'assistant', 'content': answer})
    st.rerun()