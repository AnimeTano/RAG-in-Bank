import json, requests, re
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder


VECTORSTORE_PATH = Path("data/vectorstore/faiss")
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
OLLAMA_URL = "http://localhost:11434/api/generate"
TEST_QUESTIONS_PATH = Path("evaluation/test_questions.json")
OUTPUT_METRICS_PATH = Path("evaluation/report.json")


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        str(VECTORSTORE_PATH),
        embeddings,
        allow_dangerous_deserialization = True
    )

    return vectorstore

def extract_year(question):
    match = re.search(r'\b(20\d{2})\b', question)

    return int(match.group(1)) if (match) else None

def get_retrieved_docs(question, vectorstore, k=5):
    year = extract_year(question)

    docs = vectorstore.similarity_search(question, k=k*3)
    if (year is not None):
        docs = [doc for doc in docs if doc.metadata.get("year") == year]
    return docs[:k]

def contains_number(text, expected):
    """
    Извлекает все числа из expected и проверяет, есть ли хотя бы одно в text
    """
    numbers = re.findall(r'\d+[\.,]?\d*', expected)

    if (not numbers): return False

    for num in numbers:
        num_clean = num.replace(',', '').replace('.', '')
        if (num_clean in text.replace(',', '').replace('.', '')): return True
    return False

def is_relevant(doc, relevant_docs, expected_answer=""):
    source = doc.metadata.get("source", "")
    file_match = any(rel in str(source) for rel in relevant_docs)

    if (not file_match): return False

    if (expected_answer):
        if (re.search(r'\d', expected_answer)):
            return contains_number(doc.page_content, expected_answer)
        else:
            words_expected = set(expected_answer.lower().split())
            words_content = set(doc.page_content.lower().split())
            common = words_expected & words_content
            
            return len(common) / max(len(words_expected), 1) > 0.5
    return True

def compute_metrics(questions, vectorstore, k_list = [1,3,5]):
    """
    Вычисляет Precision@K, MRR, Recall@K для каждого вопроса
    """
    results = []

    for q in questions:
        question = q["question"]
        relevant_docs = q.get("relevant_docs", [])
        expected = q.get("expected_answer", "")
        
        docs = get_retrieved_docs(question, vectorstore, k = max(k_list))
        relevances = [is_relevant(doc, relevant_docs, expected) for doc in docs]
        
        precision_at_k, recall_at_k = {}, {}

        for k in k_list:
            if k > len(relevances): continue

            top_k = relevances[:k]
            precision = sum(top_k) / k
            precision_at_k[f"precision@{k}"] = precision

            total_relevant = sum(relevances)

            if (total_relevant > 0):
                recall = sum(top_k) / total_relevant
            else:
                recall = 0.0
            recall_at_k[f"recall@{k}"] = recall
        
        mrr = 0.0
        for i, rel in enumerate(relevances):
            if (rel):
                mrr = 1.0 / (i + 1)
                break
        
        results.append({
            "question": question,
            "precisions": precision_at_k,
            "recalls": recall_at_k,
            "mrr": mrr,
            "expected_answer": expected,
            "retrieved_count": len(docs)
        })
    
    avg_prec, avg_rec, avg_mrr = {}, {}, 0.0

    for k in k_list:
        prec_key = f"precision@{k}"
        rec_key = f"recall@{k}"
        vals_prec = [r["precisions"].get(prec_key, 0) for r in results]
        vals_rec = [r["recalls"].get(rec_key, 0) for r in results]
        avg_prec[prec_key] = sum(vals_prec) / len(vals_prec)
        avg_rec[rec_key] = sum(vals_rec) / len(vals_rec)
    avg_mrr = sum(r["mrr"] for r in results) / len(results)
    
    return {
        "per_question": results,
        "average": {
            "precision": avg_prec,
            "recall": avg_rec,
            "mrr": avg_mrr
        }
    }

def generate_answer(question, vectorstore, k=3):
    """
    Генерирует ответ через LLM для сравнения с expected
    """
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Ты - эксперт в банковской сфере и документации. 
Ответь на вопрос, используя информацию только из предоставленного контекста.
Если ответа в контексте нет, то необходимо сообщить: "Ответ не найден или отсутствует в документах".

Контекст:
{context}

Вопрос: {question}

Ответ:
"""
    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 256, "temperature": 0.1}
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        if (resp.status_code == 200):
            return resp.json().get("response", "").strip()
        else:
            return f"Ошибка: {resp.status_code}"
    except Exception as e:
        return f"Ошибка: {e}"

def evaluate_answers(questions, vectorstore, compute_llm=False):
    """
    Дополнительно сравнивает ответы LLM с ожидаемыми
    """
    for q in questions:
        if compute_llm:
            llm_answer = generate_answer(q["question"], vectorstore, k=3)
            q["llm_answer"] = llm_answer
            
            expected = q.get("expected_answer", "")
            if expected:
                match = expected.lower() in llm_answer.lower()
                q["answer_match"] = match
            else:
                q["answer_match"] = None
    return questions

def main():
    vectorstore = load_vectorstore()

    with open(TEST_QUESTIONS_PATH, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    metrics = compute_metrics(questions, vectorstore, k_list=[1,3,5])

    metrics["questions_with_answers"] = questions
    
    with open(OUTPUT_METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print("\nСводка по метрикам:")
    avg = metrics["average"]
    print(f"  Precision@1: {avg['precision']['precision@1']:.3f}")
    print(f"  Precision@3: {avg['precision'].get('precision@3', 0):.3f}")
    print(f"  Precision@5: {avg['precision'].get('precision@5', 0):.3f}")
    print(f"  MRR: {avg['mrr']:.3f}")
    
    print("\nОтчёт сохранён в", OUTPUT_METRICS_PATH)

if (__name__ == "__main__"):
    main()