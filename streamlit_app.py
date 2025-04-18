import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

# Load Hugging Face + embedding models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
qa_model = pipeline("text2text-generation", model="google/flan-t5-base")

# Financial knowledge base
financial_knowledge = [
    "The 50/30/20 rule means 50% of income goes to needs, 30% to wants, and 20% to savings.",
    "The snowball method pays off the smallest debt first. The avalanche method focuses on highest interest first.",
    "Emergency savings should cover 3 to 6 months of expenses.",
    "High credit card debt can affect your credit score and cost more in interest over time."
]

# Build vector index
doc_embeddings = embedder.encode(financial_knowledge)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# Rule-based logic
def analyze_budget(data):
    income = data.get("Income", 0)
    total_expense = sum([v for k, v in data.items() if k != "Income"])
    savings = income - total_expense
    advice = []

    if savings < 0:
        advice.append("⚠️ You're spending more than you earn.")
    elif savings < 0.1 * income:
        advice.append("💡 Try to save at least 10% of your income.")

    if data.get("Credit Card Payment", 0) > 0.3 * income:
        advice.append("⚠️ Credit card spending is high.")

    if data.get("Entertainment", 0) > 0.2 * income:
        advice.append("💡 Entertainment spending may be too high.")

    if not advice:
        advice.append("✅ Great job! Your budget looks balanced.")

    return savings, advice

# LLM fallback using Hugging Face
def get_llm_advice(query):
    query_vec = embedder.encode([query])
    _, I = index.search(np.array(query_vec), k=1)
    context = financial_knowledge[I[0][0]]
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    result = qa_model(prompt, max_length=150)[0]['generated_text']
    return result

# Streamlit UI
st.set_page_config(page_title="Smart Budget & Debt Advisor", page_icon="💰")
st.title("💰 Smart Budget & Debt Advisor")
st.markdown("Input your monthly budget details and ask financial questions below:")

income = st.number_input("Monthly Income", value=2500)
rent = st.number_input("Rent", value=1000)
groceries = st.number_input("Groceries", value=300)
credit_card = st.number_input("Credit Card Payment", value=400)
entertainment = st.number_input("Entertainment", value=400)
query = st.text_input("Ask a financial question (optional)")

if st.button("Analyze"):
    data = {
        "Income": income,
        "Rent": rent,
        "Groceries": groceries,
        "Credit Card Payment": credit_card,
        "Entertainment": entertainment
    }

    savings, tips = analyze_budget(data)

    st.subheader("📊 Budget Summary")
    st.write(f"**Total Savings:** ${savings}")

    st.subheader("🧠 Expert System Advice")
    for tip in tips:
        st.markdown(f"- {tip}")

    if query:
        st.subheader("🤖 LLM Financial Tip")
        st.success(get_llm_advice(query))