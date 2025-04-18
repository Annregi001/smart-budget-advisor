import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load your trained model
@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("my_model")
    model = T5ForConditionalGeneration.from_pretrained("my_model")
    return tokenizer, model

tokenizer, model = load_model()

# Expert system logic
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

# Use your trained model for financial advice
def get_llm_advice(query):
    input_text = f"summarize: {query}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary

# Streamlit UI
st.set_page_config(page_title="Smart Budget & Debt Advisor", page_icon="💰")
st.title("💰 Smart Budget & Debt Advisor")
st.markdown("Input your budget details below and ask a financial question.")

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

    st.subheader("🧠 Expert Advice")
    for tip in tips:
        st.markdown(f"- {tip}")

    if query:
        st.subheader("🤖 AI-Based Financial Advice")
        st.success(get_llm_advice(query))