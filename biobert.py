import streamlit as st
from transformers import pipeline

# Load the QA pipeline
qa_pipeline = pipeline(
    "question-answering",
    model="D:/BAITAPNHOMML/BAITAPNHOMML/biobert-enhanced-final",
    tokenizer="D:/BAITAPNHOMML/BAITAPNHOMML/biobert-enhanced-final"
)

# Streamlit app
st.title("BioBERT QA Demo")

st.write("Enter a **context** and a **question**, and get the answer!")

# Input fields
context = st.text_area("Context", height=200)
question = st.text_input("Question")

# Predict button
if st.button("Get Answer"):
    if context and question:
        result = qa_pipeline(question=question, context=context)
        st.success(f"**Answer:** {result['answer']}")
        st.info(f"**Confidence:** {result['score']:.2f}")
    else:
        st.warning("Please fill in both the context and the question!")
