import streamlit as st

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

st.title("🩺 Healthcare QnA Chatbot with Dataset Upload")

# Step 1: Upload files
st.sidebar.header("Upload your CSV files")
train_file = st.sidebar.file_uploader("Upload train.csv", type=["csv"])
test_file = st.sidebar.file_uploader("Upload test.csv", type=["csv"])

if train_file and test_file:
    # Step 2: Load and merge datasets
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Basic validation
    if 'question' not in train_df.columns or 'answer' not in train_df.columns:
        st.error("train.csv must have 'question' and 'answer' columns")
    elif 'question' not in test_df.columns or 'answer' not in test_df.columns:
        st.error("test.csv must have 'question' and 'answer' columns")
    else:
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        st.write(f"Total QnA pairs loaded: {len(full_df)}")
        
        # Step 3: Load model and encode
        with st.spinner("Loading model and encoding questions..."):
            model = SentenceTransformer('all-MiniLM-L6-v2')
            question_list = full_df['question'].tolist()
            answer_list = full_df['answer'].tolist()
            embeddings = model.encode(question_list, convert_to_tensor=False)
            
            dimension = len(embeddings[0])
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings))
        
        st.success("Model loaded and FAISS index created!")
        
        # Step 4: User interaction
        user_question = st.text_input("Ask a healthcare question:")

        if user_question:
            query_embedding = model.encode([user_question], convert_to_tensor=False)
            D, I = index.search(np.array(query_embedding), k=1)
            best_idx = I[0][0]
            
            st.markdown(f"**Matched Question:** {question_list[best_idx]}")
            st.markdown(f"**Answer:** {answer_list[best_idx]}")
else:
    st.info("Please upload both train.csv and test.csv files to start.")
