# 🤖 NHS Chatbot – Retrieval-Based QA Using BERT

This project is a **retrieval-based healthcare chatbot** built using **Sentence-BERT embeddings**, **FAISS similarity search**, and a **Streamlit-style Python interface (notebook)**.

It allows users to upload healthcare question-answer datasets, and then interact with a chatbot that retrieves the most relevant answer based on the uploaded data.

---

## 📁 Project Structure

```
NHS-chatbot/
├── pre-processed-data/
│   ├── HF_new_traindata.csv       # Cleaned training dataset
│   └── HF_new_testdata.csv        # Cleaned test dataset
│
├── original-data/
│   ├── HF_traindata.csv           # Raw training data
│   └── HF_testdata.csv            # Raw test data
│
├── Retrived Based Chatbot Using Bert Model.ipynb   # Main chatbot code
├── Data cleaning and pre-processing code.ipynb     # EDA and cleaning code
└── README.md
```

---

## 🚀 Project Workflow

### 1. **Original Data**

- Found in `original-data/` folder
- Files:
  - `HF_traindata.csv`
  - `HF_testdata.csv`
- These contain original healthcare QnA data in two columns: `question` and `answer`.

### 2. **Data Preprocessing**

- Performed using `Data cleaning and pre-processing code.ipynb`
- Tasks include:
  - Cleaning text
  - Formatting inconsistent entries
  - Removing duplicates
  - Validating data structure
- Cleaned outputs saved as:
  - `HF_new_traindata.csv`
  - `HF_new_testdata.csv`
    (located in `pre-processed-data/`)

### 3. **Chatbot Model**

- Implemented in the notebook: `Retrived Based Chatbot Using Bert Model.ipynb`
- Uses:
  - `sentence-transformers` (MiniLM BERT model)
  - `faiss-cpu` (for fast vector similarity search)
- How it works:
  - User uploads preprocessed CSVs
  - The chatbot merges the QnA data
  - It encodes questions into vector space
  - FAISS retrieves the closest matching question
  - The matched answer is displayed to the user

---

How to run this
Save the above code in a file named app.py

Run APP :

In VS code Terminal
pip install streamlit sentence-transformers faiss-cpu pandas

streamlit run app.py
Open the browser URL that Streamlit prints

Upload your train.csv and test.csv in the sidebar

Ask your healthcare questions — get matched answers!

## 🧠 Future Development

Currently, this is a **retrieval-based system**, meaning the answers must exist in the dataset.

The next phase of this project will explore **Generative AI**, where:

- Users can ask **any** healthcare question
- A **language model (e.g., FLAN-T5 or BioGPT)** generates answers dynamically
- Will improve generalization and real-world usability

---

## 📬 Contact

Project by Harkesh Dudhat  
Exploring GenAI solutions in the healthcare domain  
Connect on GitHub or LinkedIn for collaboration!

---
