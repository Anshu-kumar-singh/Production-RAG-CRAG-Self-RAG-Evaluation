# 📚 CRAG + Self-RAG PDF Question Answering System

An advanced Retrieval-Augmented Generation (RAG) application that combines **CRAG (Corrective RAG)** and **Self-RAG** techniques to improve retrieval quality, answer grounding, and response reliability.

🚀 **Live Demo:** https://anhu321-production-rag-crag-selfrag.hf.space

The system allows users to upload PDF documents, ask questions, and receive answers generated through a multi-stage retrieval and verification pipeline. It also includes a **Plain RAG baseline** and an **LLM-based comparison module** to evaluate whether **CRAG + Self-RAG** produces better results.

---

# 🚀 Live Demo

🌐 **Try the application here:**

**https://anhu321-production-rag-crag-selfrag.hf.space**

No installation required—simply upload your PDFs and start asking questions.

---

# ✨ Features

## 📄 Document Intelligence

* Upload and query multiple PDF documents
* Automatic document chunking and embedding
* FAISS vector database for semantic retrieval
* OpenAI Embeddings for similarity search

---

## 🧠 CRAG (Corrective RAG)

* Retrieval quality evaluation
* Document relevance scoring
* Retrieval verdict classification:

  * ✅ CORRECT
  * ⚠️ AMBIGUOUS
  * ❌ INCORRECT
* Automatic Tavily web search fallback
* Query rewriting for improved retrieval
* Context refinement through sentence-level filtering

---

## 🔍 Self-RAG

* Retrieval decision gate
* Grounding verification (IsSUP)
* Answer usefulness evaluation (IsUSE)
* Automatic answer revision loop
* Recursive answer correction workflow

---

## 📊 Evaluation & Benchmarking

* Traditional RAG baseline implementation
* Side-by-side comparison against CRAG + Self-RAG
* LLM-based answer evaluator
* Comparison based on:

  * Accuracy
  * Groundedness
  * Completeness
  * Overall Answer Quality

---

## 💻 User Interface

* Streamlit-based interface
* PDF upload support
* Question history sidebar
* Pipeline diagnostics
* Interactive comparison mode

---

# 🏗️ Architecture

The application consists of **three integrated workflows**:

1. **Plain RAG Baseline**
2. **CRAG + Self-RAG Pipeline**
3. **LLM Evaluation Pipeline**

The workflow follows the architecture below:

> **Architecture Diagram**

*(Insert the provided pipeline image here in your GitHub README.)*

```
docs/pipeline.png
```

> *(Move the uploaded image into a folder such as `docs/` or `assets/` and reference it as:)*

```markdown
![CRAG + Self-RAG Pipeline](docs/pipeline.png)
```

The pipeline performs:

* Retrieval decision (Self-RAG)
* PDF retrieval
* Retrieval quality evaluation (CRAG)
* Knowledge refinement
* Query rewriting when needed
* Recursive correction loop
* Final answer generation
* Comparison against a Plain RAG baseline
* LLM-based evaluation to determine the better response

---

# 🛠️ Tech Stack

* LangGraph
* LangChain
* OpenAI GPT-4o Mini
* OpenAI Embeddings
* FAISS
* Tavily Search API
* Streamlit
* PyPDF
* Pydantic
* Python Dotenv

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/crag-self-rag.git
cd crag-self-rag
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

# ▶️ Running the Application

## Option 1: Use the Live Demo

Open your browser and visit:

**https://anhu321-production-rag-crag-selfrag.hf.space**

---

## Option 2: Run Locally

```bash
streamlit run app.py
```

The application will launch in your browser using the local Streamlit server.

---

# 🔄 How It Works

## Step 1 — PDF Processing

Uploaded PDFs are:

1. Loaded using PyPDFLoader
2. Split into semantic chunks
3. Embedded using OpenAI Embeddings
4. Stored in a FAISS vector database

---

## Step 2 — Self-RAG Retrieval Decision

Before retrieval, the system determines whether external knowledge is actually required.

Possible outcomes:

* Retrieval Required → Continue to CRAG
* Retrieval Not Required → Generate answer directly

---

## Step 3 — CRAG Retrieval Evaluation

Retrieved chunks are graded as:

### ✅ CORRECT

Highly relevant evidence found.

### ⚠️ AMBIGUOUS

Partial evidence available.

### ❌ INCORRECT

Retrieved information is not useful.

---

## Step 4 — Knowledge Refinement

Depending on the retrieval quality:

* Relevant context is retained
* Irrelevant sentences are removed
* Query rewriting is performed if necessary
* Tavily web search supplements missing knowledge

---

## Step 5 — Answer Generation

The LLM generates a grounded answer using the refined knowledge.

If insufficient evidence exists, the model responds:

```text
I don't know.
```

---

## Step 6 — Self-RAG Verification

### IsSUP (Grounding Check)

Determines whether the generated answer is supported by the retrieved knowledge.

Possible outcomes:

* fully_supported
* partially_supported
* no_support

Unsupported answers are revised automatically.

---

### IsUSE (Usefulness Check)

Determines whether the answer actually satisfies the user's question.

Possible outcomes:

* useful
* not_useful

If not useful, the workflow rewrites the query and restarts the correction cycle.

---

## Step 7 — Plain RAG Baseline

A traditional RAG pipeline runs independently:

```text
Retrieve → Generate
```

This baseline **does not include**:

* Retrieval grading
* Web search fallback
* Grounding verification
* Answer usefulness evaluation

---

## Step 8 — LLM-Based Evaluation

The system compares:

* Plain RAG Answer (Result A)
* CRAG + Self-RAG Answer (Result B)

An evaluator LLM scores both responses based on:

* Accuracy
* Grounding
* Completeness
* Relevance
* Overall Answer Quality

The comparison results are displayed on the dashboard.

---

# 📂 Project Structure

```text
.
├── app.py
├── backend.py
├── requirements.txt
├── documents/
│   ├── book1.pdf
│   ├── book2.pdf
│   └── book3.pdf
├── docs/
│   └── pipeline.png
├── .env
└── README.md
```

---

# 🚀 Future Improvements

* Hybrid Search (BM25 + Vector Search)
* Source Citations
* Streaming Responses
* Multi-modal Document Support
* Evaluation Dashboard
* LangSmith Monitoring
* Multi-user Deployment
* PostgreSQL Vector Storage

---

# 💡 Why This Project?

Traditional RAG systems often fail when retrieval quality is poor.

This project addresses those limitations by combining:

* ✅ CRAG for retrieval correction
* ✅ Self-RAG for answer verification
* ✅ Query rewriting and web fallback
* ✅ Knowledge refinement
* ✅ Recursive answer improvement
* ✅ Benchmarking against Plain RAG
* ✅ LLM-based answer evaluation

The result is a more reliable, self-correcting, and production-oriented Retrieval-Augmented Generation system.

---

# 📄 License

MIT License
