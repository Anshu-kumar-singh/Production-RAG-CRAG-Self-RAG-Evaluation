# CRAG + Self-RAG PDF Question Answering System

An advanced Retrieval-Augmented Generation (RAG) application that combines **CRAG (Corrective RAG)** and **Self-RAG** techniques to improve retrieval quality, answer grounding, and response reliability.

The system allows users to upload PDF documents, ask questions, and receive answers generated through a multi-stage retrieval and verification pipeline. It also includes a **Plain RAG baseline** and an **LLM-based comparison module** to evaluate whether CRAG + Self-RAG produces better results.

---

## Features

### Document Intelligence

* Upload and query multiple PDF documents
* Automatic document chunking and embedding
* FAISS vector database for semantic retrieval
* OpenAI embeddings for similarity search

### CRAG (Corrective RAG)

* Retrieval quality evaluation
* Document relevance scoring
* Retrieval verdict classification:

  * CORRECT
  * AMBIGUOUS
  * INCORRECT
* Automatic web search fallback using Tavily
* Query rewriting for better web retrieval
* Context refinement through sentence-level filtering

### Self-RAG

* Retrieval decision gate
* Grounding verification (IsSUP)
* Answer usefulness evaluation (IsUSE)
* Automatic answer revision loop
* Recursive answer correction workflow

### Evaluation & Benchmarking

* Traditional RAG baseline implementation
* Side-by-side comparison against CRAG + Self-RAG
* LLM-based answer evaluator
* Comparison based on:

  * Accuracy
  * Groundedness
  * Completeness
  * Answer quality

### User Interface

* Streamlit-based interface
* PDF upload support
* Question history sidebar
* Pipeline diagnostics
* Interactive comparison mode

---

# Architecture

## Main Pipeline (CRAG + Self-RAG)

```text
User Question
      │
      ▼
Decide Retrieval Needed?
      │
 ┌────┴────┐
 │         │
 ▼         ▼
Direct    Retrieve
Answer    Documents
              │
              ▼
      Evaluate Documents
              │
              ▼
     CORRECT / AMBIGUOUS /
         INCORRECT
              │
              ▼
      Rewrite Query
              │
              ▼
         Web Search
              │
              ▼
      Knowledge Refinement
              │
              ▼
       Generate Answer
              │
              ▼
     IsSUP Verification
              │
     Supported Answer?
              │
      ┌───────┴───────┐
      │               │
      ▼               ▼
 Accept         Revise Answer
      │               │
      └───────┬───────┘
              ▼
       IsUSE Check
              │
              ▼
       Final Answer
```

## Plain RAG Baseline

```text
User Question
      │
      ▼
Retrieve Documents
      │
      ▼
Generate Answer
      │
      ▼
Plain RAG Output
```

## Answer Comparison Pipeline

```text
CRAG + Self-RAG Answer
            │
            ▼
      Comparison
        Evaluator
            ▲
            │
    Plain RAG Answer
            │
            ▼
 Better Answer Selected
```

---

# Tech Stack

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

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/crag-self-rag.git
cd crag-self-rag
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

# Running the Application

```bash
streamlit run app.py
```

Open the local Streamlit URL displayed in the terminal.

---

# How It Works

## Step 1: PDF Processing

Uploaded PDFs are:

1. Loaded using PyPDFLoader
2. Split into chunks
3. Embedded using OpenAI Embeddings
4. Stored in a FAISS vector database

---

## Step 2: Self-RAG Retrieval Decision

Before retrieving documents, the system determines whether external knowledge is required.

Possible outcomes:

* Retrieval Required → Continue to CRAG pipeline
* Retrieval Not Required → Generate answer directly

---

## Step 3: CRAG Retrieval Evaluation

Retrieved chunks are scored individually.

### CORRECT

At least one document is highly relevant.

### AMBIGUOUS

Some relevant information exists but confidence is limited.

### INCORRECT

Retrieved documents are not useful.

---

## Step 4: Web Search Fallback

For ambiguous or incorrect retrieval:

* Query is rewritten
* Tavily web search is executed
* Web results are converted into documents
* Retrieved knowledge is merged with existing context

---

## Step 5: Knowledge Refinement

Documents are:

* Decomposed into sentences
* Evaluated individually
* Irrelevant sentences removed
* Useful context preserved

This creates a cleaner context for generation.

---

## Step 6: Answer Generation

The LLM generates an answer using only the refined context.

If sufficient information is unavailable, the model returns:

```text
I don't know.
```

---

## Step 7: Self-RAG Verification

### IsSUP (Support Check)

Evaluates whether the answer is grounded in the retrieved context.

Possible outcomes:

* fully_supported
* partially_supported
* no_support

If support is insufficient, the answer is revised.

---

### IsUSE (Usefulness Check)

Evaluates whether the answer actually satisfies the user's question.

Possible outcomes:

* useful
* not_useful

If not useful, the system can restart the correction cycle.

---

## Step 8: Plain RAG Benchmark

A traditional RAG pipeline runs independently:

```text
Retrieve → Generate
```

No:

* Retrieval grading
* Web search
* Grounding checks
* Usefulness checks

This serves as a baseline for evaluation.

---

## Step 9: Answer Comparison

The system compares:

* Plain RAG Answer
* CRAG + Self-RAG Answer

An evaluator model judges which answer performs better based on:

* Accuracy
* Grounding
* Completeness
* Relevance

---

# Project Structure

```text
.
├── app.py
├── backend.py
├── requirements.txt
├── documents/
│   ├── book1.pdf
│   ├── book2.pdf
│   └── book3.pdf
├── .env
└── README.md
```

---

# Future Improvements

* Hybrid Search (BM25 + Vector Search)
* Source Citations
* Streaming Responses
* Multi-modal Document Support
* Evaluation Dashboard
* LangSmith Monitoring
* Multi-user Deployment
* PostgreSQL Vector Storage

---

# Why This Project?

Traditional RAG systems often fail when retrieval quality is poor. This project addresses those limitations by combining:

* CRAG for retrieval correction
* Self-RAG for answer verification
* Web fallback for missing knowledge
* Benchmarking against standard RAG

The result is a more reliable and self-correcting question-answering system.

---

# License

MIT License
