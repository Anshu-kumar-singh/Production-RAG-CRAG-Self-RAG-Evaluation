import streamlit as st
import tempfile
import os
from backend import app, compare_node, set_active_vector_store, build_initial_state

st.set_page_config(page_title="CRAG + Self-RAG", page_icon="📚", layout="centered")

# ── Session State Init ──
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False
if "query" not in st.session_state:
    st.session_state.query = ""
if "phase" not in st.session_state:
    st.session_state.phase = "input"
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"query": ..., "result": ...}

# ── SIDEBAR: Past Questions ──
st.sidebar.title("Past Questions")
if not st.session_state.chat_history:
    st.sidebar.info("No past questions yet.")
else:
    for i, item in enumerate(reversed(st.session_state.chat_history)):
        if st.sidebar.button(item["query"][:50], key=f"hist-{i}"):
            st.session_state.query = item["query"]
            st.session_state.result = item["result"]
            st.session_state.phase = "done"
            st.rerun()

# ── SECTION 1: HERO ──
st.title("📚 CRAG + Self-RAG")
st.caption("LangGraph · Groq · Tavily · FAISS")
st.markdown("Upload your PDFs, ask a question. The agent retrieves, grades its own retrieval, falls back to web search if needed, and checks its own answer before showing it.")
st.markdown("`Retrieve` → `Grade` → `Web Search (if needed)` → `Refine` → `Generate` → `Self-Check`")
st.divider()

# ── SECTION 2: UPLOAD PDFs ──
st.subheader("1. Upload your PDFs")
uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

if st.button("Load Documents", disabled=st.session_state.phase != "input"):
    if uploaded_files:
        paths = []
        for f in uploaded_files:
            # Windows-safe: use tempfile instead of hardcoded /tmp/
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.getbuffer())
                paths.append(tmp.name)
        set_active_vector_store(paths)
        st.session_state.docs_loaded = True
        st.success(f"Loaded {len(paths)} PDF(s).")
    else:
        st.warning("Please upload at least one PDF.")

st.divider()

# ── SECTION 3: INPUT ──
st.subheader("2. Ask your question")
query = st.text_input("Enter your question", placeholder="e.g. Batch normalization vs layer normalization")

if st.button("Run", disabled=st.session_state.phase != "input"):
    if not st.session_state.docs_loaded:
        st.warning("Please load your PDFs first.")
    elif query.strip():
        st.session_state.query = query
        st.session_state.phase = "running"
        st.rerun()

st.divider()

# ── SECTION 4: RUNNING ──
if st.session_state.phase == "running":
    with st.status("Running CRAG + Self-RAG pipeline...", expanded=True) as status:
        st.write("🔎 Retrieving relevant chunks...")
        st.write("📊 Grading retrieval quality...")
        st.write("🌐 Falling back to web search (if needed)...")
        st.write("🧹 Refining context...")
        st.write("🧠 Generating answer...")
        st.write("✅ Self-checking grounding & usefulness...")

        result = app.invoke(
            build_initial_state(st.session_state.query),
            config={"recursion_limit": 80},
        )
        st.session_state.result = result

        status.update(label="✅ Answer ready", state="complete", expanded=False)

    st.session_state.phase = "done"
    st.rerun()

# ── SECTION 5: FINAL ANSWER ──
if st.session_state.phase == "done" and st.session_state.result:
    result = st.session_state.result

    st.success("✅ Done!")
    st.subheader("Answer")
    st.markdown(result.get("answer", ""))

    with st.expander("Pipeline details"):
        st.write("**Retrieval verdict:**", result.get("verdict"))
        st.write("**Reason:**", result.get("reason"))
        st.write("**Web search query (if used):**", result.get("web_query") or "—")
        st.write("**Grounded? (IsSUP):**", result.get("issup"), "| retries:", result.get("retries"))
        st.write("**Useful? (IsUSE):**", result.get("isuse"), "| reason:", result.get("use_reason"))

    st.divider()

    # ── SECTION 6: COMPARE VS PLAIN RAG ──
    st.subheader("3. Compare vs plain RAG")
    if st.button("Run Comparison"):
        with st.spinner("Running plain RAG and comparing..."):
            comparison = compare_node(result)
            st.session_state.result = {**result, **comparison}
        st.rerun()

    if "rag_answer" in result:
        st.markdown("**Plain RAG answer:**")
        st.markdown(result["rag_answer"])
        st.markdown(f"**Better:** {result['better']} — {result['comparison_reason']}")

    st.divider()

    # save to history if not already saved
    existing = [h["query"] for h in st.session_state.chat_history]
    if st.session_state.query not in existing:
        st.session_state.chat_history.append({
            "query": st.session_state.query,
            "result": result,
        })

    if st.button("Ask Another Question"):
        st.session_state.query = ""
        st.session_state.result = None
        st.session_state.phase = "input"
        st.rerun()
