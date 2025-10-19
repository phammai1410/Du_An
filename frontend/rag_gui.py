# rag_cli.py — Streamlit Chat UI for PDF RAG
# Run: streamlit run rag_cli.py

import os
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate

# -----------------------------
# Paths & Config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "vectorstore" / "faiss_index"
EMBED_META = INDEX_DIR / "embeddings.json"  # ghi lại model embeddings đã dùng (nếu rebuild từ UI)

DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL")
DEFAULT_CHAT_MODEL = "gpt-4o-mini"

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200

# -----------------------------
# Utils
# -----------------------------
def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)

def list_pdfs() -> List[Path]:
    return sorted(DATA_DIR.glob("*.pdf"))

def index_exists() -> bool:
    return (INDEX_DIR / "index.faiss").exists() and (INDEX_DIR / "index.pkl").exists()

def save_embed_meta(model_name: str):
    try:
        import json
        EMBED_META.parent.mkdir(parents=True, exist_ok=True)
        EMBED_META.write_text(json.dumps({"embedding_model": model_name}, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def load_embed_meta() -> str | None:
    try:
        import json
        if EMBED_META.exists():
            meta = json.loads(EMBED_META.read_text(encoding="utf-8"))
            return meta.get("embedding_model")
    except Exception:
        pass
    return None

def build_index_from_pdfs(
    data_dir: Path,
    index_dir: Path,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Dict[str, Any]:
    """Ingest PDFs -> split -> embed -> save FAISS."""
    pdfs = list(sorted(data_dir.glob("*.pdf")))
    if not pdfs:
        raise RuntimeError(f"Không tìm thấy file PDF trong thư mục: {data_dir}")

    docs = []
    for p in pdfs:
        loader = PyPDFLoader(str(p))  # text-based PDFs
        loaded = loader.load()        # list[Document], mỗi page là 1 Document
        for d in loaded:
            d.metadata.setdefault("source", str(p))
        docs.extend(loaded)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=embedding_model)
    vs = FAISS.from_documents(splits, embedding=embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    save_embed_meta(embedding_model)

    return {
        "pages": len(docs),
        "chunks": len(splits),
        "pdf_count": len(pdfs),
        "index_dir": str(index_dir),
        "embedding_model": embedding_model,
    }

def load_vectorstore(embedding_model: str) -> FAISS:
    embeddings = OpenAIEmbeddings(model=embedding_model)
    # allow_dangerous_deserialization=True là bắt buộc khi load FAISS pickled metadata
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

def format_docs_for_prompt(docs) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        tag = f"(p.{page}) " if page is not None else ""
        parts.append(f"[{src}] {tag}{d.page_content}")
    return "\n\n---\n\n".join(parts)

def call_llm(chat_model: str, question: str, context: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a precise assistant. Answer using ONLY the provided context. "
                "If the answer is not in the context, say you don't know."
            ),
            (
                "human",
                "Question:\n{question}\n\n"
                "Context:\n{context}\n\n"
                "Return a concise, well-structured answer with citations to sources when applicable."
            ),
        ]
    )
    llm = ChatOpenAI(model=chat_model, temperature=0)
    msgs = prompt.format_messages(question=question, context=context)
    resp = llm.invoke(msgs)
    return resp.content if hasattr(resp, "content") else str(resp)

# -----------------------------
# Streamlit App
# -----------------------------
def init_session():
    if "history" not in st.session_state:
        st.session_state.history = []  # list of {"role": "user"/"assistant", "content": "...", "sources": [...]}
    if "retriever_k" not in st.session_state:
        st.session_state.retriever_k = 4
    if "chat_model" not in st.session_state:
        st.session_state.chat_model = DEFAULT_CHAT_MODEL
    if "embed_model" not in st.session_state:
        st.session_state.embed_model = DEFAULT_EMBED_MODEL
    if "openai_key" not in st.session_state:
        st.session_state.openai_key = os.getenv("OPENAI_API_KEY") or ""

def main():
    load_dotenv()
    ensure_dirs()
    init_session()

    st.set_page_config(page_title="RAG over PDFs", page_icon="📚", layout="wide")

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")

        # API key
        st.session_state.openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_key,
            help="Đặt khóa ở đây nếu bạn chưa cấu hình trong .env",
        )
        if st.session_state.openai_key:
            os.environ["OPENAI_API_KEY"] = st.session_state.openai_key

        st.session_state.chat_model = st.selectbox(
            "Chat model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
            index=0 if st.session_state.chat_model not in ["gpt-4o", "gpt-4.1-mini", "gpt-4.1"] else
                   ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"].index(st.session_state.chat_model),
        )

        st.session_state.embed_model = st.selectbox(
            "Embedding model",
            options=["text-embedding-3-small", "text-embedding-3-large"],
            index=0 if st.session_state.embed_model == "text-embedding-3-small" else 1,
            help="Nếu đổi model, nên Rebuild Index để đồng nhất vector."
        )

        st.session_state.retriever_k = st.slider("Top-k passages", min_value=2, max_value=10, value=st.session_state.retriever_k, step=1)

        st.markdown("---")
        # Rebuild index
        if st.button("🔨 Rebuild Index from PDFs"):
            if not st.session_state.openai_key:
                st.error("Vui lòng nhập OpenAI API Key trước.")
            else:
                with st.spinner("Đang xây dựng FAISS từ PDF..."):
                    try:
                        stats = build_index_from_pdfs(
                            DATA_DIR,
                            INDEX_DIR,
                            embedding_model=st.session_state.embed_model,
                            chunk_size=DEFAULT_CHUNK_SIZE,
                            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                        )
                        st.success(
                            f"Xong! {stats['pdf_count']} PDF, {stats['pages']} trang → {stats['chunks']} chunks. "
                            f"Index: {stats['index_dir']} (emb: {stats['embedding_model']})"
                        )
                    except Exception as e:
                        st.error(f"Lỗi khi build index: {e}")

        if st.button("🧹 Clear chat"):
            st.session_state.history = []

        st.markdown("---")
        # Info
        st.caption(f"📁 PDFs in `data/`: {len(list_pdfs())} | Index: {'✅' if index_exists() else '❌'}")

        emb_used = load_embed_meta()
        if emb_used:
            st.caption(f"ℹ️ Index được build với embeddings: **{emb_used}**")
            if emb_used != st.session_state.embed_model:
                st.warning("Embedding model bạn chọn khác với model đã dùng để build index. "
                           "Nên Rebuild Index để đảm bảo chất lượng truy hồi.")

    # Main Chat Area
    st.title("📚 NEU RESEARCH CHATBOT")

    if not st.session_state.openai_key:
        st.info("⛔ Bạn chưa đặt OpenAI API Key. Nhập ở sidebar hoặc tạo file `.env`.")
    if not index_exists():
        st.warning("⚠️ Chưa có FAISS index. Nhấn **Rebuild Index** ở sidebar (hoặc chạy `python ingest_pdfs.py`).")

    # Render chat history
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            # Show sources (assistant turns)
            if turn.get("sources"):
                with st.expander("📎 Sources"):
                    for i, s in enumerate(turn["sources"], start=1):
                        st.markdown(f"**{i}.** `{s['source']}` (p.{s.get('page', '?')})")
                        if s.get("snippet"):
                            st.caption(s["snippet"])

    # Chat input
    user_q = st.chat_input("Nhập câu hỏi về các PDF của bạn...")
    if user_q:
        # Show user message
        st.session_state.history.append({"role": "user", "content": user_q})

        if not st.session_state.openai_key:
            ans = "Bạn chưa cấu hình OPENAI_API_KEY. Hãy nhập ở sidebar."
            st.session_state.history.append({"role": "assistant", "content": ans})
            st.experimental_rerun()

        if not index_exists():
            ans = "Chưa có FAISS index. Nhấn **Rebuild Index** ở sidebar (hoặc chạy `python ingest_pdfs.py`)."
            st.session_state.history.append({"role": "assistant", "content": ans})
            st.experimental_rerun()

        # Retrieve + answer
        try:
            with st.spinner("🔎 Đang truy hồi và suy luận..."):
                vectorstore = load_vectorstore(st.session_state.embed_model)
                retriever = vectorstore.as_retriever(search_kwargs={"k": st.session_state.retriever_k})
                docs = retriever.get_relevant_documents(user_q)

                # Build prompt context
                context = format_docs_for_prompt(docs)
                answer = call_llm(st.session_state.chat_model, user_q, context)

                # Collect compact source info to show
                sources = []
                for d in docs:
                    source = d.metadata.get("source", "unknown")
                    page = d.metadata.get("page")
                    snippet = (d.page_content[:400] + "…") if d.page_content and len(d.page_content) > 420 else d.page_content
                    sources.append({"source": source, "page": page, "snippet": snippet})

                st.session_state.history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
        except Exception as e:
            st.session_state.history.append({
                "role": "assistant",
                "content": f"Đã xảy ra lỗi: {e}"
            })

        st.rerun()

if __name__ == "__main__":
    main()
