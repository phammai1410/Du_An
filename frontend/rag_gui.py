"""Streamlit UI for the PDF RAG assistant."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------------------------------------------------------#
# Paths and configuration
# -----------------------------------------------------------------------------#
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "vectorstore" / "faiss_index"
BACKEND_ROOT = PROJECT_ROOT.parent / "backend"
TOOLS_DIR = BACKEND_ROOT / "tools"
LOCAL_TEI_ROOT = BACKEND_ROOT / "local-llm" / "Embedding"
EMBED_META = INDEX_DIR / "embeddings.json"

DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL")
DEFAULT_CHAT_MODEL = "gpt-4o-mini"

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200

OPENAI_EMBED_MODELS = ["text-embedding-3-small", "text-embedding-3-large"]
EMBED_BACKENDS: Dict[str, str] = {
    "openai": "Open AI Chat GPT Embedding",
    "tei": "Local TEI (text-embeddings-inference)",
}

TEI_MODELS: Dict[str, Dict[str, Any]] = {
    "BAAI/bge-m3": {
        "display": "BAAI bge-m3 0.6B",
        "local_dir": LOCAL_TEI_ROOT / "BAAI-bge-m3",
        "download_script": TOOLS_DIR / "download_bge_m3_tei.py",
        "required_file": "pytorch_model.bin",
    },
    "AITeamVN/Vietnamese_Embedding_v2": {
        "display": "BAAI bge-m3 AITeamVN 0.6B",
        "local_dir": LOCAL_TEI_ROOT / "AITeamVN-Vietnamese_Embedding_v2",
        "download_script": TOOLS_DIR / "download_vietnamese_embedding_v2_tei.py",
        "required_file": "model.safetensors",
    },
    "Alibaba-NLP/gte-multilingual-base": {
        "display": "Alibaba 0.3B",
        "local_dir": LOCAL_TEI_ROOT / "Alibaba-NLP-gte-multilingual-base",
        "download_script": TOOLS_DIR / "download_gte_multilingual_base_tei.py",
        "required_file": "model.safetensors",
    },
    "Qwen/Qwen3-Embedding-0.6B": {
        "display": "Qwen3 0.6B",
        "local_dir": LOCAL_TEI_ROOT / "Qwen-Qwen3-Embedding-0.6B",
        "download_script": TOOLS_DIR / "download_qwen3_embedding_tei.py",
        "required_file": "model.safetensors",
    },
}

# -----------------------------------------------------------------------------#
# Utility helpers
# -----------------------------------------------------------------------------#


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)


def list_pdfs() -> List[Path]:
    return sorted(DATA_DIR.glob("*.pdf"))


def index_exists() -> bool:
    return (INDEX_DIR / "index.faiss").exists() and (INDEX_DIR / "index.pkl").exists()


def save_embed_meta(backend: str, model_name: str) -> None:
    try:
        EMBED_META.parent.mkdir(parents=True, exist_ok=True)
        EMBED_META.write_text(
            json.dumps(
                {"embedding_backend": backend, "embedding_model": model_name},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def load_embed_meta() -> Optional[Dict[str, str]]:
    try:
        if EMBED_META.exists():
            meta = json.loads(EMBED_META.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                backend = meta.get("embedding_backend", "openai")
                model = meta.get("embedding_model")
                if model:
                    return {"embedding_backend": backend, "embedding_model": model}
            elif isinstance(meta, str):
                return {"embedding_backend": "openai", "embedding_model": meta}
    except Exception:
        pass
    return None


class TEIEmbeddings:
    """Client for text-embeddings-inference endpoint."""

    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.session = requests.Session()

    def _request(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        payload = {"input": texts, "model": self.model}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self.session.post(
            f"{self.base_url}/embeddings",
            json=payload,
            headers=headers,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        return [item["embedding"] for item in data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._request(texts)

    def embed_query(self, text: str) -> List[float]:
        result = self._request([text])
        return result[0] if result else []


def make_embeddings_client(backend: str, model_name: str):
    if backend == "openai":
        return OpenAIEmbeddings(model=model_name)
    if backend == "tei":
        base_url = os.getenv("TEI_BASE_URL", "http://localhost:8080")
        api_key = os.getenv("TEI_API_KEY")
        return TEIEmbeddings(base_url=base_url, model=model_name, api_key=api_key)
    raise ValueError(f"Unsupported embedding backend: {backend}")


def tei_model_is_downloaded(model_key: str) -> bool:
    config = TEI_MODELS.get(model_key)
    if not config:
        return False
    required_path = config["local_dir"] / config["required_file"]
    return required_path.exists()


def run_tei_download(model_key: str) -> subprocess.CompletedProcess[str]:
    config = TEI_MODELS.get(model_key)
    if not config:
        raise ValueError(f"Unknown TEI model: {model_key}")

    script_path = config["download_script"]
    if not script_path.exists():
        raise FileNotFoundError(f"Missing download script: {script_path}")

    return subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BACKEND_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


# -----------------------------------------------------------------------------#
# Index utilities
# -----------------------------------------------------------------------------#


def build_index_from_pdfs(
    data_dir: Path,
    index_dir: Path,
    embedding_model: str,
    embedding_backend: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Dict[str, Any]:
    pdfs = list(sorted(data_dir.glob("*.pdf")))
    if not pdfs:
        raise RuntimeError(f"No PDF files found in: {data_dir}")

    docs = []
    for pdf in pdfs:
        loader = PyPDFLoader(str(pdf))
        loaded = loader.load()
        for doc in loaded:
            doc.metadata.setdefault("source", str(pdf))
        docs.extend(loaded)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    splits = splitter.split_documents(docs)

    embeddings = make_embeddings_client(embedding_backend, embedding_model)
    vectorstore = FAISS.from_documents(splits, embedding=embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    save_embed_meta(embedding_backend, embedding_model)

    return {
        "pdf_count": len(pdfs),
        "pages": len(docs),
        "chunks": len(splits),
        "index_dir": str(index_dir),
        "embedding_model": embedding_model,
        "embedding_backend": embedding_backend,
    }


def load_vectorstore(embedding_backend: str, embedding_model: str) -> FAISS:
    embeddings = make_embeddings_client(embedding_backend, embedding_model)
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# -----------------------------------------------------------------------------#
# LLM utilities
# -----------------------------------------------------------------------------#


def format_docs_for_prompt(docs) -> str:
    parts = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        tag = f"(p.{page}) " if page is not None else ""
        parts.append(f"[{src}] {tag}{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def call_llm(chat_model: str, question: str, context: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a precise assistant. Only answer using the provided context. "
                "If the answer is not contained there, reply that you do not know.",
            ),
            ("human", "Question:\n{question}\n\nContext:\n{context}\n\nAnswer in Vietnamese."),
        ]
    )

    llm = ChatOpenAI(model=chat_model, temperature=0)
    messages = prompt.format_messages(question=question, context=context)
    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)


# -----------------------------------------------------------------------------#
# Streamlit helpers
# -----------------------------------------------------------------------------#


def apply_material_theme() -> None:
    """Inject a pastel blue Material-inspired theme into the Streamlit app."""
    st.markdown(
        """
        <style>
        :root {
            --primary-50: #e8f1ff;
            --primary-100: #d4e4ff;
            --primary-200: #afc9ff;
            --primary-300: #8eb9ff;
            --primary-400: #6ea2f6;
            --primary-500: #5a8dee;
            --primary-600: #4779d7;
            --primary-700: #355fb9;
            --text-primary: #1f2a40;
            --text-muted: #4e5d7b;
            --surface: #ffffff;
            --surface-muted: #f4f7ff;
            --border-soft: rgba(90, 141, 238, 0.2);
            --shadow-soft: 0 12px 28px rgba(90, 141, 238, 0.16);
        }

        .stApp {
            background: linear-gradient(155deg, var(--surface-muted) 0%, #edf4ff 40%, #ffffff 100%);
            color: var(--text-primary);
            font-family: "Google Sans", "Segoe UI", sans-serif;
        }

        .stApp [data-testid="stToolbar"] {
            background: transparent;
        }

        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(142, 185, 255, 0.35) 0%, rgba(236, 244, 255, 0.95) 100%);
            border-right: 1px solid var(--border-soft);
            box-shadow: 4px 0 16px rgba(82, 122, 196, 0.05);
        }

        div[data-testid="stSidebar"] > div:first-child {
            padding-top: 1.5rem;
        }

        div[data-testid="stSidebar"] label {
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
        }

        div[data-testid="stSidebar"] .stSelectbox,
        div[data-testid="stSidebar"] .stSlider,
        div[data-testid="stSidebar"] .stTextInput {
            padding: 0.75rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.85);
            box-shadow: 0 10px 25px rgba(90, 141, 238, 0.08);
            border: 1px solid var(--border-soft);
            margin-bottom: 1rem;
        }

        div[data-testid="stSidebar"] .stSelectbox:hover,
        div[data-testid="stSidebar"] .stSlider:hover,
        div[data-testid="stSidebar"] .stTextInput:hover {
            box-shadow: 0 18px 32px rgba(90, 141, 238, 0.14);
        }

        div[data-testid="stSidebar"] input,
        div[data-testid="stSidebar"] textarea {
            border-radius: 12px !important;
            border: 1px solid transparent !important;
            background-color: rgba(255, 255, 255, 0.95);
        }

        div[data-testid="stSidebar"] .stSlider > div[role="slider"] {
            color: var(--primary-600);
        }

        div.stButton > button {
            border-radius: 999px;
            background: linear-gradient(135deg, var(--primary-500), var(--primary-600));
            color: #ffffff;
            font-weight: 600;
            letter-spacing: 0.01em;
            border: none;
            box-shadow: 0 12px 24px rgba(90, 141, 238, 0.24);
        }

        div.stButton > button:hover {
            background: linear-gradient(135deg, var(--primary-600), var(--primary-700));
            box-shadow: 0 16px 28px rgba(71, 121, 215, 0.32);
        }

        section.main > div {
            padding-top: 1rem;
        }

        .hero-banner {
            background: linear-gradient(145deg, rgba(255, 255, 255, 0.9), rgba(232, 241, 255, 0.9));
            border: 1px solid var(--border-soft);
            border-radius: 24px;
            padding: 1.8rem 2rem;
            box-shadow: var(--shadow-soft);
            margin-bottom: 1.5rem;
        }

        .hero-banner h2 {
            margin-bottom: 0.4rem;
            color: var(--primary-600);
            font-weight: 700;
        }

        .hero-banner p {
            margin: 0;
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        div[data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.92);
            border-radius: 20px;
            padding: 1.1rem 1.4rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(90, 141, 238, 0.14);
            box-shadow: 0 12px 28px rgba(15, 76, 129, 0.08);
        }

        div[data-testid="stChatMessage"] pre {
            background: #f8fbff !important;
            border-radius: 14px;
        }

        div[data-testid="stChatMessage"] .stExpander {
            border-radius: 16px;
            background: rgba(232, 241, 255, 0.7);
        }

        div[data-testid="stChatMessage"] .stExpander:hover {
            background: rgba(142, 185, 255, 0.18);
        }

        div[data-testid="stChatInput"] textarea {
            border-radius: 18px;
            border: 1px solid var(--border-soft);
            box-shadow: 0 12px 24px rgba(90, 141, 238, 0.12);
            background: rgba(255, 255, 255, 0.95);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "retriever_k" not in st.session_state:
        st.session_state.retriever_k = 4
    if "chat_model" not in st.session_state:
        st.session_state.chat_model = DEFAULT_CHAT_MODEL
    if "embedding_backend" not in st.session_state:
        if DEFAULT_EMBED_MODEL and DEFAULT_EMBED_MODEL in TEI_MODELS:
            st.session_state.embedding_backend = "tei"
        else:
            st.session_state.embedding_backend = "openai"
    if "embed_model" not in st.session_state:
        if DEFAULT_EMBED_MODEL:
            st.session_state.embed_model = DEFAULT_EMBED_MODEL
        else:
            st.session_state.embed_model = (
                list(TEI_MODELS.keys())[0] if st.session_state.embedding_backend == "tei" else OPENAI_EMBED_MODELS[0]
            )
    if "openai_key" not in st.session_state:
        st.session_state.openai_key = os.getenv("OPENAI_API_KEY") or ""
    if "download_feedback" not in st.session_state:
        st.session_state.download_feedback = None

    if st.session_state.embedding_backend == "openai" and st.session_state.embed_model not in OPENAI_EMBED_MODELS:
        st.session_state.embed_model = OPENAI_EMBED_MODELS[0]
    if st.session_state.embedding_backend == "tei" and st.session_state.embed_model not in TEI_MODELS:
        st.session_state.embed_model = list(TEI_MODELS.keys())[0]


def render_sidebar() -> None:
    st.title("Settings")

    st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your API key here if you do not have a .env file.",
        key="openai_key",
    )
    if st.session_state.openai_key:
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_key

    backend_options = list(EMBED_BACKENDS.keys())
    previous_backend = st.session_state.embedding_backend
    st.selectbox(
        "Embedding source",
        backend_options,
        format_func=lambda key: EMBED_BACKENDS[key],
        key="embedding_backend",
    )
    if st.session_state.embedding_backend != previous_backend:
        st.session_state.embed_model = (
            OPENAI_EMBED_MODELS[0] if st.session_state.embedding_backend == "openai" else list(TEI_MODELS.keys())[0]
        )
        st.rerun()

    if st.session_state.embedding_backend == "openai":
        options = OPENAI_EMBED_MODELS
        if st.session_state.embed_model not in options:
            st.session_state.embed_model = options[0]
        st.selectbox(
            "Embedding model",
            options=options,
            help="Rebuild the index after changing the embedding model.",
            key="embed_model",
        )
    else:
        options = list(TEI_MODELS.keys())
        if st.session_state.embed_model not in options:
            st.session_state.embed_model = options[0]
        st.selectbox(
            "Embedding model",
            options=options,
            format_func=lambda key: TEI_MODELS[key]["display"],
            key="embed_model",
        )

        model_key = st.session_state.embed_model
        config = TEI_MODELS[model_key]
        downloaded = tei_model_is_downloaded(model_key)

        st.markdown("**Local TEI model**")
        st.caption(config["display"])
        status_col, info_col = st.columns([0.25, 0.75])
        with status_col:
            status_text = "ready" if downloaded else "download"
            st.markdown(f"`{status_text}`")
        with info_col:
            st.caption(f"Path: `{config['local_dir']}`")
            if not downloaded:
                if st.button("Download model", key=f"download-{model_key}"):
                    result = run_tei_download(model_key)
                    success = result.returncode == 0 and tei_model_is_downloaded(model_key)
                    if success:
                        st.session_state.download_feedback = (
                            "success",
                            f"Downloaded {config['display']} successfully.",
                        )
                    else:
                        detail = (result.stderr or "").strip() or (result.stdout or "").strip() or "No log output."
                        st.session_state.download_feedback = (
                            "error",
                            f"Failed to download {config['display']} (code {result.returncode}). {detail}",
                        )
                    st.rerun()

    feedback = st.session_state.download_feedback
    if feedback:
        status, message = feedback
        if status == "success":
            st.success(message)
        else:
            st.error(message)
        st.session_state.download_feedback = None

    chat_models = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
    if st.session_state.chat_model not in chat_models:
        st.session_state.chat_model = chat_models[0]
    st.selectbox(
        "Chat model",
        options=chat_models,
        key="chat_model",
    )

    st.slider(
        "Top-k passages",
        min_value=2,
        max_value=10,
        step=1,
        key="retriever_k",
    )

    st.divider()
    if st.button("Rebuild index from PDFs"):
        if st.session_state.embedding_backend == "openai" and not st.session_state.openai_key:
            st.error("Please provide an OpenAI API key first.")
        elif st.session_state.embedding_backend == "tei" and not tei_model_is_downloaded(st.session_state.embed_model):
            st.error("Selected TEI model is not downloaded yet.")
        else:
            with st.spinner("Building FAISS index from PDFs..."):
                try:
                    stats = build_index_from_pdfs(
                        DATA_DIR,
                        INDEX_DIR,
                        embedding_model=st.session_state.embed_model,
                        embedding_backend=st.session_state.embedding_backend,
                        chunk_size=DEFAULT_CHUNK_SIZE,
                        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                    )
                    st.success(
                        "Done! "
                        f"{stats['pdf_count']} PDFs, {stats['pages']} pages -> {stats['chunks']} chunks. "
                        f"Index: {stats['index_dir']} (backend: {stats['embedding_backend']} | model: {stats['embedding_model']})"
                    )
                except Exception as exc:
                    st.error(f"Failed to build index: {exc}")

    if st.button("Clear chat history"):
        st.session_state.history = []

    st.divider()
    index_state = "yes" if index_exists() else "no"
    st.caption(f"PDFs in `data/`: {len(list_pdfs())} | Index present: {index_state}")

    emb_used = load_embed_meta()
    if emb_used:
        backend_label = EMBED_BACKENDS.get(emb_used.get("embedding_backend", "openai"), "Unknown")
        st.caption(f"Index built with: {backend_label} / {emb_used['embedding_model']}")
        if (
            emb_used.get("embedding_backend") != st.session_state.embedding_backend
            or emb_used.get("embedding_model") != st.session_state.embed_model
        ):
            st.warning("The current embedding selection differs from the index. Rebuild to avoid inconsistencies.")


# -----------------------------------------------------------------------------#
# App entry point
# -----------------------------------------------------------------------------#


def main() -> None:
    load_dotenv()
    ensure_dirs()
    init_session()

    st.set_page_config(page_title="RAG over PDFs", page_icon=":books:", layout="wide")
    apply_material_theme()

    with st.sidebar:
        render_sidebar()

    st.title("NEU Research Chatbot")
    st.markdown(
        """
        <div class="hero-banner">
            <h2>Trợ lý nghiên cứu NEU</h2>
            <p>Khai thác tri thức trong tài liệu PDF của bạn với giao diện Material pastel xanh dương nhẹ nhàng.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.openai_key:
        st.info("OpenAI API key is missing. Add it in the sidebar or .env file.")
    if not index_exists():
        st.warning("No FAISS index found. Rebuild the index from the sidebar (or run `python ingest_pdfs.py`).")

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                with st.expander("Sources"):
                    for idx, source in enumerate(turn["sources"], start=1):
                        st.markdown(f"**{idx}.** `{source['source']}` (p.{source.get('page', '?')})")
                        if source.get("snippet"):
                            st.caption(source["snippet"])

    user_question = st.chat_input("Ask something about your documents...")
    if user_question:
        st.session_state.history.append({"role": "user", "content": user_question})

        if not st.session_state.openai_key:
            st.session_state.history.append({"role": "assistant", "content": "Please supply the OpenAI API key first."})
            st.rerun()

        if not index_exists():
            st.session_state.history.append({"role": "assistant", "content": "FAISS index is missing. Rebuild it first."})
            st.rerun()

        if st.session_state.embedding_backend == "tei" and not tei_model_is_downloaded(st.session_state.embed_model):
            st.session_state.history.append({
                "role": "assistant",
                "content": "Selected TEI model is not downloaded. Download it in the sidebar.",
            })
            st.rerun()

        try:
            with st.spinner("Retrieving and reasoning..."):
                vectorstore = load_vectorstore(st.session_state.embedding_backend, st.session_state.embed_model)
                retriever = vectorstore.as_retriever(search_kwargs={"k": st.session_state.retriever_k})
                documents = retriever.get_relevant_documents(user_question)

                context = format_docs_for_prompt(documents)
                answer = call_llm(st.session_state.chat_model, user_question, context)

                sources = []
                for doc in documents:
                    source = doc.metadata.get("source", "unknown")
                    page = doc.metadata.get("page")
                    snippet = doc.page_content[:400] + "..." if doc.page_content and len(doc.page_content) > 420 else doc.page_content
                    sources.append({"source": source, "page": page, "snippet": snippet})

                st.session_state.history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )
        except Exception as exc:
            st.session_state.history.append({"role": "assistant", "content": f"An error occurred: {exc}"})

        st.rerun()


if __name__ == "__main__":
    main()
