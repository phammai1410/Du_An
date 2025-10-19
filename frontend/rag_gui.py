# rag_cli.py — Streamlit Chat UI for PDF RAG
# Run: streamlit run rag_cli.py

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# Paths & Config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "vectorstore" / "faiss_index"
BACKEND_ROOT = PROJECT_ROOT.parent / "backend"
TOOLS_DIR = BACKEND_ROOT / "tools"
LOCAL_TEI_ROOT = BACKEND_ROOT / "local-llm" / "Embedding"
EMBED_META = INDEX_DIR / "embeddings.json"  # ghi lại model embeddings đã dùng (nếu rebuild từ UI)

DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL")
DEFAULT_CHAT_MODEL = "gpt-4o-mini"

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
OPENAI_EMBED_MODELS = ["text-embedding-3-small", "text-embedding-3-large"]
EMBED_BACKENDS: Dict[str, Dict[str, str]] = {
    "openai": {"label_key": "embedding_backend_openai"},
    "tei": {"label_key": "embedding_backend_tei"},
}

TEI_MODELS: Dict[str, Dict[str, Any]] = {
    "BAAI/bge-m3": {
        "display": {
            "en": "BAAI bge-m3 · 0.6B params",
            "vi": "BAAI bge-m3 · 0.6B tham số",
        },
        "local_dir": LOCAL_TEI_ROOT / "BAAI-bge-m3",
        "download_script": TOOLS_DIR / "download_bge_m3_tei.py",
        "required_file": "pytorch_model.bin",
        "base_url": "https://huggingface.co/BAAI/bge-m3/resolve/main",
        "files": [
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
            "modules.json",
            "config_sentence_transformers.json",
            "sentence_bert_config.json",
            "1_Pooling/config.json",
        ],
    },
    "AITeamVN/Vietnamese_Embedding_v2": {
        "display": {
            "en": "bge-m3 Vietnamese finetune · 0.6B params",
            "vi": "bge-m3 tiếng Việt finetune · 0.6B tham số",
        },
        "local_dir": LOCAL_TEI_ROOT / "AITeamVN-Vietnamese_Embedding_v2",
        "download_script": TOOLS_DIR / "download_vietnamese_embedding_v2_tei.py",
        "required_file": "model.safetensors",
        "base_url": "https://huggingface.co/AITeamVN/Vietnamese_Embedding_v2/resolve/main",
        "files": [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
            "modules.json",
            "config_sentence_transformers.json",
            "sentence_bert_config.json",
            "1_Pooling/config.json",
        ],
    },
    "Alibaba-NLP/gte-multilingual-base": {
        "display": {
            "en": "Alibaba gte-multilingual-base · 0.3B params",
            "vi": "Alibaba gte-multilingual-base · 0.3B tham số",
        },
        "local_dir": LOCAL_TEI_ROOT / "Alibaba-NLP-gte-multilingual-base",
        "download_script": TOOLS_DIR / "download_gte_multilingual_base_tei.py",
        "required_file": "model.safetensors",
        "base_url": "https://huggingface.co/Alibaba-NLP/gte-multilingual-base/resolve/main",
        "files": [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "modules.json",
            "sentence_bert_config.json",
            "1_Pooling/config.json",
        ],
    },
    "Qwen/Qwen3-Embedding-0.6B": {
        "display": {
            "en": "Qwen3 Embedding · 0.6B params",
            "vi": "Qwen3 Embedding · 0.6B tham số",
        },
        "local_dir": LOCAL_TEI_ROOT / "Qwen-Qwen3-Embedding-0.6B",
        "download_script": TOOLS_DIR / "download_qwen3_embedding_tei.py",
        "required_file": "model.safetensors",
        "base_url": "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/main",
        "files": [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "modules.json",
            "config_sentence_transformers.json",
            "1_Pooling/config.json",
        ],
    },
}

LANG_CHOICES = [("vi", "Tiếng Việt"), ("en", "English")]
LANG_LABEL_MAP = dict(LANG_CHOICES)
LANG_STRINGS: Dict[str, Dict[str, str]] = {
    "vi": {
        "language_label": "Ngôn ngữ",
        "app_title": "📚 NEU RESEARCH CHATBOT",
        "sidebar_title": "⚙️ Cài đặt",
        "openai_key_label": "OpenAI API Key",
        "openai_key_help": "Nhập khóa nếu bạn chưa cấu hình trong .env",
        "embedding_source_label": "Nguồn embedding",
        "embedding_backend_openai": "OpenAI ChatGPT Embedding",
        "embedding_backend_tei": "Local TEI (text-embeddings-inference)",
        "embedding_model_label": "Embedding model",
        "embedding_model_help_openai": "Nếu đổi model, hãy rebuild index để đồng nhất vector.",
        "embedding_model_help_tei": "Chọn model TEI mong muốn hoặc tải xuống.",
        "local_model_section_title": "Thông tin model TEI",
        "status_available": "Đã sẵn sàng",
        "status_not_available": "Chưa có sẵn",
        "download_button": "Tải xuống",
        "downloading": "Đang tải... {percent}%",
        "download_success": "Đã tải {model} thành công.",
        "download_failed": "Không thể tải {model} (mã {code}). {detail}",
        "download_no_log": "Không có log.",
        "download_location": "Đường dẫn: {path}",
        "rebuild_button": "Rebuild Index từ PDF",
        "rebuild_need_key": "Vui lòng nhập OpenAI API Key trước.",
        "rebuild_need_model": "Model TEI chưa được tải. Tải trước khi build index.",
        "rebuild_progress": "Đang xây dựng FAISS từ PDF...",
        "rebuild_success": "Xong! {pdfs} PDF, {pages} trang → {chunks} chunks. Index: {index} (backend: {backend} | model: {model})",
        "rebuild_failed": "Lỗi khi build index: {error}",
        "clear_chat": "Xoá lịch sử chat",
        "pdfs_info": "📁 PDFs trong `data/`: {count} | Index: {state}",
        "pdfs_state_ready": "đã sẵn sàng",
        "pdfs_state_missing": "chưa có",
        "index_built_with": "ℹ️ Index được build với: **{backend_label} · {model}**",
        "index_mismatch": "Embedding hiện tại khác với model đã dùng để build index. Hãy rebuild để đồng nhất.",
        "no_key_info": "🚫 Bạn chưa đặt OpenAI API Key. Nhập ở sidebar hoặc tạo file `.env`.",
        "no_index_warning": "⚠️ Chưa có FAISS index. Nhấn Rebuild Index ở sidebar (hoặc chạy `python ingest_pdfs.py`).",
        "sources_label": "📚 Nguồn",
        "chat_input_placeholder": "Nhập câu hỏi về tài liệu của bạn...",
        "missing_key_response": "Bạn chưa cấu hình OPENAI_API_KEY. Hãy nhập ở sidebar.",
        "missing_index_response": "Chưa có FAISS index. Nhấn Rebuild Index ở sidebar (hoặc chạy `python ingest_pdfs.py`).",
        "missing_model_response": "Model TEI bạn chọn chưa được tải. Hãy tải từ sidebar trước.",
        "retrieving_spinner": "🔎 Đang truy hồi và suy luận...",
        "error_prefix": "Đã xảy ra lỗi: {error}",
        "status_label": "Trạng thái",
        "available_label": "Sẵn sàng",
        "not_available_label": "Chưa tải",
        "download_heading": "Tải xuống model",
        "progress_label": "Tiến độ tải",
        "download_complete": "Hoàn tất!",
        "chat_model_label": "Mô hình chat",
        "topk_slider_label": "Top-k passages",
    },
    "en": {
        "language_label": "Language",
        "app_title": "📚 NEU RESEARCH CHATBOT",
        "sidebar_title": "⚙️ Settings",
        "openai_key_label": "OpenAI API Key",
        "openai_key_help": "Enter your key here if `.env` is not configured.",
        "embedding_source_label": "Embedding source",
        "embedding_backend_openai": "OpenAI ChatGPT Embedding",
        "embedding_backend_tei": "Local TEI (text-embeddings-inference)",
        "embedding_model_label": "Embedding model",
        "embedding_model_help_openai": "If you change the model, rebuild the index to keep vectors aligned.",
        "embedding_model_help_tei": "Select a TEI model or download it below.",
        "local_model_section_title": "TEI model details",
        "status_available": "Available",
        "status_not_available": "Not available",
        "download_button": "Download",
        "downloading": "Downloading... {percent}%",
        "download_success": "Downloaded {model} successfully.",
        "download_failed": "Failed to download {model} (code {code}). {detail}",
        "download_no_log": "No logs.",
        "download_location": "Path: {path}",
        "rebuild_button": "Rebuild Index from PDFs",
        "rebuild_need_key": "Please enter an OpenAI API Key first.",
        "rebuild_need_model": "The TEI model is not available. Download it before rebuilding the index.",
        "rebuild_progress": "Building FAISS from PDFs...",
        "rebuild_success": "Done! {pdfs} PDFs, {pages} pages → {chunks} chunks. Index: {index} (backend: {backend} | model: {model})",
        "rebuild_failed": "Failed to build index: {error}",
        "clear_chat": "Clear chat",
        "pdfs_info": "📁 PDFs in `data/`: {count} | Index: {state}",
        "pdfs_state_ready": "ready",
        "pdfs_state_missing": "missing",
        "index_built_with": "ℹ️ Index built with: **{backend_label} · {model}**",
        "index_mismatch": "Current embedding differs from the index. Rebuild to keep things consistent.",
        "no_key_info": "🚫 OpenAI API Key is missing. Add it in the sidebar or `.env` file.",
        "no_index_warning": "⚠️ FAISS index not found. Rebuild it from the sidebar (or run `python ingest_pdfs.py`).",
        "sources_label": "📚 Sources",
        "chat_input_placeholder": "Ask a question about your documents...",
        "missing_key_response": "OpenAI API key is missing. Please enter it in the sidebar.",
        "missing_index_response": "FAISS index is missing. Rebuild it from the sidebar (or run `python ingest_pdfs.py`).",
        "missing_model_response": "The TEI embedding model is not downloaded. Please download it from the sidebar first.",
        "retrieving_spinner": "🔎 Retrieving and reasoning...",
        "error_prefix": "An error occurred: {error}",
        "status_label": "Status",
        "available_label": "Available",
        "not_available_label": "Not available",
        "download_heading": "Download model",
        "progress_label": "Download progress",
        "download_complete": "Completed!",
        "chat_model_label": "Chat model",
        "topk_slider_label": "Top-k passages",
    },
}


def tr(key: str, **kwargs: Any) -> str:
    lang = st.session_state.get("language", "vi")
    catalog = LANG_STRINGS.get(lang, LANG_STRINGS["vi"])
    text = catalog.get(key, LANG_STRINGS["vi"].get(key, key))
    return text.format(**kwargs)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --primary-color: #1E88E5;
            --accent-color: #90CAF9;
            --text-color: #0D47A1;
            --background-color: #E3F2FD;
            --sidebar-color: #E8F1FF;
        }
        body, .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
            font-family: "Segoe UI", sans-serif;
        }
        .stSidebar {
            background-color: var(--sidebar-color) !important;
        }
        .stSidebar [data-testid="stHeader"] {
            background-color: transparent;
        }
        .stButton button, .stDownloadButton button {
            background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
            color: #ffffff;
            border: none;
            border-radius: 8px;
        }
        .stButton button:hover, .stDownloadButton button:hover {
            background: linear-gradient(135deg, #1565C0, var(--primary-color));
        }
        .stTextInput > div > div > input {
            border-radius: 8px;
        }
        .small-label {
            font-size: 0.85rem;
            color: #1A237E;
        }
        .status-tag {
            font-size: 0.75rem;
            border-radius: 999px;
            padding: 2px 10px;
            display: inline-block;
        }
        .status-ok {
            background-color: #C5E1A5;
            color: #33691E;
        }
        .status-missing {
            background-color: #FFCDD2;
            color: #B71C1C;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

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
            meta_raw = EMBED_META.read_text(encoding="utf-8")
            meta = json.loads(meta_raw)
            if isinstance(meta, dict):
                backend = meta.get("embedding_backend")
                model = meta.get("embedding_model") or meta.get("model")
                if backend and model:
                    return {"embedding_backend": backend, "embedding_model": model}
                if model:
                    return {"embedding_backend": "openai", "embedding_model": model}
            elif isinstance(meta, str):
                return {"embedding_backend": "openai", "embedding_model": meta}
    except Exception:
        pass
    return None


def backend_label_from_key(key: str) -> str:
    info = EMBED_BACKENDS.get(key, EMBED_BACKENDS["openai"])
    return tr(info["label_key"])


def backend_key_from_label(label: str) -> str:
    for key, value in EMBED_BACKENDS.items():
        if tr(value["label_key"]) == label:
            return key
    return "openai"


class TEIEmbeddings:
    """Minimal client for Hugging Face text-embeddings-inference server."""

    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._session = requests.Session()

    def _request_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        payload = {"input": texts, "model": self.model}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self._session.post(
            f"{self.base_url}/embeddings",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json().get("data", [])
        return [item["embedding"] for item in data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._request_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        result = self._request_embeddings([text])
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


def tei_display_name(model_key: str) -> str:
    config = TEI_MODELS.get(model_key)
    if not config:
        return model_key
    lang = st.session_state.get("language", "vi")
    return config["display"].get(lang, config["display"].get("en", model_key))


def download_tei_model_with_progress(
    model_key: str,
    progress_placeholder: "st.delta_generator.DeltaGenerator",
    status_placeholder: "st.delta_generator.DeltaGenerator",
) -> Tuple[bool, Optional[str]]:
    config = TEI_MODELS.get(model_key)
    if not config:
        message = tr("download_failed", model=model_key, code="?", detail="Unknown model")
        status_placeholder.error(message)
        return False, message

    base_url = config["base_url"]
    files = config["files"]
    target_dir = config["local_dir"]
    target_dir.mkdir(parents=True, exist_ok=True)

    file_sizes: Dict[str, int] = {}
    total_size = 0
    for rel in files:
        url = f"{base_url}/{rel}?download=1"
        try:
            head = requests.head(url, allow_redirects=True, timeout=10)
            size = int(head.headers.get("content-length", "0"))
        except Exception:
            size = 0
        file_sizes[rel] = size
        total_size += size

    downloaded_bytes = 0
    progress_bar = progress_placeholder.progress(0.0)

    try:
        for rel in files:
            destination_path = target_dir / rel
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            url = f"{base_url}/{rel}?download=1"
            with requests.get(url, stream=True, timeout=300) as response:
                response.raise_for_status()
                chunk_size = 1024 * 256
                expected = file_sizes.get(rel, 0)
                with open(destination_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
                        if total_size > 0:
                            percent = min(downloaded_bytes / total_size, 1.0)
                        else:
                            percent = 0.0
                        progress_bar.progress(percent)
                        status_placeholder.caption(tr("downloading", percent=int(percent * 100)))
                if expected and destination_path.stat().st_size < expected:
                    raise IOError(f"Incomplete download for {rel}")

        progress_bar.progress(1.0)
        status_placeholder.caption(tr("download_complete"))
        st.session_state.download_progress[model_key] = 1.0
        return True, None
    except Exception as exc:
        progress_placeholder.empty()
        detail = str(exc)
        if not detail:
            detail = tr("download_no_log")
        code = getattr(exc, "errno", "")
        message = tr(
            "download_failed",
            model=tei_display_name(model_key),
            code=code or "?",
            detail=detail,
        )
        status_placeholder.error(message)
        return False, message

def build_index_from_pdfs(
    data_dir: Path,
    index_dir: Path,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    embedding_backend: str = "openai",
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

    embeddings = make_embeddings_client(embedding_backend, embedding_model)
    vs = FAISS.from_documents(splits, embedding=embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    save_embed_meta(embedding_backend, embedding_model)

    return {
        "pages": len(docs),
        "chunks": len(splits),
        "pdf_count": len(pdfs),
        "index_dir": str(index_dir),
        "embedding_model": embedding_model,
        "embedding_backend": embedding_backend,
    }

def load_vectorstore(embedding_backend: str, embedding_model: str) -> FAISS:
    embeddings = make_embeddings_client(embedding_backend, embedding_model)
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
    if "language" not in st.session_state:
        st.session_state.language = "vi"
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
                next(iter(TEI_MODELS)) if st.session_state.embedding_backend == "tei" else OPENAI_EMBED_MODELS[0]
            )
    if "openai_key" not in st.session_state:
        st.session_state.openai_key = os.getenv("OPENAI_API_KEY") or ""
    if "download_feedback" not in st.session_state:
        st.session_state.download_feedback = None
    if "download_progress" not in st.session_state:
        st.session_state.download_progress = {}
    if "downloading_model" not in st.session_state:
        st.session_state.downloading_model = None

    if st.session_state.embedding_backend == "openai" and st.session_state.embed_model not in OPENAI_EMBED_MODELS:
        st.session_state.embed_model = OPENAI_EMBED_MODELS[0]
    if st.session_state.embedding_backend == "tei" and st.session_state.embed_model not in TEI_MODELS:
        st.session_state.embed_model = next(iter(TEI_MODELS))

def main():
    load_dotenv()
    ensure_dirs()
    init_session()

    st.set_page_config(page_title="RAG over PDFs", page_icon="📚", layout="wide")

    # Sidebar
    with st.sidebar:
        st.title(tr("sidebar_title"))

        lang_options = [code for code, _ in LANG_CHOICES]
        current_lang_index = lang_options.index(st.session_state.language) if st.session_state.language in lang_options else 0
        selected_language = st.selectbox(
            tr("language_label"),
            lang_options,
            index=current_lang_index,
            format_func=lambda code: LANG_LABEL_MAP.get(code, code),
        )
        if selected_language != st.session_state.language:
            st.session_state.language = selected_language
            st.experimental_rerun()

        st.session_state.openai_key = st.text_input(
            tr("openai_key_label"),
            type="password",
            value=st.session_state.openai_key,
            help=tr("openai_key_help"),
        )
        if st.session_state.openai_key:
            os.environ["OPENAI_API_KEY"] = st.session_state.openai_key

        backend_options = list(EMBED_BACKENDS.keys())
        backend_index = backend_options.index(st.session_state.embedding_backend)
        selected_backend = st.selectbox(
            tr("embedding_source_label"),
            backend_options,
            index=backend_index,
            format_func=backend_label_from_key,
        )
        if selected_backend != st.session_state.embedding_backend:
            st.session_state.embedding_backend = selected_backend
            if selected_backend == "openai":
                st.session_state.embed_model = OPENAI_EMBED_MODELS[0]
            else:
                st.session_state.embed_model = next(iter(TEI_MODELS))
            st.experimental_rerun()

        if st.session_state.embedding_backend == "openai":
            embedding_options = OPENAI_EMBED_MODELS
            if st.session_state.embed_model not in embedding_options:
                st.session_state.embed_model = embedding_options[0]
            st.session_state.embed_model = st.selectbox(
                tr("embedding_model_label"),
                options=embedding_options,
                index=embedding_options.index(st.session_state.embed_model),
                help=tr("embedding_model_help_openai"),
            )
        else:
            embedding_options = list(TEI_MODELS.keys())
            if st.session_state.embed_model not in embedding_options:
                st.session_state.embed_model = embedding_options[0]
            st.session_state.embed_model = st.selectbox(
                tr("embedding_model_label"),
                options=embedding_options,
                index=embedding_options.index(st.session_state.embed_model),
                format_func=tei_display_name,
                help=tr("embedding_model_help_tei"),
            )

            config = TEI_MODELS[st.session_state.embed_model]
            available = tei_model_is_downloaded(st.session_state.embed_model)
            st.markdown(f"<div class='small-label'>{tr('local_model_section_title')}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='small-label'><strong>{tei_display_name(st.session_state.embed_model)}</strong></div>",
                unsafe_allow_html=True,
            )
            status_class = "status-ok" if available else "status-missing"
            status_text = tr("status_available") if available else tr("status_not_available")
            st.markdown(
                f"<span class='status-tag {status_class}'>{status_text}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='small-label'>{tr('download_location', path=str(config['local_dir']))}</div>",
                unsafe_allow_html=True,
            )
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            if not available:
                if st.button(tr("download_button"), key=f"download-{st.session_state.embed_model}"):
                    success, message = download_tei_model_with_progress(
                        st.session_state.embed_model,
                        progress_placeholder,
                        status_placeholder,
                    )
                    display_name = tei_display_name(st.session_state.embed_model)
                    if success:
                        st.session_state.download_feedback = ("success", tr("download_success", model=display_name))
                    else:
                        fallback_message = message or tr(
                            "download_failed",
                            model=display_name,
                            code="?",
                            detail=tr("download_no_log"),
                        )
                        st.session_state.download_feedback = ("error", fallback_message)
                    st.experimental_rerun()
            else:
                status_placeholder.caption(tr("available_label"))

        feedback = st.session_state.get("download_feedback")
        if feedback:
            status_msg, message = feedback
            if status_msg == "success":
                st.success(message)
            else:
                st.error(message)
            st.session_state.download_feedback = None

        st.session_state.chat_model = st.selectbox(
            tr("chat_model_label"),
            options=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
            index=0 if st.session_state.chat_model not in ["gpt-4o", "gpt-4.1-mini", "gpt-4.1"] else
                   ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"].index(st.session_state.chat_model),
        )

        st.session_state.retriever_k = st.slider(
            tr("topk_slider_label"),
            min_value=2,
            max_value=10,
            value=st.session_state.retriever_k,
            step=1,
        )

        st.markdown("---")
        if st.button(tr("rebuild_button")):
            if st.session_state.embedding_backend == "openai" and not st.session_state.openai_key:
                st.error(tr("rebuild_need_key"))
            elif st.session_state.embedding_backend == "tei" and not tei_model_is_downloaded(st.session_state.embed_model):
                st.error(tr("rebuild_need_model"))
            else:
                with st.spinner(tr("rebuild_progress")):
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
                            tr(
                                "rebuild_success",
                                pdfs=stats["pdf_count"],
                                pages=stats["pages"],
                                chunks=stats["chunks"],
                                index=stats["index_dir"],
                                backend=stats["embedding_backend"],
                                model=stats["embedding_model"],
                            )
                        )
                    except Exception as e:
                        st.error(tr("rebuild_failed", error=e))

        if st.button(tr("clear_chat")):
            st.session_state.history = []

        st.markdown("---")
        index_state = tr("pdfs_state_ready") if index_exists() else tr("pdfs_state_missing")
        st.caption(
            tr("pdfs_info", count=len(list_pdfs()), state=index_state)
        )

        emb_used = load_embed_meta()
        if emb_used:
            used_label = backend_label_from_key(emb_used.get("embedding_backend", "openai"))
            st.caption(tr("index_built_with", backend_label=used_label, model=emb_used["embedding_model"]))
            if (
                emb_used.get("embedding_backend") != st.session_state.embedding_backend
                or emb_used.get("embedding_model") != st.session_state.embed_model
            ):
                st.warning(tr("index_mismatch"))
    # Main Chat Area
    st.title(tr("app_title"))

    if not st.session_state.openai_key:
        st.info(tr("no_key_info"))
    if not index_exists():
        st.warning(tr("no_index_warning"))

    # Render chat history
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                with st.expander(tr("sources_label")):
                    for i, s in enumerate(turn["sources"], start=1):
                        st.markdown(f"**{i}.** {s['source']} (p.{s.get('page', '?')})")
                        if s.get("snippet"):
                            st.caption(s["snippet"])

    user_q = st.chat_input(tr("chat_input_placeholder"))
    if user_q:
        st.session_state.history.append({"role": "user", "content": user_q})

        if not st.session_state.openai_key:
            ans = tr("missing_key_response")
            st.session_state.history.append({"role": "assistant", "content": ans})
            st.experimental_rerun()

        if not index_exists():
            ans = tr("missing_index_response")
            st.session_state.history.append({"role": "assistant", "content": ans})
            st.experimental_rerun()

        if st.session_state.embedding_backend == "tei" and not tei_model_is_downloaded(st.session_state.embed_model):
            ans = tr("missing_model_response")
            st.session_state.history.append({"role": "assistant", "content": ans})
            st.experimental_rerun()

        try:
            with st.spinner(tr("retrieving_spinner")):
                vectorstore = load_vectorstore(
                    st.session_state.embedding_backend,
                    st.session_state.embed_model,
                )
                retriever = vectorstore.as_retriever(search_kwargs={"k": st.session_state.retriever_k})
                docs = retriever.get_relevant_documents(user_q)

                context = format_docs_for_prompt(docs)
                answer = call_llm(st.session_state.chat_model, user_q, context)

                sources = []
                for d in docs:
                    source = d.metadata.get("source", "unknown")
                    page = d.metadata.get("page")
                    snippet = (d.page_content[:400] + "?") if d.page_content and len(d.page_content) > 420 else d.page_content
                    sources.append({"source": source, "page": page, "snippet": snippet})

                st.session_state.history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
        except Exception as e:
            st.session_state.history.append({
                "role": "assistant",
                "content": tr("error_prefix", error=e)
            })

        st.rerun()

if __name__ == "__main__":
    main()
