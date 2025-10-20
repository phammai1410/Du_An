"""Streamlit UI for the PDF RAG assistant."""

import json
import os
import subprocess
import sys
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

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
BACKEND_ROOT = PROJECT_ROOT.parent / "backend"
DATA_DIR = BACKEND_ROOT / "data" / "raw"
UPLOADS_ROOT = DATA_DIR / "uploads"
INDEX_ROOT = BACKEND_ROOT / "data" / "index"
TOOLS_DIR = BACKEND_ROOT / "tools"
LOCAL_TEI_ROOT = BACKEND_ROOT / "local-llm" / "Embedding"
MODELS_CONFIG_PATH = LOCAL_TEI_ROOT / "models.json"
TEI_CONTAINER_PREFIX = "tei-"

DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL")
DEFAULT_CHAT_MODEL = "gpt-4o-mini"

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200

CHUNK_MODES: Dict[str, str] = {
    "structured": "Structured-chunks",
    "direct": "Direct-chunks",
}
DEFAULT_CHUNK_MODE = "structured"

UPLOAD_ALLOWED_EXTS = {"pdf", "docx", "xlsx"}

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

TEI_RUNTIME_MODES: Dict[str, Dict[str, Any]] = {
    "cpu": {
        "label": "CPU",
        "image": "ghcr.io/huggingface/text-embeddings-inference:cpu-1.8",
        "requires_gpu": False,
        "description": "Default mode for CPU-only machines.",
    },
    "turing": {
        "label": "Turing (T4 / RTX 2000 series)",
        "image": "ghcr.io/huggingface/text-embeddings-inference:turing-1.8",
        "requires_gpu": True,
        "description": "Optimized build for NVIDIA Turing GPUs such as T4 or RTX 2000.",
    },
    "ampere_80": {
        "label": "Ampere 80 (A100 / A30)",
        "image": "ghcr.io/huggingface/text-embeddings-inference:1.8",
        "requires_gpu": True,
        "description": "Use on A100 or A30 class Ampere GPUs.",
    },
    "ampere_86": {
        "label": "Ampere 86 (A10 / A40)",
        "image": "ghcr.io/huggingface/text-embeddings-inference:86-1.8",
        "requires_gpu": True,
        "description": "Tune for Ampere 86 GPUs including A10, A40, and RTX A series.",
    },
    "ada_lovelace": {
        "label": "Ada Lovelace (RTX 4000 series)",
        "image": "ghcr.io/huggingface/text-embeddings-inference:89-1.8",
        "requires_gpu": True,
        "description": "Optimized for the RTX 4000 Ada Lovelace family.",
    },
    "hopper": {
        "label": "Hopper (H100, experimental)",
        "image": "ghcr.io/huggingface/text-embeddings-inference:hopper-1.8",
        "requires_gpu": True,
        "description": "Experimental build for NVIDIA Hopper GPUs (H100).",
    },
}
DEFAULT_TEI_RUNTIME_MODE = "cpu"

DOCKER_INSTALL_GUIDE_MD = """
**Docker is required to start Text Embeddings Inference.**

- **Windows**
  - [Docker Desktop installer](https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe)
  - [Docker Desktop (Microsoft Store)](https://apps.microsoft.com/detail/xp8cbj40xlbwkx)
- **macOS**
  - [Apple silicon build](https://desktop.docker.com/mac/main/arm64/Docker.dmg)
  - [Intel build](https://desktop.docker.com/mac/main/amd64/Docker.dmg)
- **Linux**
  - [Docker Engine](https://docs.docker.com/engine/install/)
  - [Docker Desktop](https://docs.docker.com/desktop/setup/install/linux/)
"""

LINUX_GPU_TOOLKIT_MD = """
GPU modes on Linux also need the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Verify the setup with:

```
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```
"""


def load_tei_models_config() -> Dict[str, Dict[str, Any]]:
    try:
        with MODELS_CONFIG_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return {}


def get_tei_model_port(model_key: str) -> int:
    config = load_tei_models_config()
    model_cfg = config.get(model_key, {})
    try:
        return int(model_cfg.get("port", 8800))
    except (TypeError, ValueError):
        return 8800


def sanitize_tei_container_name(model_key: str, runtime_key: str) -> str:
    base = f"{TEI_CONTAINER_PREFIX}{model_key}-{runtime_key}"
    slug = []
    for char in base.lower():
        if char.isalnum() or char == "-":
            slug.append(char)
        else:
            slug.append("-")
    sanitized = "".join(slug).strip("-")
    while "--" in sanitized:
        sanitized = sanitized.replace("--", "-")
    return sanitized[:63] or f"{TEI_CONTAINER_PREFIX}runtime"


def get_running_tei_containers() -> Tuple[List[str], Optional[str]]:
    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--format",
                "{{.Names}}",
                "--filter",
                f"name={TEI_CONTAINER_PREFIX}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return [], "Docker CLI not found."
    except Exception as exc:
        return [], str(exc)

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "Unknown Docker error."
        return [], detail

    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return names, None


def run_launch_tei(args: List[str]) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(TOOLS_DIR / "launch_tei.py"), *args]
    return subprocess.run(
        command,
        cwd=str(BACKEND_ROOT),
        capture_output=True,
        text=True,
    )


def summarize_process(result: subprocess.CompletedProcess[str]) -> str:
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if stdout and stderr:
        return f"{stdout}\n{stderr}"
    return stdout or stderr or "No output."


def set_tei_base_url(port: int) -> str:
    base_url = f"http://localhost:{port}"
    os.environ["TEI_BASE_URL"] = base_url
    st.session_state.tei_base_url = base_url
    return base_url


def start_tei_runtime(model_key: str, runtime_key: str, port: int) -> Tuple[bool, str]:
    args = [
        "--model",
        model_key,
        "--runtime",
        runtime_key,
        "--port",
        str(port),
        "--detach",
    ]
    result = run_launch_tei(args)
    success = result.returncode == 0
    runtime_label = TEI_RUNTIME_MODES.get(runtime_key, {}).get("label", runtime_key)
    if success:
        set_tei_base_url(port)
        message = f"Started TEI ({runtime_label}) on http://localhost:{port}."
    else:
        message = summarize_process(result)
    return success, message


def stop_tei_runtime(model_key: str, runtime_key: str) -> Tuple[bool, str]:
    args = [
        "--model",
        model_key,
        "--runtime",
        runtime_key,
        "--stop",
    ]
    result = run_launch_tei(args)
    success = result.returncode == 0
    container_name = sanitize_tei_container_name(model_key, runtime_key)
    runtime_label = TEI_RUNTIME_MODES.get(runtime_key, {}).get("label", runtime_key)
    if success:
        message = (result.stdout or "").strip() or f"Stopped TEI container `{container_name}` ({runtime_label})."
    else:
        message = summarize_process(result)
    return success, message


def stop_all_tei_runtimes() -> Tuple[bool, str]:
    result = run_launch_tei(["--stop-all"])
    success = result.returncode == 0
    if success:
        message = (result.stdout or "").strip() or "No TEI containers were running."
    else:
        message = summarize_process(result)
    return success, message


def tei_backend_is_active(model_key: str, runtime_key: str) -> bool:
    containers, _ = get_running_tei_containers()
    return sanitize_tei_container_name(model_key, runtime_key) in containers


# -----------------------------------------------------------------------------#
# Utility helpers
# -----------------------------------------------------------------------------#


def safe_model_dir(model_name: str) -> str:
    slug = model_name.strip().lower().replace("/", "-")
    cleaned = []
    for ch in slug:
        if ch.isalnum() or ch in {"-", "_"}:
            cleaned.append(ch)
        else:
            cleaned.append("-")
    safe = "".join(cleaned).strip("-")
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe or "default"


def resolve_index_dir(model_name: str) -> Path:
    return INDEX_ROOT / safe_model_dir(model_name)


def get_embed_meta_path(index_dir: Path) -> Path:
    return index_dir / "embeddings.json"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
    for ext in UPLOAD_ALLOWED_EXTS:
        (UPLOADS_ROOT / ext).mkdir(parents=True, exist_ok=True)
    for cfg in TEI_MODELS.values():
        local_dir: Path = cfg["local_dir"]
        local_dir.mkdir(parents=True, exist_ok=True)


def list_pdfs() -> List[Path]:
    return sorted(DATA_DIR.rglob("*.pdf"))


def index_exists(index_dir: Path) -> bool:
    return (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()


def save_embed_meta(index_dir: Path, backend: str, model_name: str, chunk_mode: str) -> None:
    try:
        meta_path = get_embed_meta_path(index_dir)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(
                {
                    "embedding_backend": backend,
                    "embedding_model": model_name,
                    "chunk_mode": chunk_mode,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass  # Silently ignore metadata persistence issues


def load_embed_meta(index_dir: Path) -> Optional[Dict[str, Optional[str]]]:
    try:
        meta_path = get_embed_meta_path(index_dir)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                backend = meta.get("embedding_backend", "openai")
                model = meta.get("embedding_model")
                if model:
                    chunk = meta.get("chunk_mode")
                    return {
                        "embedding_backend": backend,
                        "embedding_model": model,
                        "chunk_mode": chunk,
                    }
            elif isinstance(meta, str):
                return {
                    "embedding_backend": "openai",
                    "embedding_model": meta,
                    "chunk_mode": None,
                }
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


def check_docker_cli() -> Tuple[bool, str]:
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False, "Docker command not found in PATH."
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"Failed to execute docker: {exc}"

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error."
        return False, detail

    return True, result.stdout.strip()


def docker_supports_nvidia() -> Tuple[bool, Optional[str]]:
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{json .Runtimes}}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False, "Docker command not found in PATH."
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"Failed to query docker: {exc}"

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error."
        return False, detail

    if "nvidia" in result.stdout.lower():
        return True, None

    return False, "NVIDIA runtime not detected."


def make_embeddings_client(backend: str, model_name: str):
    if backend == "openai":
        return OpenAIEmbeddings(model=model_name)
    if backend == "tei":
        base_url = st.session_state.get("tei_base_url") or os.getenv("TEI_BASE_URL", "http://localhost:8080")
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
    chunk_mode: str = DEFAULT_CHUNK_MODE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Dict[str, Any]:
    pdfs = list(sorted(data_dir.rglob("*.pdf")))
    if not pdfs:
        raise RuntimeError(f"No PDF files found in: {data_dir}")

    docs = []
    for pdf in pdfs:
        loader = PyPDFLoader(str(pdf))
        loaded = loader.load()
        for doc in loaded:
            doc.metadata.setdefault("source", str(pdf))
        docs.extend(loaded)

    if chunk_mode == "direct":
        splits = docs
    elif chunk_mode == "structured":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
        splits = splitter.split_documents(docs)
    else:
        raise ValueError(f"Unsupported chunk mode: {chunk_mode}")

    embeddings = make_embeddings_client(embedding_backend, embedding_model)
    vectorstore = FAISS.from_documents(splits, embedding=embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    save_embed_meta(index_dir, embedding_backend, embedding_model, chunk_mode)

    return {
        "pdf_count": len(pdfs),
        "pages": len(docs),
        "chunks": len(splits),
        "index_dir": str(index_dir),
        "embedding_model": embedding_model,
        "embedding_backend": embedding_backend,
        "chunk_mode": chunk_mode,
    }


def load_vectorstore(index_dir: Path, embedding_backend: str, embedding_model: str) -> FAISS:
    embeddings = make_embeddings_client(embedding_backend, embedding_model)
    return FAISS.load_local(
        str(index_dir),
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

        div.stButton > button[aria-label="Start TEI"] {
            background: linear-gradient(135deg, #16a34a, #15803d);
            border: 1px solid #166534;
        }

        div.stButton > button[aria-label="Start TEI"]:hover {
            background: linear-gradient(135deg, #15803d, #166534);
            box-shadow: 0 14px 28px rgba(21, 128, 61, 0.28);
        }

        div.stButton > button[aria-label="Stop TEI"] {
            background: linear-gradient(135deg, #dc2626, #b91c1c);
            border: 1px solid #991b1b;
        }

        div.stButton > button[aria-label="Stop TEI"]:hover {
            background: linear-gradient(135deg, #b91c1c, #991b1b);
            box-shadow: 0 14px 28px rgba(220, 38, 38, 0.28);
        }

        div.stButton > button:hover:not([aria-label="Start TEI"]):not([aria-label="Stop TEI"]) {
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

        div[data-testid="stSidebar"] div[data-testid="stRadio"] {
            padding: 0.6rem 0.75rem;
            background: rgba(255, 255, 255, 0.85);
            border-radius: 18px;
            border: 1px solid var(--border-soft);
            box-shadow: 0 10px 22px rgba(90, 141, 238, 0.08);
            margin-bottom: 1rem;
        }

        div[data-testid="stSidebar"] div[data-testid="stRadio"] label {
            font-size: 0.9rem;
            color: var(--text-primary);
            font-weight: 600;
            padding: 0.25rem 0.45rem;
            border-radius: 12px;
        }

        div[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
            background: rgba(142, 185, 255, 0.16);
        }

        .runtime-chip {
            display: inline-block;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.01em;
            margin-bottom: 0.35rem;
        }

        .runtime-chip--ok {
            background: rgba(118, 204, 145, 0.22);
            color: #2f7a47;
            border: 1px solid rgba(118, 204, 145, 0.42);
        }

        .runtime-chip--pending {
            background: rgba(255, 200, 142, 0.22);
            color: #a35a00;
            border: 1px solid rgba(255, 200, 142, 0.4);
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
    if "chunk_mode" not in st.session_state:
        st.session_state.chunk_mode = DEFAULT_CHUNK_MODE
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
    if "upload_feedback" not in st.session_state:
        st.session_state.upload_feedback = None
    if "tei_runtime_mode" not in st.session_state:
        st.session_state.tei_runtime_mode = DEFAULT_TEI_RUNTIME_MODE
    if "tei_control_feedback" not in st.session_state:
        st.session_state.tei_control_feedback = None
    if "tei_base_url" not in st.session_state:
        st.session_state.tei_base_url = os.getenv("TEI_BASE_URL")

    if st.session_state.embedding_backend == "openai" and st.session_state.embed_model not in OPENAI_EMBED_MODELS:
        st.session_state.embed_model = OPENAI_EMBED_MODELS[0]
    if st.session_state.embedding_backend == "tei" and st.session_state.embed_model not in TEI_MODELS:
        st.session_state.embed_model = list(TEI_MODELS.keys())[0]


def render_settings_body() -> None:
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

        runtime_options = list(TEI_RUNTIME_MODES.keys())
        if st.session_state.tei_runtime_mode not in runtime_options:
            st.session_state.tei_runtime_mode = DEFAULT_TEI_RUNTIME_MODE
        st.selectbox(
            "TEI runtime mode",
            options=runtime_options,
            format_func=lambda key: TEI_RUNTIME_MODES[key]["label"],
            key="tei_runtime_mode",
        )
        runtime_mode = st.session_state.tei_runtime_mode
        runtime_info = TEI_RUNTIME_MODES[runtime_mode]

        docker_ok, docker_detail = check_docker_cli()
        running_containers: List[str] = []
        running_error: Optional[str] = None
        if docker_ok:
            st.success(f"Docker detected ({docker_detail}).")
            running_containers, running_error = get_running_tei_containers()
        else:
            st.error("Docker CLI not available. Install Docker to run TEI containers.")
            st.markdown(DOCKER_INSTALL_GUIDE_MD)

        st.caption(f"Docker image: `{runtime_info['image']}`")
        if runtime_info.get("description"):
            st.caption(runtime_info["description"])

        if runtime_info.get("requires_gpu"):
            st.info("This runtime requires an NVIDIA GPU.")
            if platform.system().lower() == "linux" and docker_ok:
                nvidia_ok, nvidia_detail = docker_supports_nvidia()
                if nvidia_ok:
                    st.success("NVIDIA Container Toolkit detected via `docker info`.")
                else:
                    st.error("NVIDIA Container Toolkit not detected. Install it before launching GPU runtimes on Linux.")
                    st.markdown(LINUX_GPU_TOOLKIT_MD)
                    if nvidia_detail and "not detected" not in nvidia_detail.lower():
                        st.caption(nvidia_detail)
        elif not docker_ok:
            st.info("Docker installation is required before running the TEI backend.")

        model_key = st.session_state.embed_model
        config = TEI_MODELS[model_key]
        downloaded = tei_model_is_downloaded(model_key)
        port_value = get_tei_model_port(model_key)
        container_name = sanitize_tei_container_name(model_key, runtime_mode)
        set_tei_base_url(port_value)
        is_running = docker_ok and container_name in running_containers

        st.caption(f"TEI endpoint URL: `http://localhost:{port_value}/embed`")
        if running_error:
            st.warning(f"Could not inspect Docker containers: {running_error}")
        elif is_running:
            st.success(f"Container `{container_name}` is running.")
        elif docker_ok:
            st.info("No TEI container is running right now.")

        button_label = "Stop TEI" if is_running else "Start TEI"
        button_disabled = not docker_ok
        if st.button(
            button_label,
            key=f"tei-toggle-{model_key}-{runtime_mode}",
            use_container_width=True,
            type="primary",
            disabled=button_disabled,
        ):
            if is_running:
                success, message = stop_tei_runtime(model_key, runtime_mode)
            else:
                success, message = start_tei_runtime(model_key, runtime_mode, port_value)
            st.session_state.tei_control_feedback = ("success" if success else "error", message)
            st.rerun()

        control_feedback = st.session_state.tei_control_feedback
        if control_feedback:
            status, message = control_feedback
            if status == "success":
                st.success(message)
            else:
                st.error(message)
            st.session_state.tei_control_feedback = None

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

    st.selectbox(
        "Chunk mode",
        options=list(CHUNK_MODES.keys()),
        format_func=lambda key: CHUNK_MODES[key],
        key="chunk_mode",
    )

    st.slider(
        "Top-k passages",
        min_value=2,
        max_value=10,
        step=1,
        key="retriever_k",
    )

def render_sidebar_quick_actions() -> None:
    st.subheader("Data")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=sorted(UPLOAD_ALLOWED_EXTS),
        accept_multiple_files=True,
        help="Files are saved to `backend/data/raw/uploads/<ext>/`.",
    )
    if uploaded_files:
        saved_paths = []
        errors = []
        for file in uploaded_files:
            suffix = Path(file.name).suffix.lower()
            ext = suffix.lstrip(".")
            if ext not in UPLOAD_ALLOWED_EXTS:
                errors.append(f"Unsupported file type: {file.name}")
                continue
            target_dir = (UPLOADS_ROOT / ext)
            target_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(file.name).stem.strip()
            safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in stem).strip("-") or "document"
            unique_name = f"{safe_stem}-{uuid4().hex[:8]}{suffix}"
            target_path = target_dir / unique_name
            try:
                with open(target_path, "wb") as out_file:
                    out_file.write(file.getbuffer())
                saved_paths.append(target_path)
            except Exception as exc:
                errors.append(f"Failed to save {file.name}: {exc}")
        messages = []
        if saved_paths:
            rel_paths = [str(path.relative_to(PROJECT_ROOT.parent)) for path in saved_paths]
            messages.append("Saved: " + ", ".join(rel_paths))
        if errors:
            messages.extend(errors)
        if saved_paths and errors:
            status = "warning"
        elif saved_paths:
            status = "success"
        elif errors:
            status = "error"
        else:
            status = "info"
        st.session_state.upload_feedback = (status, " | ".join(messages) if messages else "No files processed.")
        st.rerun()

    upload_feedback = st.session_state.upload_feedback
    if upload_feedback:
        status, message = upload_feedback
        if status == "success":
            st.success(message)
        elif status == "warning":
            st.warning(message)
        elif status == "info":
            st.info(message)
        else:
            st.error(message)
        st.session_state.upload_feedback = None

    st.divider()
    current_index_dir = resolve_index_dir(st.session_state.embed_model)
    if st.button("Rebuild index from PDFs", use_container_width=True):
        if st.session_state.embedding_backend == "openai" and not st.session_state.openai_key:
            st.error("Please provide an OpenAI API key first.")
        elif st.session_state.embedding_backend == "tei" and not tei_model_is_downloaded(st.session_state.embed_model):
            st.error("Selected TEI model is not downloaded yet.")
        else:
            with st.spinner("Building FAISS index from PDFs..."):
                try:
                    stats = build_index_from_pdfs(
                        DATA_DIR,
                        current_index_dir,
                        embedding_model=st.session_state.embed_model,
                        embedding_backend=st.session_state.embedding_backend,
                        chunk_mode=st.session_state.chunk_mode,
                        chunk_size=DEFAULT_CHUNK_SIZE,
                        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                    )
                    st.success(
                        "Done! "
                        f"{stats['pdf_count']} PDFs, {stats['pages']} pages -> {stats['chunks']} chunks. "
                        "Index: "
                        f"{stats['index_dir']} (backend: {stats['embedding_backend']} | model: {stats['embedding_model']} | chunk: {CHUNK_MODES.get(stats['chunk_mode'], stats['chunk_mode'])})"
                    )
                except Exception as exc:
                    st.error(f"Failed to build index: {exc}")

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.divider()
    index_state = "yes" if index_exists(current_index_dir) else "no"
    st.caption(
        f"PDFs in `backend/data/raw`: {len(list_pdfs())} | "
        f"Index `{current_index_dir.relative_to(PROJECT_ROOT.parent)}` present: {index_state}"
    )

    emb_used = load_embed_meta(current_index_dir)
    if emb_used:
        backend_label = EMBED_BACKENDS.get(emb_used.get("embedding_backend", "openai"), "Unknown")
        chunk_value = emb_used.get("chunk_mode")
        chunk_label = CHUNK_MODES.get(chunk_value, chunk_value if chunk_value else "unknown")
        st.caption(f"Index built with: {backend_label} / {emb_used['embedding_model']} / {chunk_label}")
        if (
            emb_used.get("embedding_backend") != st.session_state.embedding_backend
            or emb_used.get("embedding_model") != st.session_state.embed_model
            or emb_used.get("chunk_mode") != st.session_state.chunk_mode
        ):
            st.warning("The current embedding selection differs from the index. Rebuild to avoid inconsistencies.")


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

    current_index_dir = resolve_index_dir(st.session_state.embed_model)

    if st.session_state.embedding_backend == "openai" and not st.session_state.openai_key:
        st.info("No OpenAI API key detected. Enter it in the sidebar or the `.env` file before generating embeddings.")
    elif st.session_state.embedding_backend == "tei":
        runtime_mode = st.session_state.tei_runtime_mode
        model_key = st.session_state.embed_model
        if not tei_backend_is_active(model_key, runtime_mode):
            st.warning("The Docker-based TEI service is not running. Start it from the sidebar before continuing.")
    if not index_exists(current_index_dir):
        st.warning("No FAISS index found. Rebuild the index from the sidebar or run `python ingest_pdfs.py`.")

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

        runtime_mode = st.session_state.tei_runtime_mode
        tei_model_key = st.session_state.embed_model

        if st.session_state.embedding_backend == "openai" and not st.session_state.openai_key:
            st.session_state.history.append({"role": "assistant", "content": "Please add your OpenAI API key in the sidebar first."})
            st.rerun()

        if st.session_state.embedding_backend == "tei" and not tei_backend_is_active(tei_model_key, runtime_mode):
            st.session_state.history.append({
                "role": "assistant",
                "content": "The Docker-based TEI service is not running. Start it from the sidebar before continuing.",
            })
            st.rerun()

        if not index_exists(current_index_dir):
            st.session_state.history.append({"role": "assistant", "content": "FAISS index is missing. Rebuild it first."})
            st.rerun()

        if st.session_state.embedding_backend == "tei" and not tei_model_is_downloaded(tei_model_key):
            st.session_state.history.append({
                "role": "assistant",
                "content": "Selected TEI model is not downloaded. Download it from the sidebar.",
            })
            st.rerun()

        try:
            with st.spinner("Retrieving and reasoning..."):
                vectorstore = load_vectorstore(
                    current_index_dir,
                    st.session_state.embedding_backend,
                    st.session_state.embed_model,
                )
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


def render_sidebar() -> None:
    render_settings_body()
    st.divider()
    render_sidebar_quick_actions()


if __name__ == "__main__":
    main()
