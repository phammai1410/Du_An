"""Streamlit UI for the PDF RAG assistant."""

import importlib.util
import json
import os
import subprocess
import sys
import platform
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import numpy as np
import unicodedata

try:
    import faiss  # type: ignore
except Exception:  # noqa: BLE001
    faiss = None  # type: ignore

SOFT_PID_REGISTRY: dict[str, int] = {}


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

DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_LOCALAI_BASE_URL = os.getenv("LOCALAI_BASE_URL", "http://localhost:8080/v1")
DEFAULT_LOCALAI_API_KEY = os.getenv("LOCALAI_API_KEY", "localai-temp-key")

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_LOCALAI_BASE_URL = os.getenv("LOCALAI_BASE_URL", "http://localhost:8080/v1")
DEFAULT_LOCALAI_API_KEY = os.getenv("LOCALAI_API_KEY", "localai-temp-key")
LOCALAI_IMAGE = os.getenv("LOCALAI_IMAGE", "localai/localai:latest")
LOCALAI_CONTAINER_NAME = os.getenv("LOCALAI_CONTAINER_NAME", "localai-runtime")
LOCALAI_PORT = int(os.getenv("LOCALAI_PORT", "8080"))
LOCALAI_RUNTIME_MODEL = os.getenv("LOCALAI_RUNTIME_MODEL", "llama-3.2-1b-instruct:q4_k_m")
LOCALAI_COMPOSE_PROJECT = os.getenv("LOCALAI_COMPOSE_PROJECT", "khoa_luan")
LOCALAI_COMPOSE_SERVICE = os.getenv("LOCALAI_COMPOSE_SERVICE", "localai")
SMALL_TALK_KEYWORDS = {
    "hi",
    "hello",
    "hey",
    "xin chao",
    "xin chào",
    "chao",
    "chào",
    "alo",
    "yo",
}
CAPABILITY_PHRASES = {
    "ban co the lam gi",
    "bạn có thể làm gì",
    "what can you do",
    "what can u do",
    "can you help",
    "ban lam duoc gi",
    "bạn làm được gì",
}

CHUNK_MODES: Dict[str, str] = {
    "structured": "Structured-chunks",
    "direct": "Direct-chunks",
}
DEFAULT_CHUNK_MODE = "structured"

UPLOAD_ALLOWED_EXTS = {"pdf", "docx", "xlsx"}

OPENAI_EMBED_MODELS = ["text-embedding-3-small", "text-embedding-3-large"]
EMBED_BACKENDS: Dict[str, str] = {
    "openai": "Open AI Chat GPT Embedding",
    "tei": "Local Text-Embeddings-Inference",
}
GLOBAL_DOCKER_ERROR_SNIPPETS = ("docker desktop is manually paused",)

TEI_MODELS: Dict[str, Dict[str, Any]] = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "display": "MiniLM-L6 v2 0.2B (Q4)",
        "config_key": "sentence-transformers-all-MiniLM-L6-v2",
        "local_dir": LOCAL_TEI_ROOT / "sentence-transformers-all-MiniLM-L6-v2",
        "download_script": TOOLS_DIR / "download_all_minilm_l6_v2_tei.py",
        "required_file": "model.safetensors",
    },
    "intfloat/e5-small-v2": {
        "display": "IntFloat E5 Small V2",
        "config_key": "intfloat-e5-small-v2",
        "local_dir": LOCAL_TEI_ROOT / "intfloat-e5-small-v2",
        "download_script": TOOLS_DIR / "download_e5_small_v2_tei.py",
        "required_file": "model.safetensors",
    },
}

RUN_MODE_OPTIONS = {
    "Open AI ChatGPT": "openai",
    "Local Embedding/LLM": "local",
}

OPENAI_CHAT_MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
]

LOCAL_CHAT_MODELS = {
    "LLAMA 3.1 1B": "llama-3.2-1b-instruct:q4_k_m",
}


def _slugify_identifier(value: str) -> str:
    chars: List[str] = []
    for char in value:
        lower = char.lower()
        if lower.isalnum():
            chars.append(lower)
        else:
            chars.append("-")
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "default"


def resolve_tei_config_key(model_key: str) -> str:
    """Map UI model keys to the config key expected by launch_tei.py/models.json."""
    config = TEI_MODELS.get(model_key, {})
    config_key = config.get("config_key")
    if isinstance(config_key, str) and config_key:
        return config_key
    # Fallback: replace characters unsupported in JSON config keys.
    return model_key.replace("/", "-")


def resolve_tei_ui_key(config_key: str) -> Optional[str]:
    if not config_key:
        return None
    normalized = _slugify_identifier(config_key)
    for ui_key, meta in TEI_MODELS.items():
        candidates = {
            meta.get("config_key"),
            _slugify_identifier(meta.get("config_key", "") or ""),
            _slugify_identifier(ui_key),
        }
        if normalized in {c for c in candidates if c}:
            return ui_key
    return None


def format_embedding_display(backend_key: Optional[str], model_key: Optional[str]) -> str:
    if not model_key:
        return "Unknown model"
    if backend_key == "tei":
        meta = TEI_MODELS.get(model_key)
        if meta and meta.get("display"):
            return meta["display"]
        resolved = resolve_tei_ui_key(model_key)
        if resolved and resolved in TEI_MODELS:
            return TEI_MODELS[resolved].get("display", resolved)
        return model_key
    return str(model_key)


DEFAULT_TEI_RUNTIME_MODE = "cpu"
DEFAULT_TEI_RUNTIME_LABEL = "CPU"
DEFAULT_TEI_RUNTIME_IMAGE = os.getenv(
    "TEI_RUNTIME_IMAGE", "ghcr.io/huggingface/text-embeddings-inference:cpu-1.8"
)

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
    config_key = resolve_tei_config_key(model_key)
    model_cfg = config.get(config_key, {})
    try:
        return int(model_cfg.get("port", 8800))
    except (TypeError, ValueError):
        return 8800


def _tei_container_prefix(model_key: str, runtime_key: str) -> str:
    model_slug = _slugify_identifier(resolve_tei_config_key(model_key))
    runtime_slug = _slugify_identifier(runtime_key)
    return f"{TEI_CONTAINER_PREFIX}{model_slug}-{runtime_slug}"


def sanitize_tei_container_name(model_key: str, runtime_key: str) -> str:
    base = _tei_container_prefix(model_key, runtime_key)
    sanitized = base.strip("-")
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
            encoding="utf-8",
            errors="replace",
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


def get_tei_runtime_status(model_key: str, runtime_key: str) -> Dict[str, Any]:
    expected_prefix = _tei_container_prefix(model_key, runtime_key)
    names, error = get_running_tei_containers()
    match = None
    others: List[str] = []
    for name in names:
        if name == expected_prefix or name.startswith(f"{expected_prefix}-"):
            match = name
        else:
            others.append(name)
    return {
        "running": match is not None,
        "match": match,
        "others": others,
        "error": error,
    }


def format_tei_container_label(container_name: str) -> str:
    if not container_name.startswith(TEI_CONTAINER_PREFIX):
        return container_name
    suffix = container_name[len(TEI_CONTAINER_PREFIX) :]
    if not suffix:
        return container_name
    parts = suffix.split("-")
    model_slug = suffix
    runtime_slug = ""
    cpu_slug = _slugify_identifier(DEFAULT_TEI_RUNTIME_MODE)
    if len(parts) >= 2 and parts[-1] == cpu_slug:
        runtime_slug = parts[-1]
        model_slug = "-".join(parts[:-1])
    model_label = None
    for key, meta in TEI_MODELS.items():
        slug = _slugify_identifier(resolve_tei_config_key(key))
        if slug == model_slug:
            model_label = meta.get("display") or key
            break
    if runtime_slug in {cpu_slug, DEFAULT_TEI_RUNTIME_MODE, ""}:
        runtime_label = DEFAULT_TEI_RUNTIME_LABEL
    else:
        runtime_label = runtime_slug.upper() if runtime_slug else DEFAULT_TEI_RUNTIME_LABEL
    if model_label:
        return f"{model_label} - {runtime_label}"
    return f"{model_slug} - {runtime_label}"


def localai_is_running() -> bool:
    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--format",
                "{{.Names}}",
                "--filter",
                f"name={LOCALAI_CONTAINER_NAME}",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError:
        return False
    except Exception:
        return False
    if result.returncode != 0:
        return False
    return any(line.strip() for line in result.stdout.splitlines())


def start_localai_service() -> tuple[bool, str]:
    if localai_is_running():
        return True, "LocalAI service already running."
    try:
        result = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                LOCALAI_CONTAINER_NAME,
                "--label",
                f"com.docker.compose.project={LOCALAI_COMPOSE_PROJECT}",
                "--label",
                f"com.docker.compose.service={LOCALAI_COMPOSE_SERVICE}",
                "--label",
                "com.docker.compose.version=1.29.2",
                "-p",
                f"{LOCALAI_PORT}:8080",
                "-e",
                "LOG_LEVEL=info",
                LOCALAI_IMAGE,
                "local-ai",
                "run",
                LOCALAI_RUNTIME_MODEL,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError:
        return False, "Docker command not found."
    except Exception as exc:
        return False, str(exc)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "Failed to start localai service."
        return False, detail
    return True, (result.stdout.strip() or "LocalAI service started.")


def stop_localai_service() -> tuple[bool, str]:
    if not localai_is_running():
        return True, "LocalAI service is not running."
    try:
        result = subprocess.run(
            ["docker", "stop", LOCALAI_CONTAINER_NAME],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError:
        return False, "Docker command not found."
    except Exception as exc:
        return False, str(exc)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "Failed to stop localai service."
        return False, detail
    return True, (result.stdout.strip() or "LocalAI service stopped.")


def run_launch_tei(args: List[str]) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(TOOLS_DIR / "launch_tei.py"), *args]
    return subprocess.run(
        command,
        cwd=str(BACKEND_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def run_backend_tool(
    script_name: str,
    *args: str,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(TOOLS_DIR / script_name), *args]
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        command,
        cwd=str(BACKEND_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=merged_env,
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
    config_key = resolve_tei_config_key(model_key)
    args = [
        "--model",
        config_key,
        "--runtime",
        runtime_key,
        "--port",
        str(port),
        "--detach",
    ]
    result = run_launch_tei(args)
    success = result.returncode == 0
    runtime_label = DEFAULT_TEI_RUNTIME_LABEL
    if success:
        set_tei_base_url(port)
        message = f"Started TEI ({runtime_label}) on http://localhost:{port}."
    else:
        message = summarize_process(result)
    return success, message


def stop_tei_runtime(model_key: str, runtime_key: str) -> Tuple[bool, str]:
    config_key = resolve_tei_config_key(model_key)
    args = [
        "--model",
        config_key,
        "--runtime",
        runtime_key,
        "--stop",
    ]
    result = run_launch_tei(args)
    success = result.returncode == 0
    container_name = sanitize_tei_container_name(model_key, runtime_key)
    runtime_label = DEFAULT_TEI_RUNTIME_LABEL
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
    status = get_tei_runtime_status(model_key, runtime_key)
    return bool(status.get("running"))


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


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)
    for ext in UPLOAD_ALLOWED_EXTS:
        (UPLOADS_ROOT / ext).mkdir(parents=True, exist_ok=True)
    for cfg in TEI_MODELS.values():
        local_dir: Path = cfg["local_dir"]
        local_dir.mkdir(parents=True, exist_ok=True)


def list_docx_files() -> List[Path]:
    return sorted(DATA_DIR.rglob("*.docx"))


def index_exists(index_dir: Path) -> bool:
    return backend_index_exists(index_dir)


def load_embed_meta(index_dir: Path) -> Optional[Dict[str, Optional[str]]]:
    try:
        manifest_path = index_dir / "index_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            model = manifest.get("model")
            if isinstance(model, str):
                ui_key = resolve_tei_ui_key(model)
                if ui_key:
                    model = ui_key
            return {
                "embedding_backend": "tei",
                "embedding_model": model,
                "chunk_mode": manifest.get("chunk_mode"),
                "vector_backend": manifest.get("backend"),
            }
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

    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None):
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
            encoding="utf-8",
            errors="replace",
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
            encoding="utf-8",
            errors="replace",
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


def _resolve_model_targets(model_key: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return (required, optional) file tuples for the given model."""
    config = TEI_MODELS.get(model_key)
    if not config:
        return (), ()

    script_path = config.get("download_script")
    if script_path and script_path.exists():
        required, optional = _load_download_targets(script_path)
    else:
        required, optional = (), ()

    if not required:
        fallback = config.get("required_file")
        if isinstance(fallback, str) and fallback:
            required = (fallback,)

    return required, optional


def tei_model_is_downloaded(model_key: str) -> bool:
    config = TEI_MODELS.get(model_key)
    if not config:
        return False

    required, _ = _resolve_model_targets(model_key)
    if not required:
        return False

    base_dir: Path = config["local_dir"]
    for rel_path in required:
        if not (base_dir / rel_path).exists():
            return False
    return True


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
        encoding="utf-8",
        errors="replace",
    )


# -----------------------------------------------------------------------------#
# Index utilities (backend pipeline)
# -----------------------------------------------------------------------------#


def detect_docx_languages() -> List[str]:
    languages: List[str] = []
    if not DATA_DIR.exists():
        return languages
    for child in sorted(DATA_DIR.iterdir()):
        if child.name.lower() == "uploads":
            continue
        if not child.is_dir():
            continue
        if any(child.glob("*.docx")):
            languages.append(child.name)
    return languages


def backend_index_exists(index_dir: Path) -> bool:
    manifest = index_dir / "index_manifest.json"
    if not manifest.exists():
        return False
    faiss_path = index_dir / "index.faiss"
    brute_path = index_dir / "vectors.npy"
    return faiss_path.exists() or brute_path.exists()


PIPELINE_STEP_MESSAGES: Dict[int, str] = {
    1: "Running Step 1: DOCX -> Sections",
    2: "Running Step 2: DOCX -> JSON",
    3: "Running Step 3: JSON -> Index",
}


def run_backend_pipeline(
    model_name: str,
    langs: List[str],
    base_url: Optional[str],
    out_dir: Path,
    chunk_mode: Optional[str] = None,
    on_output: Callable[[str], None] | None = None,
) -> Tuple[bool, str]:
    script = TOOLS_DIR / "ingest_docx_pipeline.py"
    if not script.exists():
        return False, f"Pipeline script not found: {script}"

    args = [
        sys.executable,
        str(script),
        "--model",
        model_name,
        "--out-dir",
        str(out_dir),
    ]
    if base_url:
        args.extend(["--base-url", base_url])
    if langs:
        args.extend(["--langs", *langs])
    if chunk_mode:
        args.extend(["--chunk-mode", chunk_mode])

    args.extend(["--batch-size", "8"])

    existing_proc = st.session_state.get("backend_pipeline_proc")
    if existing_proc and existing_proc.poll() is None:
        try:
            existing_proc.terminate()
            existing_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            existing_proc.kill()
        except Exception:
            pass
    st.session_state["backend_pipeline_proc"] = None
    SOFT_PID_REGISTRY.pop("backend_pipeline", None)

    try:
        st.session_state["pipeline_running"] = True
        result = subprocess.Popen(
            args,
            cwd=str(PROJECT_ROOT.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            bufsize=1,
        )
    except Exception as exc:
        return False, f"Failed to launch pipeline: {exc}"

    st.session_state["backend_pipeline_proc"] = result
    SOFT_PID_REGISTRY["backend_pipeline"] = result.pid

    output_lines: List[str] = []
    try:
        if result.stdout:
            for raw_line in result.stdout:
                line = raw_line.rstrip("\r\n")
                output_lines.append(line)
                if on_output:
                    on_output(line)
        return_code = result.wait()
    finally:
        st.session_state["backend_pipeline_proc"] = None
        SOFT_PID_REGISTRY.pop("backend_pipeline", None)
        st.session_state["pipeline_running"] = False

    success = return_code == 0
    if success:
        message = "Pipeline completed successfully."
    else:
        tail_lines = [line for line in output_lines if line][-20:]
        message = "\n".join(tail_lines).strip() or "Pipeline failed."
    return success, message


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def _load_backend_resources(index_dir: Path) -> Dict[str, Any]:
    manifest_path = index_dir / "index_manifest.json"
    meta_path = index_dir / "meta.jsonl"
    if not manifest_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Backend index manifest or meta.jsonl is missing.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metas: List[Dict[str, Any]] = []
    with meta_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                metas.append(json.loads(line))

    backend_type = manifest.get("backend", "faiss")
    faiss_index = None
    vectors = None
    if backend_type == "faiss":
        if faiss is None:
            raise RuntimeError("faiss is required to load this index but is not available.")
        index_path = index_dir / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index missing: {index_path}")
        faiss_index = faiss.read_index(str(index_path))
    elif backend_type == "bruteforce":
        vector_path = index_dir / "vectors.npy"
        if not vector_path.exists():
            raise FileNotFoundError(f"Vector file missing: {vector_path}")
        vectors = np.load(vector_path)
    else:
        raise RuntimeError(f"Unsupported backend type: {backend_type}")

    return {
        "manifest": manifest,
        "metas": metas,
        "faiss_index": faiss_index,
        "vectors": vectors,
    }


def ensure_backend_index_cache(model_name: str) -> Dict[str, Any]:
    key = safe_model_dir(model_name)
    cache: Dict[str, Dict[str, Any]] = st.session_state.setdefault("backend_index_cache", {})
    if key not in cache:
        cache[key] = _load_backend_resources(resolve_index_dir(model_name))
    return cache[key]


def invalidate_backend_index_cache(model_name: Optional[str] = None):
    if "backend_index_cache" not in st.session_state:
        return
    if model_name is None:
        st.session_state.backend_index_cache = {}
        return
    key = safe_model_dir(model_name)
    st.session_state.backend_index_cache.pop(key, None)


def embed_query_vector(question: str, embedding_backend: str, embedding_model: str) -> np.ndarray:
    embeddings = make_embeddings_client(embedding_backend, embedding_model)
    vector = embeddings.embed_query(question)
    array = np.asarray([vector], dtype="float32")
    return _l2_normalize(array)


def search_backend_index(question: str, top_k: int) -> List[Dict[str, Any]]:
    resources = ensure_backend_index_cache(st.session_state.embed_model)
    qvec = embed_query_vector(question, st.session_state.embedding_backend, st.session_state.embed_model)

    backend_type = resources["manifest"].get("backend", "faiss")
    metas = resources["metas"]
    total = len(metas)
    if total == 0:
        return []
    k = min(top_k, total)
    results: List[Tuple[float, int]] = []

    if backend_type == "faiss":
        faiss_index = resources["faiss_index"]
        if faiss_index is None:
            raise RuntimeError("FAISS index not loaded.")
        distances, indices = faiss_index.search(qvec, k)
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append((float(score), int(idx)))
    elif backend_type == "bruteforce":
        vectors = resources["vectors"]
        if vectors is None:
            raise RuntimeError("Vector array not loaded.")
        sims = (vectors @ qvec.T).reshape(-1)
        top_indices = np.argsort(-sims)[:k]
        for idx in top_indices:
            results.append((float(sims[int(idx)]), int(idx)))
    else:
        raise RuntimeError(f"Unsupported backend: {backend_type}")

    ranked: List[Dict[str, Any]] = []
    for score, idx in results:
        if idx < 0 or idx >= total:
            continue
        meta = metas[idx]
        ranked.append({"score": score, "meta": meta})
    return ranked


def format_backend_context(chunks: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for idx, item in enumerate(chunks, start=1):
        meta = item["meta"]
        source = meta.get("source_filename") or meta.get("filename") or meta.get("doc_id", "unknown")
        heading = meta.get("section_heading") or meta.get("primary_heading") or meta.get("breadcrumbs") or ""
        prefix = f"[{idx}] {source}"
        if heading:
            prefix += f" | {heading}"
        text = meta.get("text") or ""
        parts.append(f"{prefix}\n{text}")
    return "\n\n---\n\n".join(parts)


def _build_chat_llm(chat_model: str) -> ChatOpenAI:
    """Return a ChatOpenAI client configured for the active run mode."""
    run_mode = st.session_state.get("run_mode", "local")
    temperature = 0

    if run_mode == "openai":
        api_key = st.session_state.get("openai_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI API key is missing. Enter it in the sidebar or set OPENAI_API_KEY."
            )
        return ChatOpenAI(model=chat_model, temperature=temperature, api_key=api_key)

    base_url = st.session_state.get("localai_base_url") or DEFAULT_LOCALAI_BASE_URL
    api_key = os.getenv("LOCALAI_API_KEY") or DEFAULT_LOCALAI_API_KEY
    return ChatOpenAI(model=chat_model, temperature=temperature, base_url=base_url, api_key=api_key)


def _normalize_query(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return (
        normalized.encode("ascii", "ignore")
        .decode("ascii")
        .lower()
        .strip()
    )


def maybe_handle_smalltalk(question: str) -> Optional[str]:
    normalized = _normalize_query(question)
    if not normalized:
        return None

    if normalized in SMALL_TALK_KEYWORDS:
        return (
            "Xin chào! Tôi đang kết nối với kho đề cương học phần. "
            "Bạn có thể hỏi về mục tiêu, nội dung, thời lượng, điều kiện của bất kỳ học phần nào."
        )

    if any(phrase in normalized for phrase in CAPABILITY_PHRASES):
        return (
            "Tôi có thể tìm thông tin trong các đề cương môn học, trích dẫn nguồn và tóm tắt nội dung chính. "
            "Chỉ cần nêu tên học phần hoặc câu hỏi cụ thể, tôi sẽ tìm trong dữ liệu và trả lời kèm nguồn."
        )

    return None


def call_llm(chat_model: str, question: str, context: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Bạn là trợ lý RAG cho đề cương môn học. "
                "Luôn ưu tiên sử dụng nội dung trong Context. "
                "Mỗi khi trích dẫn, ghi chú dạng [số] tương ứng với mục trong Context. "
                "Nếu Context không có thông tin phù hợp, hãy nói rõ và đề nghị người dùng cung cấp chi tiết hơn, "
                "không tự bịa ra dữ kiện.",
            ),
            ("human", "Question:\n{question}\n\nContext:\n{context}\n\nAnswer in Vietnamese."),
        ]
    )

    llm = _build_chat_llm(chat_model)
    messages = prompt.format_messages(question=question, context=context)
    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)


# -----------------------------------------------------------------------------#
# Streamlit helpers
# -----------------------------------------------------------------------------#


def apply_material_theme():
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


.runtime-status {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    border: 1px solid transparent;
    margin-bottom: 0.6rem;
    text-align: center;
    gap: 0.25rem;
}

.runtime-status--on {
    background: rgba(80, 200, 138, 0.18);
    color: #2f9e5b;
    border-color: rgba(80, 200, 138, 0.32);
}

.runtime-status--off {
    background: rgba(234, 76, 76, 0.14);
    color: #d66565;
    border-color: rgba(234, 76, 76, 0.28);
}
.runtime-status--pending {
    background: rgba(255, 193, 7, 0.18);
    color: #a76a00;
    border-color: rgba(255, 193, 7, 0.32);
}
.runtime-status--missing {
    background: rgba(148, 163, 184, 0.18);
    color: #475569;
    border-color: rgba(148, 163, 184, 0.30);
}

.runtime-button-row {
    display: flex;
    gap: 0.6rem;
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


def init_session():
    existing_proc = st.session_state.get("backend_pipeline_proc")
    if existing_proc and existing_proc.poll() is None:
        try:
            existing_proc.terminate()
            existing_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            existing_proc.kill()
        except Exception:
            pass
    st.session_state["backend_pipeline_proc"] = None
    SOFT_PID_REGISTRY.pop("backend_pipeline", None)

    if "run_mode" not in st.session_state:
        default_mode = "local" if st.session_state.get("embedding_backend") == "tei" else "openai"
        st.session_state.run_mode = default_mode
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
    if "pipeline_running" not in st.session_state:
        st.session_state.pipeline_running = False
    if "pipeline_request" not in st.session_state:
        st.session_state.pipeline_request = None
    if "pipeline_feedback" not in st.session_state:
        st.session_state.pipeline_feedback = None

    if st.session_state.embedding_backend == "openai" and st.session_state.embed_model not in OPENAI_EMBED_MODELS:
        st.session_state.embed_model = OPENAI_EMBED_MODELS[0]
    if st.session_state.embedding_backend == "tei" and st.session_state.embed_model not in TEI_MODELS:
        st.session_state.embed_model = list(TEI_MODELS.keys())[0]
    if "backend_index_cache" not in st.session_state:
        st.session_state.backend_index_cache = {}



def render_settings_body():
    st.title("Settings")

    run_mode_display = st.selectbox(
        "Run mode",
        options=list(RUN_MODE_OPTIONS.keys()),
        index=list(RUN_MODE_OPTIONS.values()).index(st.session_state.get("run_mode", "local")),
        disabled=st.session_state.get("pipeline_running", False),
    )
    run_mode = RUN_MODE_OPTIONS[run_mode_display]
    st.session_state.run_mode = run_mode
    st.session_state.embedding_backend = "openai" if run_mode == "openai" else "tei"

    if run_mode == "openai":
        st.session_state.openai_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_key,
            type="password",
            help="Enter your API key here if you do not have a .env file.",
            disabled=st.session_state.get("pipeline_running", False),
        )
        if st.session_state.openai_key:
            os.environ["OPENAI_API_KEY"] = st.session_state.openai_key
    else:
        st.caption("Local mode uses Text-Embeddings-Inference and LocalAI")

    st.divider()

    if run_mode == "openai":
        embed_options = OPENAI_EMBED_MODELS
        if st.session_state.embed_model not in embed_options:
            st.session_state.embed_model = embed_options[0]
        selected_embed = st.selectbox(
            "Embedding model",
            options=embed_options,
            index=embed_options.index(st.session_state.embed_model),
            disabled=st.session_state.get("pipeline_running", False),
        )
        st.session_state.embed_model = selected_embed
    else:
        tei_options = list(TEI_MODELS.keys())
        if st.session_state.embed_model not in tei_options:
            st.session_state.embed_model = tei_options[0]
        selected_embed = st.selectbox(
            "Embedding model",
            options=tei_options,
            index=tei_options.index(st.session_state.embed_model),
            format_func=lambda key: TEI_MODELS[key]["display"],
            disabled=st.session_state.get("pipeline_running", False),
        )
        st.session_state.embed_model = selected_embed

    if run_mode == "openai":
        chat_options = OPENAI_CHAT_MODELS
        if st.session_state.chat_model not in chat_options:
            st.session_state.chat_model = chat_options[0]
        selected_chat = st.selectbox(
            "Chat model",
            options=chat_options,
            index=chat_options.index(st.session_state.chat_model),
            disabled=st.session_state.get("pipeline_running", False),
        )
        st.session_state.chat_model = selected_chat
    else:
        display_options = list(LOCAL_CHAT_MODELS.keys())
        current_display = next(
            (name for name, value in LOCAL_CHAT_MODELS.items() if value == st.session_state.chat_model),
            display_options[0],
        )
        selected_display = st.selectbox(
            "Chat model",
            options=display_options,
            index=display_options.index(current_display),
            disabled=st.session_state.get("pipeline_running", False),
        )
        st.session_state.chat_model = LOCAL_CHAT_MODELS[selected_display]

    current_chunk_mode = st.session_state.get("chunk_mode", DEFAULT_CHUNK_MODE)
    new_chunk_mode = st.selectbox(
        "Chunk mode",
        options=list(CHUNK_MODES.keys()),
        index=list(CHUNK_MODES.keys()).index(current_chunk_mode),
        format_func=lambda key: CHUNK_MODES[key],
        disabled=st.session_state.get("pipeline_running", False),
    )
    st.session_state.chunk_mode = new_chunk_mode
    st.session_state.retriever_k = st.slider(
        "Top-k passages",
        min_value=2,
        max_value=10,
        step=1,
        value=st.session_state.get("retriever_k", 4),
        disabled=st.session_state.get("pipeline_running", False),
    )

    if run_mode == "local":
        render_local_runtime_controls()

def _trigger_streamlit_rerun() -> None:
    """Call the available Streamlit rerun API across versions."""
    rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun_fn is None:
        raise RuntimeError("Streamlit rerun function not available.")
    rerun_fn()


def _render_runtime_control(
    title: str,
    running: bool,
    feedback_key: str,
    start_cb,
    stop_cb,
    button_key: str,
    status_detail: Optional[str] = None,
    disabled: bool = False,
) -> List[str]:
    suppressed_messages: List[str] = []
    st.markdown(f"**{title}**")
    status_class = "runtime-status--on" if running else "runtime-status--off"
    status_label = "Running" if running else "Stopped"

    st.markdown(f'<span class="runtime-status {status_class}">{status_label}</span>', unsafe_allow_html=True)

    action_label = "Stop" if running else "Start"
    action_cb = stop_cb if running else start_cb
    if st.button(
        action_label,
        key=button_key,
        type="primary",
        use_container_width=True,
        disabled=disabled,
    ):
        success, message = action_cb()
        if success:
            st.session_state[feedback_key] = ("success", None)
        else:
            st.session_state[feedback_key] = ("error", message)
        _trigger_streamlit_rerun()

    if status_detail and _is_global_docker_error(status_detail):
        suppressed_messages.append(status_detail)
        status_detail = None

    detail_placeholder = st.empty()
    if status_detail:
        detail_placeholder.caption(status_detail)
    else:
        detail_placeholder.empty()

    feedback = st.session_state.get(feedback_key)
    if feedback:
        status, message = feedback
        if status != "success" and message:
            if _is_global_docker_error(message):
                suppressed_messages.append(message)
            else:
                st.error(message)
        st.session_state[feedback_key] = None

    return suppressed_messages


def _render_status_badge(target, label: str, css_class: str) -> None:
    """Render or update the runtime status pill."""
    target.markdown(
        f'<span class="runtime-status {css_class}">{label}</span>',
        unsafe_allow_html=True,
    )


def _is_global_docker_error(message: Optional[str]) -> bool:
    if not message:
        return False
    lowered = message.lower()
    return any(snippet in lowered for snippet in GLOBAL_DOCKER_ERROR_SNIPPETS)


def _load_download_targets(script_path: Path) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return required/optional file lists declared by a download helper script."""
    try:
        module_name = f"_tei_download_{script_path.stem.replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if not spec or not spec.loader:
            return (), ()
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        required = tuple(getattr(module, "REQUIRED_FILES", ()))
        optional = tuple(getattr(module, "OPTIONAL_FILES", ()))
        return required, optional
    except Exception:
        return (), ()


def _download_tei_model_with_progress(
    model_key: str,
    status_placeholder,
) -> Tuple[bool, str]:
    config = TEI_MODELS.get(model_key)
    if not config:
        return False, f"Unknown TEI model: {model_key}"

    script_path = config.get("download_script")
    if not script_path or not script_path.exists():
        return False, f"Download script not found: {script_path}"

    required, optional = _resolve_model_targets(model_key)
    base_dir: Path = config["local_dir"]

    missing_targets = [
        rel_path for rel_path in (*required, *optional) if not (base_dir / rel_path).exists()
    ]

    if not missing_targets:
        return True, "Model assets already present."

    total_targets = len(missing_targets)
    if total_targets <= 0:
        total_targets = 1

    _render_status_badge(status_placeholder, "Downloading", "runtime-status--pending")

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--target",
                str(base_dir),
            ],
            cwd=str(BACKEND_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        return False, f"Failed to start download: {exc}"

    if result.returncode != 0:
        message = (result.stdout or result.stderr or "").strip() or "Model download failed."
        return False, message

    for rel_path in required:
        if not (base_dir / rel_path).exists():
            message = f"Missing required file after download: {rel_path}"
            return False, message

    return True, "Model assets downloaded."


def _pull_tei_image(
    runtime_key: str,
    status_placeholder,
) -> Tuple[bool, str]:
    _ = runtime_key  # runtime is fixed to CPU mode but kept for signature compatibility
    image = DEFAULT_TEI_RUNTIME_IMAGE
    if not image:
        return True, "No Docker image configured for this runtime."

    _render_status_badge(status_placeholder, "Pulling image", "runtime-status--pending")

    try:
        result = subprocess.run(
            ["docker", "pull", image],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError:
        return False, "Docker command not found. Install Docker Desktop/Engine and try again."
    except OSError as exc:
        return False, f"Failed to execute docker pull: {exc}"

    if result.returncode != 0:
        message = (result.stdout or result.stderr or "").strip() or "Failed to pull Docker image."
        return False, message

    return True, f"Docker image {image} ready."


def _handle_start_tei_runtime(
    model_key: str,
    runtime_mode: str,
    status_placeholder,
) -> Tuple[bool, str]:
    needs_download = not tei_model_is_downloaded(model_key)

    if needs_download:
        ok, message = _download_tei_model_with_progress(
            model_key,
            status_placeholder,
        )
        if not ok:
            _render_status_badge(status_placeholder, "Not Downloaded", "runtime-status--missing")
            return False, message
        _render_status_badge(status_placeholder, "Stopped", "runtime-status--off")
    else:
        pass

    pull_ok, pull_message = _pull_tei_image(runtime_mode, status_placeholder)
    if not pull_ok:
        _render_status_badge(status_placeholder, "Stopped", "runtime-status--off")
        return False, pull_message

    _render_status_badge(status_placeholder, "Starting", "runtime-status--pending")
    port = get_tei_model_port(model_key)
    success, message = start_tei_runtime(model_key, runtime_mode, port)
    if not success:
        _render_status_badge(status_placeholder, "Stopped", "runtime-status--off")
    else:
        _render_status_badge(status_placeholder, "Running", "runtime-status--on")
    return success, message


def _render_tei_runtime_control(
    tei_status: Dict[str, Any],
    model_key: str,
    runtime_mode: str,
    status_detail: Optional[str],
    disabled: bool,
) -> List[str]:
    suppressed_messages: List[str] = []
    tei_running = bool(tei_status.get("running"))
    downloaded = tei_model_is_downloaded(model_key)
    download_in_progress = st.session_state.get("tei_download_in_progress", False)

    st.markdown("**Embedding runtime**")

    status_placeholder = st.empty()
    if tei_running:
        status_label, status_class = "Running", "runtime-status--on"
    elif download_in_progress:
        status_label, status_class = "Downloading", "runtime-status--pending"
    elif downloaded:
        status_label, status_class = "Stopped", "runtime-status--off"
    else:
        status_label, status_class = "Not Downloaded", "runtime-status--missing"

    _render_status_badge(status_placeholder, status_label, status_class)

    action_label = "Stop" if tei_running else "Start"
    action_disabled = disabled or (download_in_progress and not tei_running)

    if st.button(
        action_label,
        key="tei_runtime_button",
        type="primary",
        use_container_width=True,
        disabled=action_disabled,
    ):
        if tei_running:
            success, message = stop_tei_runtime(model_key, runtime_mode)
        else:
            st.session_state["tei_download_in_progress"] = True
            try:
                success, message = _handle_start_tei_runtime(
                    model_key,
                    runtime_mode,
                    status_placeholder,
                )
            finally:
                st.session_state["tei_download_in_progress"] = False

        if success:
            st.session_state["tei_runtime_feedback"] = ("success", None)
        else:
            st.session_state["tei_runtime_feedback"] = ("error", message)
        _trigger_streamlit_rerun()

    if status_detail and _is_global_docker_error(status_detail):
        suppressed_messages.append(status_detail)
        status_detail = None

    detail_placeholder = st.empty()
    if status_detail:
        detail_placeholder.caption(status_detail)
    else:
        detail_placeholder.empty()

    feedback = st.session_state.get("tei_runtime_feedback")
    if feedback:
        status, message = feedback
        if status != "success" and message:
            if _is_global_docker_error(message):
                suppressed_messages.append(message)
            else:
                st.error(message)
        st.session_state["tei_runtime_feedback"] = None

    return suppressed_messages


def render_local_runtime_controls() -> None:
    pipeline_running = st.session_state.get("pipeline_running", False)
    model_key = st.session_state.embed_model
    runtime_mode = st.session_state.tei_runtime_mode
    tei_status = get_tei_runtime_status(model_key, runtime_mode)
    localai_running = localai_is_running()
    pipeline_request = st.session_state.get("pipeline_request")

    st.subheader("Docker Runtime control")
    tei_detail: Optional[str] = None
    error_detail = tei_status.get("error")
    if error_detail:
        tei_detail = error_detail
    elif tei_status.get("match"):
        port = get_tei_model_port(model_key)
        if not st.session_state.get("tei_base_url"):
            st.session_state.tei_base_url = f"http://localhost:{port}"
    elif tei_status.get("others"):
        label = format_tei_container_label(tei_status["others"][0])
        tei_detail = f"Different TEI running: {label}"

    col_embed, col_chat = st.columns(2, gap="large")
    shared_warnings: List[str] = []

    with col_embed:
        shared_warnings.extend(
            _render_tei_runtime_control(
                tei_status,
                model_key,
                runtime_mode,
                tei_detail,
                disabled=pipeline_running,
            )
        )

    with col_chat:
        shared_warnings.extend(
            _render_runtime_control(
                "Chat runtime",
                localai_running,
                "localai_runtime_feedback",
                start_localai_service,
                stop_localai_service,
                "localai_runtime_button",
                status_detail=None,
                disabled=pipeline_running,
            )
        )

    if shared_warnings:
        unique_messages: List[str] = []
        for message in shared_warnings:
            normalized = message.strip()
            if not normalized or normalized in unique_messages:
                continue
            unique_messages.append(normalized)
        if unique_messages:
            st.error("\n\n".join(unique_messages))

    docx_langs = detect_docx_languages()
    current_index_dir = resolve_index_dir(model_key)

    if pipeline_request:
        request_model_key = pipeline_request.get("model_key", model_key)
        langs = pipeline_request.get("langs") or (docx_langs or ["vi"])
        base_url = pipeline_request.get("base_url") or st.session_state.get("tei_base_url") or os.getenv("TEI_BASE_URL")
        index_dir_str = pipeline_request.get("index_dir")
        chunk_mode = pipeline_request.get("chunk_mode")
        target_index_dir = Path(index_dir_str) if index_dir_str else resolve_index_dir(request_model_key)

        progress_status = st.empty()
        progress_status.info("Preparing pipeline...")
        progress_container = st.container()
        step_pattern = re.compile(r"Step\s+(\d+)/(\d+)")
        embedding_progress_pattern = re.compile(
            r"Embedding:\s+(?P<pct>\d+)%\|.*?\|\s*(?P<done>\d+)/(?P<total>\d+)"
        )
        progress_state: Dict[str, Any] = {
            "bars": {},
            "active_step": None,
            "total": None,
            "last_values": {},
        }

        def ensure_step_bars(total_steps: int) -> None:
            if progress_state["bars"]:
                return
            for idx in range(1, total_steps + 1):
                label = PIPELINE_STEP_MESSAGES.get(idx, f"Step {idx}/{total_steps}")
                row = progress_container.container()
                row.caption(label)
                bar = row.progress(0)
                percent_placeholder = row.empty()
                progress_state["bars"][idx] = {"bar": bar, "percent": percent_placeholder}
                progress_state["last_values"][idx] = 0

        def mark_complete(step_idx: Optional[int]) -> None:
            if not step_idx:
                return
            entry = progress_state["bars"].get(step_idx)
            if entry:
                entry["bar"].progress(100)
                entry["percent"].markdown("**100%**")
                progress_state["last_values"][step_idx] = 100

        def handle_pipeline_output(line: str) -> None:
            match = step_pattern.search(line)
            if match:
                current = int(match.group(1))
                total = max(int(match.group(2)), 1)
                progress_state["total"] = total
                ensure_step_bars(total)

                previous = progress_state.get("active_step")
                if previous and previous != current:
                    mark_complete(previous)

                progress_state["active_step"] = current
                entry = progress_state["bars"].get(current)
                if entry:
                    last_value = progress_state["last_values"].get(current, 0)
                    if last_value < 5:
                        last_value = 5
                    entry["bar"].progress(last_value)
                    entry["percent"].markdown(f"**{last_value}%**")
                    progress_state["last_values"][current] = last_value

                label = PIPELINE_STEP_MESSAGES.get(current, f"Running Step {current}/{total}")
                progress_status.info(label)
                return

            progress_match = embedding_progress_pattern.search(line)
            if progress_match:
                pct = int(progress_match.group("pct"))
                step_idx = progress_state.get("active_step") or progress_state.get("total")
                if step_idx and step_idx in progress_state["bars"]:
                    entry = progress_state["bars"][step_idx]
                    bar = entry["bar"]
                    previous_pct = progress_state["last_values"].get(step_idx, 0)
                    pct = max(previous_pct, min(pct, 100))
                    bar.progress(pct)
                    entry["percent"].markdown(f"**{pct}%**")
                    progress_state["last_values"][step_idx] = pct

            if "Pipeline completed successfully" in line:
                mark_complete(progress_state.get("active_step"))
                return

        success, message = run_backend_pipeline(
            request_model_key,
            langs,
            base_url,
            target_index_dir,
            chunk_mode,
            on_output=handle_pipeline_output,
        )

        if success:
            mark_complete(progress_state.get("active_step"))
            for idx, entry in progress_state["bars"].items():
                if idx != progress_state.get("active_step"):
                    entry["bar"].progress(100)
                    entry["percent"].markdown("**100%**")
                    progress_state["last_values"][idx] = 100
            st.session_state.pipeline_feedback = ("success", message or "Pipeline completed successfully.")
            invalidate_backend_index_cache(request_model_key)
        else:
            st.session_state.pipeline_feedback = ("error", message or "Pipeline failed.")
            progress_status.error("Pipeline failed. See details below.")

        st.session_state.pipeline_request = None
        st.session_state.pipeline_running = False
        _trigger_streamlit_rerun()
        st.stop()

def render_sidebar_quick_actions():
    pipeline_running = st.session_state.get("pipeline_running", False)
    pipeline_request = st.session_state.get('pipeline_request')
    pipeline_feedback = st.session_state.get("pipeline_feedback")

    st.subheader("Data")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=sorted(UPLOAD_ALLOWED_EXTS),
        accept_multiple_files=True,
        help="Files are saved to `backend/data/raw/uploads/<ext>/`.",
        disabled=pipeline_running,
    )
    if uploaded_files:
        saved_paths: List[Path] = []
        errors: List[str] = []
        for file in uploaded_files:
            suffix = Path(file.name).suffix.lower()
            ext = suffix.lstrip('.')
            if ext not in UPLOAD_ALLOWED_EXTS:
                errors.append(f"Unsupported file type: {file.name}")
                continue
            target_dir = UPLOADS_ROOT / ext
            target_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(file.name).stem.strip()
            safe_stem = ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '-' for ch in stem).strip('-') or 'document'
            unique_name = f"{safe_stem}-{uuid4().hex[:8]}{suffix}"
            target_path = target_dir / unique_name
            try:
                with open(target_path, 'wb') as out_file:
                    out_file.write(file.getbuffer())
                saved_paths.append(target_path)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Failed to save {file.name}: {exc}")
        messages = []
        if saved_paths:
            rel_paths = [str(path.relative_to(PROJECT_ROOT.parent)) for path in saved_paths]
            messages.append('Saved: ' + ', '.join(rel_paths))
        if errors:
            messages.extend(errors)
        if saved_paths and errors:
            status = 'info'
        elif saved_paths:
            status = 'success'
        elif errors:
            status = 'error'
        else:
            status = 'info'
        st.session_state.upload_feedback = (status, ' | '.join(messages) if messages else 'No files processed.')
        st.rerun()

    upload_feedback = st.session_state.upload_feedback
    if upload_feedback:
        status, message = upload_feedback
        if status in ('success', 'info'):
            st.info(message)
        else:
            st.error(message)
        st.session_state.upload_feedback = None

    st.divider()
    current_index_dir = resolve_index_dir(st.session_state.embed_model)
    docx_langs = detect_docx_languages()
    runtime_mode = st.session_state.tei_runtime_mode
    model_key = st.session_state.embed_model

    if pipeline_request:
        request_model_key = pipeline_request.get('model_key', model_key)
        langs = pipeline_request.get('langs') or (docx_langs or ['vi'])
        base_url = (
            pipeline_request.get('base_url')
            or st.session_state.get('tei_base_url')
            or os.getenv('TEI_BASE_URL')
        )
        index_dir_str = pipeline_request.get('index_dir')
        chunk_mode = pipeline_request.get('chunk_mode')
        target_index_dir = Path(index_dir_str) if index_dir_str else resolve_index_dir(request_model_key)

        progress_status = st.empty()
        progress_status.info('Preparing pipeline...')
        progress_container = st.container()
        step_pattern = re.compile(r'Step\s+(\d+)/(\d+)')
        embedding_progress_pattern = re.compile(
            r'Embedding:\s+(?P<pct>\d+)%\|.*?\|\s*(?P<done>\d+)/(?P<total>\d+)'
        )
        progress_state: Dict[str, Any] = {
            'bars': {},
            'active_step': None,
            'total': None,
            'last_values': {},
        }

        def ensure_step_bars(total_steps: int) -> None:
            if progress_state['bars']:
                return
            for idx in range(1, total_steps + 1):
                label = PIPELINE_STEP_MESSAGES.get(idx, f'Step {idx}/{total_steps}')
                row = progress_container.container()
                row.caption(label)
                bar = row.progress(0)
                percent_placeholder = row.empty()
                progress_state['bars'][idx] = {'bar': bar, 'percent': percent_placeholder}
                progress_state['last_values'][idx] = 0

        def mark_complete(step_idx: Optional[int]) -> None:
            if not step_idx:
                return
            entry = progress_state['bars'].get(step_idx)
            if entry:
                entry['bar'].progress(100)
                entry['percent'].markdown('**100%**')
                progress_state['last_values'][step_idx] = 100

        def handle_pipeline_output(line: str) -> None:
            match = step_pattern.search(line)
            if match:
                current = int(match.group(1))
                total = max(int(match.group(2)), 1)
                progress_state['total'] = total
                ensure_step_bars(total)

                previous = progress_state.get('active_step')
                if previous and previous != current:
                    mark_complete(previous)

                progress_state['active_step'] = current
                entry = progress_state['bars'].get(current)
                if entry:
                    last_value = progress_state['last_values'].get(current, 0)
                    if last_value < 5:
                        last_value = 5
                    entry['bar'].progress(last_value)
                    entry['percent'].markdown(f"**{last_value}%**")
                    progress_state['last_values'][current] = last_value

                label = PIPELINE_STEP_MESSAGES.get(current, f'Step {current}/{total}')
                progress_status.info(label)
                return

            progress_match = embedding_progress_pattern.search(line)
            if progress_match:
                pct = int(progress_match.group('pct'))
                step_idx = progress_state.get('active_step') or progress_state.get('total')
                if step_idx and step_idx in progress_state['bars']:
                    entry = progress_state['bars'][step_idx]
                    bar = entry['bar']
                    previous_pct = progress_state['last_values'].get(step_idx, 0)
                    pct = max(previous_pct, min(pct, 100))
                    bar.progress(pct)
                    entry['percent'].markdown(f"**{pct}%**")
                    progress_state['last_values'][step_idx] = pct

            if 'Pipeline completed successfully' in line:
                mark_complete(progress_state.get('active_step'))
                return

        success, message = run_backend_pipeline(
            request_model_key,
            langs,
            base_url,
            target_index_dir,
            chunk_mode,
            on_output=handle_pipeline_output,
        )

        if success:
            mark_complete(progress_state.get('active_step'))
            for idx, entry in progress_state['bars'].items():
                if idx != progress_state.get('active_step'):
                    entry['bar'].progress(100)
                    entry['percent'].markdown('**100%**')
                    progress_state['last_values'][idx] = 100
            st.session_state.pipeline_feedback = ('success', message or 'Pipeline completed successfully.')
            invalidate_backend_index_cache(request_model_key)
        else:
            st.session_state.pipeline_feedback = ('error', message or 'Pipeline failed.')
            progress_status.error('Pipeline failed. See details below.')

        st.session_state.pipeline_request = None
        st.session_state.pipeline_running = False
        _trigger_streamlit_rerun()
        st.stop()

    if pipeline_feedback:
        status, message = pipeline_feedback
        if status == 'success':
            st.success(message or 'Pipeline completed successfully.')
        else:
            st.error(message or 'Pipeline run failed.')
        st.session_state.pipeline_feedback = None

    if st.button(
        'Rebuild backend index',
        use_container_width=True,
        disabled=pipeline_running,
        key="sidebar_rebuild_button",
    ):
        if st.session_state.embedding_backend != 'tei':
            st.error('Rebuild is only available in Local TEI mode.')
        elif not tei_model_is_downloaded(model_key):
            st.error('The selected TEI model has not been downloaded.')
        else:
            base_url = st.session_state.get('tei_base_url') or os.getenv('TEI_BASE_URL')
            if not base_url:
                st.error('TEI base URL is not configured. Start the TEI runtime first.')
            elif not tei_backend_is_active(model_key, runtime_mode):
                st.error('TEI runtime is not running. Start it before rebuilding.')
            else:
                langs = docx_langs or ['vi']
                st.session_state.pipeline_request = {
                    'model_key': st.session_state.embed_model,
                    'runtime_mode': runtime_mode,
                    'langs': langs,
                    'base_url': base_url,
                    'index_dir': str(current_index_dir),
                    'chunk_mode': st.session_state.chunk_mode,
                }
                st.session_state.pipeline_running = True
                st.session_state.pipeline_feedback = None
                _trigger_streamlit_rerun()
                st.stop()

    if st.button(
        'Clear chat history',
        use_container_width=True,
        disabled=pipeline_running,
        key="sidebar_clear_history_button",
    ):
        st.session_state.history = []
        st.rerun()

    index_state = 'yes' if index_exists(current_index_dir) else 'no'
    st.caption(
        f"DOCX in `backend/data/raw`: {len(list_docx_files())} | "
        f"Index `{current_index_dir.relative_to(PROJECT_ROOT.parent)}` present: {index_state}"
    )

    emb_used = load_embed_meta(current_index_dir)
    expected_backend = st.session_state.embedding_backend
    expected_model = st.session_state.embed_model
    expected_chunk = st.session_state.chunk_mode
    expected_backend_label = EMBED_BACKENDS.get(expected_backend, expected_backend.title())
    expected_model_display = format_embedding_display(expected_backend, expected_model)
    current_chunk_label = CHUNK_MODES.get(expected_chunk, expected_chunk if expected_chunk else 'Not configured')

    if emb_used:
        backend_key = emb_used.get('embedding_backend') or expected_backend
        backend_label = EMBED_BACKENDS.get(backend_key, backend_key.title())
        index_model = emb_used.get('embedding_model')
        chunk_value = emb_used.get('chunk_mode')
        chunk_label = CHUNK_MODES.get(chunk_value, chunk_value if chunk_value else 'Not recorded')
        display_name = format_embedding_display(backend_key, index_model if index_model else None)

        st.caption(f"Index built with: {backend_label} / {display_name} / {chunk_label}")

        model_matches = False
        if index_model and expected_model:
            if index_model == expected_model:
                model_matches = True
            else:
                model_matches = (
                    _slugify_identifier(str(index_model)) == _slugify_identifier(str(expected_model))
                )
        elif not index_model:
            model_matches = True

        chunk_matches = False
        if chunk_value and expected_chunk:
            chunk_matches = chunk_value == expected_chunk
        else:
            chunk_matches = True

        backend_matches = backend_key == expected_backend

        if not (backend_matches and model_matches and chunk_matches):
            current_summary = f"{expected_backend_label} / {expected_model_display} / {current_chunk_label}"
            index_summary = f"{backend_label} / {display_name} / {chunk_label}"
            st.info(
                'The current embedding selection differs from the index. Rebuild to avoid inconsistencies.\n\n'
                f"Index: {index_summary}\n"
                f"Current selection: {current_summary}"
            )
    else:
        st.caption(
            'Index status: Not built for '
            f"{expected_backend_label} / {expected_model_display} / {current_chunk_label}"
        )

def main():
    load_dotenv()
    ensure_dirs()
    init_session()

    st.set_page_config(page_title="RAG over PDFs", page_icon=":books:", layout="wide")
    apply_material_theme()

    with st.sidebar:
        render_sidebar()

    current_index_dir = resolve_index_dir(st.session_state.embed_model)

    if st.session_state.embedding_backend == "openai" and not st.session_state.openai_key:
        st.info("No OpenAI API key detected. Enter it in the sidebar or the .env file before generating embeddings.")
    elif st.session_state.embedding_backend == "tei":
        runtime_mode = st.session_state.tei_runtime_mode
        model_key = st.session_state.embed_model
        if not tei_backend_is_active(model_key, runtime_mode):
            st.info("The Docker-based TEI service is not running. Start it from the sidebar before continuing.")

    if not index_exists(current_index_dir):
        st.info("No backend index detected. Run the pipeline from the sidebar to build it from DOCX files.")

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                with st.expander("Sources"):
                    for idx, source in enumerate(turn["sources"], start=1):
                        st.markdown(f"**{idx}.** `{source['source']}` (p.{source.get('page', '?')})")
                        if source.get("snippet"):
                            st.caption(source["snippet"])

    pending_question = st.session_state.pop("_pending_question", None)
    user_question = pending_question or st.chat_input("Ask something about your documents...")

    if user_question:
        st.session_state.history.append({"role": "user", "content": user_question})

        smalltalk_reply = maybe_handle_smalltalk(user_question)
        if smalltalk_reply:
            st.session_state.history.append({"role": "assistant", "content": smalltalk_reply})
            st.rerun()

        runtime_mode = st.session_state.tei_runtime_mode
        tei_model_key = st.session_state.embed_model

        if st.session_state.embedding_backend != "tei":
            st.session_state.history.append({
                "role": "assistant",
                "content": "Retrieval requires Local TEI. Switch to Local TEI in the sidebar.",
            })
            st.rerun()

        if st.session_state.embedding_backend == "tei" and not tei_backend_is_active(tei_model_key, runtime_mode):
            st.session_state.history.append({
                "role": "assistant",
                "content": "TEI is not running. Start the container from the sidebar before continuing.",
            })
            st.rerun()

        if not tei_model_is_downloaded(tei_model_key):
            st.session_state.history.append({
                "role": "assistant",
                "content": "The selected TEI model has not been downloaded. Download it from the sidebar.",
            })
            st.rerun()

        if not backend_index_exists(current_index_dir):
            st.session_state.history.append({
                "role": "assistant",
                "content": 'No backend index detected. Click "Rebuild backend index" in the sidebar after preparing DOCX files.',
            })
            st.rerun()

        try:
            with st.spinner("Retrieving and reasoning..."):
                chunks = search_backend_index(user_question, st.session_state.retriever_k)
                if not chunks:
                    st.session_state.history.append({
                        "role": "assistant",
                        "content": "No relevant chunks found in the current index.",
                    })
                    st.rerun()

                context = format_backend_context(chunks)
                answer = call_llm(st.session_state.chat_model, user_question, context)

                sources = []
                for item in chunks:
                    meta = item["meta"]
                    source = meta.get("source_filename") or meta.get("filename") or meta.get("doc_id", "unknown")
                    snippet = meta.get("text") or ""
                    breadcrumbs = meta.get("breadcrumbs") or meta.get("section_heading")
                    sources.append({
                        "source": source,
                        "page": breadcrumbs,
                        "snippet": snippet[:400] + ("..." if snippet and len(snippet) > 400 else ""),
                    })

                st.session_state.history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
        except Exception as exc:
            st.session_state.history.append({"role": "assistant", "content": f"An error occurred: {exc}"})

        st.rerun()


def render_sidebar():
    render_settings_body()
    st.divider()
    render_sidebar_quick_actions()


if __name__ == "__main__":
    main()


