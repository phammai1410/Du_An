"""Streamlit UI for the PDF RAG assistant."""

# Tổng quan các phần của module:
# 1. Biến cấu hình/đường dẫn (cấu trúc thư mục, giá trị mặc định, metadata TEI).
# 2. Tiện ích quản lý runtime cho container TEI và LocalAI chat.
# 3. Hỗ trợ chung dùng lại giữa backend và frontend.
# 4. Tiện ích pipeline/index giao tiếp với các script ingest.
# 5. Logic truy hồi + trả lời (tìm chunk, heuristic, prompt).
# 6. Công cụ và widget Streamlit cho giao diện.
#
# Bảng tra cứu nhãn (Ctrl+F theo mã sau để nhảy đến phần mong muốn):
#   [S1]  Cấu hình & biến đường dẫn.
#   [S2]  Quản lý runtime Docker (TEI/LocalAI).
#   [S3]  Tiện ích chung (path, manifest, FS) dùng lại giữa backend/frontend.
#   [S4]  Tiện ích pipeline/index giao tiếp backend.
#   [S5A] Chấm điểm chunk + fallback lexical.
#   [S5B] Cache index & truy hồi vector.
#   [S5C] Heuristic giới hạn phạm vi khóa học.
#   [S5D] Prompt/LLM + xử lý small-talk.
#   [S5E] Heuristic đặc thù domain (giảng viên/tín chỉ...).
#   [S5F] Lắp ráp câu trả lời & phản hồi mặc định.
#   [S6A] Theme + khởi tạo session.
#   [S6B] Sidebar Settings (model, chunking, retriever).
#   [S6C] Điều khiển Docker runtime TEI/LocalAI.
#   [S6D] Quick actions: upload, rebuild index, feedback.
#   [S6E] View switcher + đăng nhập admin + quản lý file.
#   [S6F] Giao diện người dùng & điểm vào ứng dụng.

import importlib.util
import json
import os
import subprocess
import sys
import platform
import re
import socket
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Set, Union
from dataclasses import dataclass
from collections import defaultdict
from urllib.parse import urlparse
from uuid import uuid4

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import numpy as np
import unicodedata

try:
    import faiss  # type: ignore
except Exception:  # noqa: BLE001
    faiss = None  # type: ignore

SOFT_PID_REGISTRY: dict[str, int] = {}


# -----------------------------------------------------------------------------#
# [S1] Paths and configuration
# -----------------------------------------------------------------------------#
# Gom cấu trúc thư mục, biến môi trường, model mặc định và bảng tra cứu giúp
# các phần sau dùng cấu hình như dữ liệu thống nhất. Phần này cũng định nghĩa
# mọi giá trị mặc định (model, chunking, backend URL) để khi dò lỗi cấu hình
# chỉ cần tìm “[S1]” là thấy toàn bộ bối cảnh cấu hình.


def _safe_int(value: Optional[str], default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _port_from_url(url: str) -> Optional[int]:
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    if parsed.port:
        return parsed.port
    if parsed.scheme == "https":
        return 443
    if parsed.scheme == "http":
        return 80
    return None


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

DEFAULT_LOCALAI_BASE_URL = os.getenv("LOCALAI_BASE_URL", "http://localhost:8081/v1")
DEFAULT_LOCALAI_API_KEY = os.getenv("LOCALAI_API_KEY", "localai-temp-key")
_LOCALAI_PORT_ENV = os.getenv("LOCALAI_PORT")
LOCALAI_PORT = _safe_int(_LOCALAI_PORT_ENV, _port_from_url(DEFAULT_LOCALAI_BASE_URL) or 8081)
LOCALAI_PORT_LOCKED = _LOCALAI_PORT_ENV is not None

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
LOCALAI_IMAGE = os.getenv("LOCALAI_IMAGE", "localai/localai:latest")
LOCALAI_CONTAINER_NAME = os.getenv("LOCALAI_CONTAINER_NAME", "localai-runtime")
LOCALAI_RUNTIME_MODEL = os.getenv("LOCALAI_RUNTIME_MODEL", "llama-3.2-1b-instruct:q4_k_m")

DEFAULT_CHAT_MODEL = LOCALAI_RUNTIME_MODEL
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
TEXT_EDITABLE_EXTS = {"txt", "md", "markdown", "json", "yaml", "yml", "csv", "tsv", "ini", "toml"}

EMBED_BACKENDS: Dict[str, str] = {
    "tei": "Local Text-Embeddings-Inference",
}
GLOBAL_DOCKER_ERROR_SNIPPETS = ("docker desktop is manually paused",)

TEI_MODELS: Dict[str, Dict[str, Any]] = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "display": "MiniLM L6 v2 (22M/Q4)",
        "config_key": "sentence-transformers-all-MiniLM-L6-v2",
        "local_dir": LOCAL_TEI_ROOT / "sentence-transformers-all-MiniLM-L6-v2",
        "download_script": TOOLS_DIR / "download_all_minilm_l6_v2_tei.py",
        "required_file": "model.safetensors",
    },
    "intfloat/e5-small-v2": {
        "display": "IntFloat E5 Small v2 (33.4M)",
        "config_key": "intfloat-e5-small-v2",
        "local_dir": LOCAL_TEI_ROOT / "intfloat-e5-small-v2",
        "download_script": TOOLS_DIR / "download_e5_small_v2_tei.py",
        "required_file": "model.safetensors",
    },
}

LOCAL_CHAT_MODELS = {
    "LLAMA 3.1 1B": "llama-3.2-1b-instruct:q4_k_m",
}

USER_VIEW = "user"
ADMIN_VIEW = "admin"
ADMIN_PASSWORD_ENV_VAR = "ADMIN_PASSWORD"


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


# -----------------------------------------------------------------------------#
# [S2] Runtime management (Docker orchestration)
# -----------------------------------------------------------------------------#
# Bộ hàm kiểm tra/khởi động/dừng runtime TEI và LocalAI để UI hiển thị trạng
# thái/trình điều khiển runtime nhất quán. Khi cần xử lý container/port,
# cứ tìm “[S2]” để thấy toàn bộ logic start/stop, health-check và cấp phát port.


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


def _is_port_available(port: int, host: str = "0.0.0.0") -> bool:
    """Return True if the given host/port combination can be bound."""
    if port <= 0:
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(0.5)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _reserve_localai_port() -> Tuple[Optional[int], Optional[int]]:
    """Pick an available host port for the LocalAI container.

    Returns (available_port, conflicted_port). If available_port is None,
    the caller should report that conflicted_port is already in use.
    """
    preferred = LOCALAI_PORT or 8081
    if _is_port_available(preferred):
        return preferred, None
    if LOCALAI_PORT_LOCKED:
        return None, preferred
    search_range = list(range(8081, 8105))
    if preferred not in search_range:
        search_range.insert(0, preferred)
    for candidate in search_range:
        if candidate == preferred:
            continue
        if _is_port_available(candidate):
            return candidate, preferred
    return None, preferred


def set_localai_base_url(port: int) -> str:
    base_url = f"http://localhost:{port}/v1"
    os.environ["LOCALAI_BASE_URL"] = base_url
    st.session_state["localai_base_url"] = base_url
    return base_url


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
    selected_port, conflicted_port = _reserve_localai_port()
    if selected_port is None:
        return (
            False,
            (
                f"Port {conflicted_port} is already in use. "
                "Stop the service occupying it or set LOCALAI_PORT to a different free port."
            ),
        )
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
                f"{selected_port}:8080",
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
    set_localai_base_url(selected_port)
    message = result.stdout.strip() or f"LocalAI service started on port {selected_port}."
    if conflicted_port and conflicted_port != selected_port:
        message += f" (Port {conflicted_port} was busy, so {selected_port} was used instead.)"
    return True, message


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
# [S3] Utility helpers
# -----------------------------------------------------------------------------#
# Các wrapper nhỏ dùng chung cho pipeline ingest, logic truy hồi và UI
# Streamlit (chuẩn hóa đường dẫn, manifest, tạo thư mục, ...). Nhìn vào "[S3]"
# sẽ thấy nơi chuẩn hoá model -> thư mục, đọc metadata index và client TEI.


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
    base_url = st.session_state.get("tei_base_url") or os.getenv("TEI_BASE_URL", "http://localhost:8080")
    api_key = os.getenv("TEI_API_KEY")
    if backend != "tei":
        raise ValueError(f"Unsupported embedding backend: {backend}")
    return TEIEmbeddings(base_url=base_url, model=model_name, api_key=api_key)


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
# [S4] Index utilities (backend pipeline)
# -----------------------------------------------------------------------------#
# Quản lý pipeline ingest DOCX, phát hiện artifact và các helper nối FAISS/
# brute-force index với phía front-end. Dò các tác vụ ingest, tiến trình pipeline
# hay DOCX -> index chỉ cần search “[S4]”.


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


# -----------------------------------------------------------------------------#
# [S5] Retrieval and answer generation
# -----------------------------------------------------------------------------#
# Tải cache index, chạy truy hồi FAISS + kết hợp keyword, áp dụng heuristic đặc
# thù đề cương (doc_id, giảng viên, tín chỉ, ...), build prompt và dựng câu trả
# lời. Tìm “[S5]” để thấy toàn bộ tầng suy luận/tìm kiếm trước khi hiển thị UI.
# Các tiểu mục [S5A..F] giúp định vị nhanh các heuristics đặc thù.


# ----- [S5A] Chunk scoring & lexical fallback --------------------------------
# Gom helper chuẩn hoá vector, xác định chunk_id và fallback lexicon để kết hợp
# cùng FAISS. Đây là nơi cần soi khi muốn thay đổi thứ tự rank hoặc loại bỏ
# chunk trùng.


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def _is_connection_issue(exc: Exception) -> bool:
    """Return True if the exception looks like a transient connection failure."""
    if isinstance(exc, (requests.RequestException, ConnectionError, TimeoutError)):
        return True
    message = str(exc).lower()
    return any(token in message for token in ("connection", "timed out", "temporarily unavailable"))


def _chunk_identity(meta: Dict[str, Any]) -> str:
    """Return a deterministic identifier for a chunk for deduplication."""
    if not isinstance(meta, dict):
        return str(id(meta))
    for key in ("chunk_id", "id"):
        value = meta.get(key)
        if value is not None:
            return str(value)
    doc_id = meta.get("doc_id") or "doc"
    order = meta.get("chunk_order", "0")
    subindex = meta.get("chunk_subindex", "0")
    filename = meta.get("source_filename") or meta.get("filename") or "src"
    return f"{doc_id}-{order}-{subindex}-{filename}"


def _tokenize_keywords(question: str) -> List[str]:
    normalized = _normalize_query(question)
    if not normalized:
        return []
    tokens = re.split(r"[^a-z0-9]+", normalized)
    return [token for token in tokens if len(token) >= 3]


def _score_chunk_keywords(tokens: List[str], meta: Dict[str, Any]) -> float:
    if not tokens:
        return 0.0
    haystack_parts = [
        meta.get("section_heading"),
        meta.get("primary_heading"),
        meta.get("breadcrumbs"),
        _get_chunk_text(meta),
    ]
    haystack = _normalize_query(" ".join(part for part in haystack_parts if part))
    if not haystack:
        return 0.0
    raw_score = sum(1 for token in tokens if token in haystack)
    if raw_score == 0:
        return 0.0
    penalty = math.log1p(max(len(haystack), 1))
    return raw_score / penalty


def _keyword_search_chunks(
    question: str,
    limit: int,
    resources: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Perform a lightweight lexical scan over meta.jsonl chunks as a fallback."""
    tokens = _tokenize_keywords(question)
    if not tokens or limit <= 0:
        return []
    if resources is None:
        resources = ensure_backend_index_cache(st.session_state.embed_model)
    metas = resources.get("metas") or []
    scored: List[Tuple[float, int]] = []
    for idx, meta in enumerate(metas):
        score = _score_chunk_keywords(tokens, meta)
        if score > 0:
            scored.append((score, idx))
    if not scored:
        return []
    scored.sort(key=lambda item: item[0], reverse=True)
    results: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for score, meta_idx in scored:
        meta = metas[meta_idx]
        ident = _chunk_identity(meta)
        if ident in seen:
            continue
        seen.add(ident)
        results.append({"score": float(score), "meta": meta})
        if len(results) >= limit:
            break
    return results


def _merge_ranked_chunks(
    primary: List[Dict[str, Any]],
    fallback: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    """Merge vector and lexical results, keeping order and removing duplicates."""
    merged: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    def _consume(items: List[Dict[str, Any]]) -> None:
        for item in items:
            meta = item.get("meta") if isinstance(item, dict) else {}
            ident = _chunk_identity(meta or {})
            if ident in seen:
                continue
            seen.add(ident)
            merged.append(item)
            if len(merged) >= limit:
                break

    _consume(primary)
    if len(merged) < limit:
        _consume(fallback)
    return merged[:limit]


def _dedupe_chunks(chunks: List[Dict[str, Any]], max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    """Remove duplicate chunks by id/text to avoid repeated answers."""
    deduped: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    seen_texts: Set[str] = set()
    for chunk in chunks:
        meta = chunk.get("meta") if isinstance(chunk, dict) else {}
        ident = _chunk_identity(meta or {})
        text = re.sub(r"\s+", " ", (_get_chunk_text(meta) or "").strip())
        if ident in seen_ids or (text and text in seen_texts):
            continue
        seen_ids.add(ident)
        if text:
            seen_texts.add(text)
        deduped.append(chunk)
        if max_items and len(deduped) >= max_items:
            break
    return deduped


# ----- [S5B] Backend index cache & vector search -----------------------------
# Load manifest/meta.jsonl, giữ cache trong session và thực hiện FAISS/
# brute-force retrieval rồi kết hợp với lexical search. Các hàm dưới đây xử lý
# read/write cache, chuẩn hoá context và dựng chuỗi đầu vào LLM.


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


def _get_doc_meta(doc_id: str) -> Optional[Dict[str, Any]]:
    if not doc_id:
        return None
    meta_cache = st.session_state.setdefault("doc_meta_cache", {})
    if doc_id in meta_cache:
        return meta_cache[doc_id]
    resources = ensure_backend_index_cache(st.session_state.embed_model)
    for meta in resources.get("metas", []):
        if meta.get("doc_id") == doc_id:
            meta_cache[doc_id] = meta
            return meta
    return None


def _load_course_document(doc_id: str) -> Optional[Dict[str, Any]]:
    if not doc_id:
        return None
    doc_cache = st.session_state.setdefault("course_document_cache", {})
    if doc_id in doc_cache:
        return doc_cache[doc_id]
    meta = _get_doc_meta(doc_id)
    if not meta:
        return None
    source_path = meta.get("source_path")
    if not source_path:
        return None
    try:
        data = json.loads(Path(source_path).read_text(encoding="utf-8"))
    except OSError:
        return None
    doc_cache[doc_id] = data
    return data


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
    try:
        qvec = embed_query_vector(
            question,
            st.session_state.embedding_backend,
            st.session_state.embed_model,
        )
    except Exception as exc:
        if _is_connection_issue(exc):
            return _keyword_search_chunks(question, top_k, resources)
        raise

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

    if len(ranked) < k:
        lexical = _keyword_search_chunks(question, max(k * 2, 10), resources)
        if lexical:
            ranked = _merge_ranked_chunks(ranked, lexical, k)

    if not ranked:
        ranked = _keyword_search_chunks(question, k, resources)

    return ranked


def retrieve_relevant_chunks(question: str) -> List[Dict[str, Any]]:
    """Full retrieval pipeline with course filtering and attribute expansion."""
    target_doc_ids = _identify_target_doc_ids(question)
    chunks = search_backend_index(question, st.session_state.retriever_k)
    chunks = _filter_chunks_by_course(question, chunks, target_doc_ids)
    chunks = _maybe_expand_instructor_chunks(question, chunks, target_doc_ids)
    chunks = _ensure_attribute_chunks(question, chunks, target_doc_ids)
    return _dedupe_chunks(chunks)


def format_backend_context(chunks: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for idx, item in enumerate(chunks, start=1):
        meta = item["meta"]
        source = meta.get("source_filename") or meta.get("filename") or meta.get("doc_id", "unknown")
        heading = meta.get("section_heading") or meta.get("primary_heading") or meta.get("breadcrumbs") or ""
        prefix = f"[{idx}] {source}"
        if heading:
            prefix += f" | {heading}"
        text = _get_chunk_text(meta)
        parts.append(f"{prefix}\n{text}")
    return "\n\n---\n\n".join(parts)


def _normalize_chunk_text(text: str) -> str:
    normalized = (text or "").replace("\t", " ").replace("\u00a0", " ")
    normalized = re.sub(r"[ ]{2,}", " ", normalized)
    normalized = re.sub(r"\.{3,}", " ", normalized)
    lines: List[str] = []
    for raw in normalized.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "|" in line:
            cells = [cell.strip() for cell in line.split("|")]
            if any(cell for cell in cells if cell):
                trimmed = [cell for cell in cells if cell]
                if trimmed:
                    line = "| " + " | ".join(trimmed) + " |"
        lines.append(line)
    return "\n".join(lines)


def _get_chunk_text(meta: Dict[str, Any]) -> str:
    cached = meta.get("_normalized_text")
    if isinstance(cached, str):
        return cached
    text = meta.get("text") or ""
    normalized = _normalize_chunk_text(text)
    meta["_normalized_text"] = normalized
    return normalized


def _is_table_line(line: str) -> bool:
    if "|" not in line:
        return False
    segments = [segment.strip() for segment in line.split("|")]
    return sum(1 for segment in segments if segment) >= 2


def _format_markdown_table(lines: List[str]) -> str:
    rows = []
    max_cols = 0
    for raw in lines:
        if not _is_table_line(raw):
            continue
        parts = [part.strip() for part in raw.split("|") if part.strip()]
        if not parts:
            continue
        rows.append(parts)
        max_cols = max(max_cols, len(parts))
    if len(rows) < 2:
        return "\n".join(lines)
    normalized_rows: List[List[str]] = []
    for row in rows:
        padded = row + ["" for _ in range(max_cols - len(row))]
        normalized_rows.append(padded)
    header = normalized_rows[0]
    body = normalized_rows[1:]
    sep = "| " + " | ".join("---" for _ in header) + " |"
    output = ["| " + " | ".join(header) + " |", sep]
    for row in body:
        output.append("| " + " | ".join(row) + " |")
    return "\n".join(output)


def _render_tables_in_text(text: str) -> str:
    lines = text.splitlines()
    out: List[str] = []
    table_buffer: List[str] = []

    def flush_table():
        if table_buffer:
            out.append(_format_markdown_table(table_buffer))
            table_buffer.clear()

    for line in lines:
        stripped = line.rstrip()
        if _is_table_line(stripped):
            table_buffer.append(stripped)
        else:
            flush_table()
            out.append(stripped)
    flush_table()
    return "\n".join(part for part in out if part.strip())


# ----- [S5C] Course scoping & matching heuristics ----------------------------
# Gắn câu hỏi vào doc_id cụ thể (tên học phần, mã môn học) để filter chunk và
# bổ sung chunk bắt buộc (giảng viên, tín chỉ). Khi muốn điều chỉnh cách dò
# học phần, tìm tới “[S5C]”.
def _slugify_match_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return ascii_text


def _normalize_alias_text(value: str) -> str:
    value = value.strip(" \t-•")
    if ":" in value:
        value = value.split(":", 1)[1]
    elif "-" in value:
        parts = value.split("-", 1)
        if len(parts[0].split()) <= 5:
            value = parts[1]
    value = value.strip(" \t:-•")
    value = value.split("|", 1)[0].strip()
    value = value.rstrip(".")
    return value.strip()


def _extract_course_alias_candidates(text: str) -> List[str]:
    aliases: List[str] = []
    for raw_line in text.splitlines():
        cleaned = raw_line.strip()
        if not cleaned:
            continue
        normalized = _normalize_query(cleaned)
        if not normalized:
            continue
        has_marker = False
        if "english" in normalized and "course" in normalized:
            has_marker = True
        elif "ten hoc phan" in normalized and "tieng anh" in normalized:
            has_marker = True
        if not has_marker:
            continue
        alias = _normalize_alias_text(cleaned)
        if alias:
            aliases.append(alias)
    return aliases


def _extract_course_aliases(meta: Dict[str, Any]) -> List[str]:
    doc_id = meta.get("doc_id")
    if not doc_id:
        return []
    cache = st.session_state.setdefault("course_alias_cache", {})
    if doc_id in cache:
        return cache[doc_id]
    data = _load_course_document(doc_id)
    aliases: List[str] = []
    if data:
        text = data.get("full_text") or ""
        if text:
            aliases.extend(_extract_course_alias_candidates(text))
        for chunk in data.get("chunks", []):
            chunk_text = chunk.get("text")
            if not chunk_text:
                continue
            normalized = _normalize_query(chunk_text)
            if "ten hoc phan" not in normalized and "course" not in normalized:
                continue
            aliases.extend(_extract_course_alias_candidates(chunk_text))
    seen_aliases: List[str] = []
    seen_norms: Set[str] = set()
    for alias in aliases:
        normalized = alias.lower()
        if normalized in seen_norms:
            continue
        seen_norms.add(normalized)
        seen_aliases.append(alias)
    cache[doc_id] = seen_aliases
    return seen_aliases


def _extract_possible_course_slugs(meta: Dict[str, Any]) -> List[str]:
    slugs: List[str] = []
    course_name = meta.get("course_name")
    if course_name:
        slug_value = _slugify_match_text(course_name)
        if slug_value:
            slugs.append(slug_value)
    source_filename = meta.get("source_filename")
    if source_filename:
        stem = Path(source_filename).stem
        prefix = stem.split("_")[0]
        slug_value = _slugify_match_text(prefix)
        if slug_value:
            slugs.append(slug_value)
    for alias in _extract_course_aliases(meta):
        slug_value = _slugify_match_text(alias)
        if slug_value:
            slugs.append(slug_value)
    deduped: List[str] = []
    seen: Set[str] = set()
    for slug in slugs:
        if slug in seen:
            continue
        deduped.append(slug)
        seen.add(slug)
    return deduped


def _get_course_registry() -> Dict[str, Dict[str, Any]]:
    cache = st.session_state.setdefault("course_registry_cache", {})
    key = safe_model_dir(st.session_state.embed_model)
    if key in cache:
        return cache[key]

    resources = ensure_backend_index_cache(st.session_state.embed_model)
    registry: Dict[str, Dict[str, Any]] = {}
    for meta in resources.get("metas", []):
        doc_id = meta.get("doc_id")
        if not doc_id:
            continue
        entry = registry.setdefault(
            doc_id,
            {
                "course_name": meta.get("course_name"),
                "course_code": meta.get("course_code"),
                "slugs": set(),
            },
        )
        entry["slugs"].update(_extract_possible_course_slugs(meta))
    cache[key] = registry
    return registry


def _identify_target_doc_ids(question: str) -> List[str]:
    registry = _get_course_registry()
    question_slug = _slugify_match_text(question)
    question_upper = question.upper()
    code_matches: List[str] = []
    name_matches: List[str] = []

    for doc_id, entry in registry.items():
        course_code = (entry.get("course_code") or "").upper()
        if course_code and course_code in question_upper:
            code_matches.append(doc_id)
            continue

        for slug in entry.get("slugs", []):
            if slug and slug in question_slug:
                name_matches.append(doc_id)
                break

    if code_matches:
        return code_matches
    if name_matches:
        return name_matches
    return []


def _course_matches_question(
    question: str,
    meta: Dict[str, Any],
    target_doc_ids: Optional[List[str]] = None,
) -> bool:
    if target_doc_ids:
        doc_id = meta.get("doc_id")
        if doc_id and doc_id in target_doc_ids:
            return True
    normalized_question = _slugify_match_text(question)
    if not normalized_question:
        return False

    course_code = (meta.get("course_code") or "").upper()
    if course_code and course_code in question.upper():
        return True

    course_name = meta.get("course_name") or ""
    if course_name:
        course_slug = _slugify_match_text(course_name)
        if course_slug and course_slug in normalized_question:
            return True
    return False


def _search_course_chunks_from_index(
    question: str,
    limit: int,
    target_doc_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    try:
        resources = ensure_backend_index_cache(st.session_state.embed_model)
    except Exception:
        return []

    metas = resources.get("metas") or []
    hits: List[Dict[str, Any]] = []
    for meta in metas:
        if not _course_matches_question(question, meta, target_doc_ids):
            continue
        hits.append({"score": 0.0, "meta": meta})
        if len(hits) >= limit:
            break
    return hits


def _filter_chunks_by_course(
    question: str,
    chunks: List[Dict[str, Any]],
    target_doc_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    matched = [
        chunk for chunk in chunks if _course_matches_question(question, chunk["meta"], target_doc_ids)
    ]
    if matched:
        return matched
    fallback = _search_course_chunks_from_index(
        question,
        st.session_state.retriever_k,
        target_doc_ids,
    )
    return fallback or chunks


# ----- [S5D] Prompt/LLM wiring & small talk guardrails -----------------------
# Quản lý client ChatOpenAI/LocalAI, chuẩn hóa text người dùng và các phản hồi
# small-talk để trước khi truy hồi tập trung vào câu hỏi hợp lệ.
def _build_chat_llm(chat_model: str) -> ChatOpenAI:
    """Return a ChatOpenAI-compatible client for the LocalAI runtime."""
    temperature = 0
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
                "Luôn trả lời dựa 100% vào phần Context kèm theo, không suy luận chung chung. "
                "Nếu Context có bảng hoặc danh sách (ví dụ mục Giảng viên, kế hoạch dạy học), hãy trích xuất rõ ràng "
                "các mục trong bảng theo thứ tự xuất hiện và nêu đủ họ tên, email, ghi chú nếu có. "
                "Khi câu hỏi dạng 'Ai/những ai/giảng viên nào', chỉ liệt kê những người được Context nhắc tới, không mô tả lý thuyết. "
                "Mỗi khi trích dẫn, ghi chú dạng [số] tương ứng với mục trong Context. "
                "Nếu Context không có thông tin phù hợp, hãy nói rõ và đề nghị người dùng cung cấp chi tiết hơn, "
                "không tự bịa ra dữ kiện. Trả lời rõ ràng và đầy đủ theo Context, không cần giới hạn số câu cố định.",
            ),
            (
                "human",
                "Question:\n{question}\n\nContext:\n{context}\n\n"
                "Answer using the same language as the Question.",
            ),
        ]
    )

    llm = _build_chat_llm(chat_model)
    messages = prompt.format_messages(question=question, context=context)
    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)


# ----- [S5E] Domain heuristics (giảng viên, tín chỉ, CLO, khoa) --------------
# Bộ constants + parser đọc bảng/đoạn text để nhận biết từ khóa đặc thù đề
# cương. Các hàm kế tiếp xử lý nhận dạng giảng viên, tín chỉ, giờ học, CLO,
# khoa viện để khi không cần LLM vẫn trả lời chính xác.
INSTRUCTOR_KEYWORDS = {
    "giang vien",
    "giao vien",
    "giang day",
    "giao vien phu trach",
    "teacher",
    "instructor",
    "faculty",
    "lecturer",
    "lecturers",
    "full name",
}
CREDIT_KEYWORDS = {
    "tin chi",
    "tinchl",  # safeguard for missing accents
    "credits",
    "credit hours",
    "so tin chi",
    "credit",
}
CLASS_HOUR_KEYWORDS = {
    "so gio tren lop",
    "so gio ly thuyet",
    "gio tren lop",
    "gio ly thuyet",
    "tin chi ly thuyet",
    "thoi luong tren lop",
    "class hours",
    "lecture hours",
    "contact hours",
    "in-class hours",
}
SELF_STUDY_KEYWORDS = {
    "so gio tu hoc",
    "gio tu hoc",
    "tu hoc",
    "self study",
    "independent study",
    "self-learning hours",
}
@dataclass(frozen=True)
class SectionFocus:
    label: str
    keywords: Tuple[str, ...]
    slug_prefixes: Tuple[str, ...] = ()
    min_segments: int = 2
    exclude_keys: Tuple[str, ...] = ()


def _focus(
    label: str,
    keywords: Sequence[str],
    *,
    slug_prefixes: Optional[Sequence[str]] = None,
    min_segments: int = 2,
    exclude_keys: Optional[Sequence[str]] = None,
) -> SectionFocus:
    return SectionFocus(
        label=label,
        keywords=tuple(keywords),
        slug_prefixes=tuple(slug_prefixes or ()),
        min_segments=min_segments,
        exclude_keys=tuple(exclude_keys or ()),
    )


SECTION_FOCUS_CONFIG: Dict[str, SectionFocus] = {
    "general": _focus(
        "Thông tin chung",
        (
            "thong tin chung",
            "general information",
            "ten hoc phan",
            "ma hoc phan",
            "course information",
            "course code",
            "course name",
            "trinh do dao tao",
        ),
        slug_prefixes=("1-thong-tin-chung", "thong-tin-chung", "general-information"),
    ),
    "description": _focus(
        "Mô tả học phần",
        ("mo ta", "course description", "overview", "tong quan", "gioi thieu hoc phan"),
        slug_prefixes=("3-mo-ta-hoc-phan-course-descriptions", "mo-ta-hoc-phan-course-descriptions", "mo-ta-hoc-phan"),
        min_segments=3,
        exclude_keys=("goals",),
    ),
    "goals": _focus(
        "Mục tiêu học phần",
        ("muc tieu", "course goal", "course goals", "muc tieu hoc phan", "course objective", "muc tieu hoc tap"),
        slug_prefixes=("5-muc-tieu-hoc-phan-course-goals", "muc-tieu-hoc-phan-course-goals", "muc-tieu-hoc-phan"),
    ),
    "outcomes": _focus(
        "Chuẩn đầu ra học phần",
        ("chuan dau ra", "learning outcome", "clo", "muc tieu dau ra", "ket qua hoc tap"),
        slug_prefixes=("bang-2-chuan-dau-ra-hoc-phan-clo", "chuan-dau-ra-hoc-phan", "course-learning-outcome"),
    ),
    "resources": _focus(
        "Tài liệu học tập",
        ("tai lieu", "tai lieu tham khao", "giao trinh", "reference book", "learning resource", "tai lieu hoc tap"),
        slug_prefixes=("4-tai-lieu-hoc-tap", "tai-lieu-hoc-tap", "tai-lieu-tham-khao", "leaning-resources"),
    ),
    "assessment": _focus(
        "Đánh giá học phần",
        ("danh gia", "co cau diem", "course assessment", "evaluation", "assessment method", "bai danh gia"),
        slug_prefixes=("7-danh-gia-hoc-phan", "danh-gia-hoc-phan", "co-cau-diem-thanh-phan", "course-assessment"),
    ),
    "lesson_plan": _focus(
        "Kế hoạch dạy học",
        ("ke hoach day hoc", "lesson plan", "ke hoach giang day", "noi dung day hoc", "lecture plan"),
        slug_prefixes=("8-ke-hoach-day-hoc", "ke-hoach-day-hoc", "ke-hoach-va-noi-dung-day-hoc", "lesson-plan"),
    ),
    "requirements": _focus(
        "Quy định học phần",
        ("quy dinh", "course requirement", "yeu cau hoc phan", "class rule", "expectation"),
        slug_prefixes=(
            "10-quy-dinh-cua-hoc-phan",
            "quy-dinh-cua-hoc-phan",
            "course-requirements",
            "course-requirements-and-expectation",
        ),
        min_segments=1,
    ),
    "outcome_assessment": _focus(
        "Đánh giá chuẩn đầu ra",
        ("danh gia chuan dau ra", "clo assessment", "learning outcomes assessment"),
        slug_prefixes=("9-danh-gia-chuan-dau-ra-hoc-phan", "course-leaning-outcomes-assessment"),
        min_segments=1,
    ),
    "exam_matrix": _focus(
        "Ma trận đề thi",
        ("ma tran de thi", "exam matrix", "phu luc 1", "so cau hoi thi"),
        slug_prefixes=("phu-luc-1-ma-tran-de-thi", "ma-tran-de-thi", "exam-matrix"),
        min_segments=1,
    ),
    "rubrics": _focus(
        "Rubric đánh giá",
        ("rubric", "phu luc 2", "tieu chi danh gia", "rubric danh gia"),
        slug_prefixes=("phu-luc-2", "rubric", "rubrics"),
        min_segments=1,
    ),
}


def _normalize_segment_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def _segment_mentions_focus(text: str, focus_key: str) -> bool:
    config = SECTION_FOCUS_CONFIG.get(focus_key)
    if not config:
        return False
    normalized = _normalize_query(text)
    return any(keyword in normalized for keyword in config.keywords)


def _filter_section_segments(segments: List[str], config: SectionFocus) -> List[str]:
    filtered: List[str] = []
    seen: Set[str] = set()
    for segment in segments:
        trimmed = segment.strip()
        if not trimmed:
            continue
        if config.exclude_keys and any(_segment_mentions_focus(trimmed, key) for key in config.exclude_keys):
            continue
        norm = _normalize_segment_key(trimmed)
        if norm in seen:
            continue
        seen.add(norm)
        filtered.append(trimmed)
    return filtered
DEPARTMENT_SECTION_SLUGS = [
    "2-khoa-vien-quan-ly-va-giang-vien-giang-day",
    "khoa-vien-quan-ly-va-giang-vien-giang-day",
]
INSTRUCTOR_SECTION_SLUGS = [
    "2-khoa-vien-quan-ly-va-giang-vien-giang-day",
    "khoa-vien-quan-ly-va-giang-vien-giang-day",
    "giang-vien-giang-day",
    "giang-vien",
    "lecturers-information",
]
DEPARTMENT_KEYWORDS = {
    "khoa",
    "viện",
    "vien",
    "phu trach",
    "phụ trách",
    "khoa quan ly",
    "department",
    "faculty",
}


def _question_targets_instructors(question: str) -> bool:
    normalized = _normalize_query(question)
    return any(keyword in normalized for keyword in INSTRUCTOR_KEYWORDS)


def _iter_context_entries(context: str) -> List[Tuple[Optional[str], str]]:
    entries: List[Tuple[Optional[str], str]] = []
    for block in context.split("\n\n---\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if not lines:
            continue
        header = lines[0]
        match = re.match(r"\[(\d+)\]", header)
        chunk_id = match.group(1) if match else None
        body = "\n".join(lines[1:]).strip()
        if not body:
            continue
        entries.append((chunk_id, body))
    return entries


def _collect_instructors_from_text(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen: set[Tuple[str, str]] = set()

    table_rows = _extract_instructor_rows(text)
    if table_rows:
        collectors = (lambda t: table_rows, )
    else:
        collectors = (_extract_instructor_key_values,)

    for collector in collectors:
        for row in collector(text):
            name = (row.get("name") or "").strip()
            email = (row.get("email") or "").strip()
            if not _is_valid_instructor_name(name):
                continue
            key = (name, email)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"name": name, "email": email})

    inline_patterns = [
        re.compile(r"(?P<name>[A-Za-zÀ-ỹĐđ\.'`\-\s]{3,})\s*\((?P<email>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\)"),
        re.compile(
            r"(?P<name>[A-Za-zÀ-ỹĐđ\.'`\-\s]{3,})\s*(?:Email|E-mail)\s*[:：]\s*(?P<email>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
            re.IGNORECASE,
        ),
    ]

    for pattern in inline_patterns:
        for match in pattern.finditer(text):
            name = match.group("name").strip()
            email = match.group("email").strip()
            if not _is_valid_instructor_name(name):
                continue
            key = (name, email)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"name": name, "email": email})

    return rows


def _extract_instructor_names_from_text(text: str) -> List[str]:
    names: List[str] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header_seen = False
    for line in lines:
        normalized = _normalize_query(line)
        if not header_seen:
            if "ho va ten" in normalized and "email" in normalized:
                header_seen = True
            continue
        if "|" not in line:
            if names:
                break
            continue
        parts = [part.strip() for part in line.split("|") if part.strip()]
        if len(parts) < 2:
            continue
        candidate = parts[1]
        if _is_valid_instructor_name(candidate):
            names.append(candidate)
    return names


def _collect_instructors_from_doc(doc_id: str) -> List[str]:
    texts = _collect_section_texts(doc_id, INSTRUCTOR_SECTION_SLUGS)
    seen: set[str] = set()
    names: List[str] = []
    if not texts:
        data = _load_course_document(doc_id)
        if data:
            texts = [chunk.get("text") or "" for chunk in data.get("chunks", [])]
    for block in texts:
        for name in _extract_instructor_names_from_text(block):
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
    return names


def _looks_like_name(value: str) -> bool:
    if not value:
        return False
    normalized = value.lower()
    prefixes = ("ts", "ths", "pgs", "gs", "dr", "prof")
    if any(prefix + "." in normalized for prefix in prefixes):
        return True
    tokens = [token for token in value.split() if token]
    capitalized = sum(1 for token in tokens if token[0].isalpha() and token[0].isupper())
    return capitalized >= 2


def _is_valid_instructor_name(name: str) -> bool:
    if not name:
        return False
    stripped = name.strip()
    if len(stripped) < 3 or len(stripped) > 80:
        return False
    normalized = _normalize_query(stripped)
    banned_keywords = {
        "khoa",
        "vien",
        "quan ly",
        "giang vien giang day",
        "giang vien giang day hoc phan",
        "course",
        "description",
        "lesson",
        "giang vien",
        "dia chi",
        "bang",
        "bảng",
        "phu trach khoa",
        "phụ trách khoa",
        "ten hoc phan",
        "ma hoc phan",
        "so tin chi",
        "trinh do",
        "noi dung",
        "cau truc",
        "bao ve",
        "hinh thuc",
        "ma tran",
        "tai lieu",
        "ket qua",
    }
    if any(keyword in normalized for keyword in banned_keywords):
        return False
    if "|" in stripped or stripped.endswith(":"):
        return False
    if re.search(r"\d", stripped):
        return False
    if not _looks_like_name(stripped):
        return False
    if any(char in stripped for char in [",", ";", ":", "(", ")", "[", "]"]):
        return False
    if len(stripped.split()) > 6:
        return False
    return True


def _extract_instructor_rows(text: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header_seen = False
    for line in lines:
        if not header_seen:
            normalized = _normalize_query(line)
            if "ho va ten" in normalized and "email" in normalized:
                header_seen = True
            continue
        if "|" not in line:
            if rows:
                break
            continue
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) < 3:
            continue
        stt = cells[0].strip(" .-")
        name = cells[1].strip()
        email = cells[2].strip()
        if not name:
            continue
        rows.append({"stt": stt, "name": name, "email": email})
    return rows


def _extract_instructor_key_values(text: str) -> List[Dict[str, str]]:
    markers = (
        "Full name",
        "Lecturer",
        "Giảng viên",
        "Giang vien",
        "Họ và tên",
        "Ho va ten",
        "Email",
    )
    normalized = text
    for marker in markers:
        normalized = re.sub(
            rf"\s*({re.escape(marker)})\s*:",
            r"\n\1:",
            normalized,
            flags=re.IGNORECASE,
        )
    entries: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    for raw in normalized.splitlines():
        line = raw.strip(" -\t")
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            continue
        lowered = _normalize_query(key)
        if lowered in {"full name", "lecturer", "giang vien", "ho va ten", "teacher", "instructor"}:
            if current.get("name"):
                entries.append(current)
                current = {}
            current["name"] = value
        elif lowered == "email":
            current["email"] = value
        elif lowered in {"phone number", "title", "address"}:
            continue
    if current.get("name"):
        entries.append(current)
    return entries


def _extract_instructor_list_entries(text: str) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        email_match = re.search(r"email[:\s]+([^\s,]+)", line, flags=re.IGNORECASE)
        email = email_match.group(1).strip() if email_match else ""
        cleaned = re.sub(r"\.{2,}", " ", line)
        cleaned = re.sub(r"tel[:\s].*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.split("Email", 1)[0]
        cleaned = re.sub(r"^\s*\d+[\.)]?\s*", "", cleaned)
        cleaned = cleaned.strip(" -:")
        if not cleaned:
            continue
        lowered = _normalize_query(cleaned)
        if lowered.startswith("lecturer"):
            continue
        entries.append({"name": cleaned, "email": email})
    return entries


def _chunk_contains_instructor_info(meta: Dict[str, Any]) -> bool:
    text = _get_chunk_text(meta)
    if not text:
        return False

    heading_parts = [
        meta.get("section_heading"),
        meta.get("primary_heading"),
        " > ".join(meta.get("heading_path") or []),
    ]
    normalized_heading = _normalize_query(" ".join(part for part in heading_parts if part))
    normalized_text = _normalize_query(text)

    heading_has_keyword = any(keyword in normalized_heading for keyword in INSTRUCTOR_KEYWORDS)
    text_has_keyword = any(keyword in normalized_text for keyword in INSTRUCTOR_KEYWORDS)
    if not (heading_has_keyword or text_has_keyword):
        return False

    return bool(
        _extract_instructor_rows(text)
        or _extract_instructor_key_values(text)
        or _extract_instructor_list_entries(text)
        or _collect_instructors_from_text(text)
    )


def _extract_course_codes(question: str) -> List[str]:
    pattern = re.compile(r"[A-Z]{2,}[A-Z0-9]*\d{2,}[A-Z0-9]*")
    return pattern.findall(question.upper())


def _search_instructor_chunks_from_index(
    question: str,
    target_doc_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    try:
        resources = ensure_backend_index_cache(st.session_state.embed_model)
    except Exception:
        return []

    metas = resources.get("metas") or []
    if not metas:
        return []

    hits: List[Dict[str, Any]] = []
    for meta in metas:
        if not _chunk_contains_instructor_info(meta):
            continue

        if not _course_matches_question(question, meta, target_doc_ids):
            continue

        hits.append({"score": 0.0, "meta": meta})
        if len(hits) >= 2:
            break
    return hits


def _maybe_expand_instructor_chunks(
    question: str,
    chunks: List[Dict[str, Any]],
    target_doc_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if not _question_targets_instructors(question):
        return chunks

    relevant = [
        item
        for item in chunks
        if _chunk_contains_instructor_info(item["meta"])
        and _course_matches_question(question, item["meta"], target_doc_ids)
    ]
    if relevant:
        return relevant

    extras = _search_instructor_chunks_from_index(question, target_doc_ids)
    return extras or chunks


def _ensure_attribute_chunks(
    question: str,
    chunks: List[Dict[str, Any]],
    target_doc_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    need_credit = _question_targets_credits(question) and not any(
        _extract_credit_value(_get_chunk_text(chunk["meta"])) for chunk in chunks
    )
    need_hours = _question_targets_class_hours(question) and not any(
        _extract_class_hours_value(_get_chunk_text(chunk["meta"])) for chunk in chunks
    )
    need_self_study = _question_targets_self_study_hours(question) and not any(
        _extract_self_study_hours_value(_get_chunk_text(chunk["meta"])) for chunk in chunks
    )
    focus_matches = _match_section_focus(question)
    section_needs = {key: config.min_segments for key, config in focus_matches.items()}

    if not (need_credit or need_hours or need_self_study or section_needs):
        return chunks

    existing_ids = {chunk["meta"].get("chunk_id") for chunk in chunks}
    extras: List[Dict[str, Any]] = []
    search_limit = max(st.session_state.retriever_k * 4, 20)
    candidates = _search_course_chunks_from_index(
        question,
        search_limit,
        target_doc_ids,
    )
    for candidate in candidates:
        chunk_id = candidate["meta"].get("chunk_id")
        if chunk_id in existing_ids:
            continue
        text = _get_chunk_text(candidate["meta"])
        added = False
        if need_credit and _extract_credit_value(text):
            extras.append(candidate)
            need_credit = False
            added = True
        if not added and need_hours and _extract_class_hours_value(text):
            extras.append(candidate)
            need_hours = False
            added = True
        if not added and need_self_study and _extract_self_study_hours_value(text):
            extras.append(candidate)
            need_self_study = False
            added = True
        if not added and section_needs:
            section_key = _detect_section_key(candidate["meta"], text, focus_matches)
            if section_key and section_needs.get(section_key, 0) > 0:
                extras.append(candidate)
                section_needs[section_key] -= 1
                added = True
        if added:
            existing_ids.add(chunk_id)
        if not (need_credit or need_hours or need_self_study or any(value > 0 for value in section_needs.values())):
            break

    return chunks + extras if extras else chunks


def _should_answer_in_english(question: str) -> bool:
    normalized = question.strip()
    if not normalized:
        return False
    ascii_chars = sum(1 for ch in normalized if ord(ch) < 128)
    ratio = ascii_chars / max(1, len(normalized))
    if ratio < 0.85:
        return False
    lowered = normalized.lower()
    return any(word in lowered for word in ("who", "lecturer", "teacher"))


def _question_targets_credits(question: str) -> bool:
    normalized = _normalize_query(question)
    return any(keyword in normalized for keyword in CREDIT_KEYWORDS)


def _question_targets_class_hours(question: str) -> bool:
    normalized = _normalize_query(question)
    return any(keyword in normalized for keyword in CLASS_HOUR_KEYWORDS)


def _question_targets_self_study_hours(question: str) -> bool:
    normalized = _normalize_query(question)
    return any(keyword in normalized for keyword in SELF_STUDY_KEYWORDS)


def _question_targets_department(question: str) -> bool:
    normalized = _normalize_query(question)
    return any(keyword in normalized for keyword in DEPARTMENT_KEYWORDS)


def _match_section_focus(question: str) -> Dict[str, SectionFocus]:
    normalized = _normalize_query(question)
    matches: Dict[str, SectionFocus] = {}
    for key, config in SECTION_FOCUS_CONFIG.items():
        if any(keyword in normalized for keyword in config.keywords):
            matches[key] = config
    return matches


def _section_focus_matches(heading: str, text: str, config: SectionFocus) -> bool:
    slug = _slugify_identifier(heading or "")
    normalized_heading = _normalize_query(heading)

    if config.slug_prefixes:
        if slug and any(slug.startswith(prefix) for prefix in config.slug_prefixes):
            return True
        return any(keyword in normalized_heading for keyword in config.keywords)

    normalized_text = _normalize_query(f"{heading} {text}")
    return any(keyword in normalized_text for keyword in config.keywords)


def _detect_section_key(meta: Dict[str, Any], text: str, focus_matches: Dict[str, SectionFocus]) -> Optional[str]:
    heading_raw = meta.get("section_heading") or meta.get("primary_heading") or meta.get("breadcrumbs") or ""
    for key, config in focus_matches.items():
        if _section_focus_matches(heading_raw, text, config):
            return key
    return None


CRE_CREDIT_PATTERNS = [
    re.compile(r"(số\s+tín\s+chỉ|so\s+tin\s+chi)\s*[:\-]?\s*([0-9]+)", re.IGNORECASE),
    re.compile(r"(credits?)\s*[:\-]?\s*([0-9]+)", re.IGNORECASE),
    re.compile(r"([0-9]+)\s*(tín\s*chỉ|tin\s*chi|credits?)", re.IGNORECASE),
]


def _extract_credit_value(text: str) -> Optional[str]:
    for pattern in CRE_CREDIT_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        for group in reversed(match.groups()):
            if group and group.strip().isdigit():
                return group.strip()
        digits = re.findall(r"[0-9]+", match.group(0))
        if digits:
            return digits[-1]
    text_lower = text.lower()
    for token in ("credit", "tín chỉ", "tin chi"):
        if token in text_lower:
            digits = re.findall(r"[0-9]+", text)
            if digits:
                return digits[-1]
    return None


HOUR_PATTERNS = [
    re.compile(r"(số\s+giờ\s+trên\s+lớp|số\s+giờ\s+lý\s+thuyết)\s*[:\-]?\s*([0-9]+)", re.IGNORECASE),
    re.compile(r"(class\s+hours|contact\s+hours|lecture\s+hours)\s*[:\-]?\s*([0-9]+)", re.IGNORECASE),
    re.compile(r"([0-9]+)\s*(giờ|gio)\s*(trên\s+lớp|lý\s+thuyết)", re.IGNORECASE),
]
SELF_STUDY_PATTERNS = [
    re.compile(r"(số\s+giờ\s+tự\s+học|so\s+gio\s+tu\s+hoc)\s*[:\-]?\s*([0-9]+)", re.IGNORECASE),
    re.compile(r"(self[\-\s]*study\s+hours|independent\s+study)\s*[:\-]?\s*([0-9]+)", re.IGNORECASE),
    re.compile(r"([0-9]+)\s*(giờ|gio)\s*(tự\s+học|tu\s+hoc)", re.IGNORECASE),
]


def _extract_class_hours_value(text: str) -> Optional[str]:
    for pattern in HOUR_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        for group in reversed(match.groups()):
            if group and group.strip().isdigit():
                return group.strip()
        digits = re.findall(r"[0-9]+", match.group(0))
        if digits:
            return digits[-1]
    return None


def _extract_self_study_hours_value(text: str) -> Optional[str]:
    for pattern in SELF_STUDY_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        for group in reversed(match.groups()):
            if group and group.strip().isdigit():
                return group.strip()
        digits = re.findall(r"[0-9]+", match.group(0))
        if digits:
            return digits[-1]
    return None


def _extract_self_study_hours_from_doc(doc_id: str) -> Optional[str]:
    data = _load_course_document(doc_id)
    if not data:
        return None
    for chunk in data.get("chunks", []):
        text = _normalize_chunk_text(chunk.get("text") or "")
        value = _extract_self_study_hours_value(text)
        if value:
            return value
    return None


def _answer_course_credits(question: str, chunks: List[Dict[str, Any]], _: str) -> Optional[str]:
    if not _question_targets_credits(question):
        return None
    english = _should_answer_in_english(question)
    label = "Credits" if english else "Số tín chỉ"
    for idx, chunk in enumerate(chunks, start=1):
        text = _get_chunk_text(chunk["meta"])
        value = _extract_credit_value(text)
        if value:
            return f"{label}: {value} [{idx}]"
    return None


def _answer_class_hours(question: str, chunks: List[Dict[str, Any]], _: str) -> Optional[str]:
    if not _question_targets_class_hours(question):
        return None
    english = _should_answer_in_english(question)
    label = "Class hours" if english else "Số giờ trên lớp"
    for idx, chunk in enumerate(chunks, start=1):
        text = _get_chunk_text(chunk["meta"])
        value = _extract_class_hours_value(text)
        if value:
            return f"{label}: {value} [{idx}]"
    return None


def _answer_self_study_hours(question: str, chunks: List[Dict[str, Any]], _: str) -> Optional[str]:
    if not _question_targets_self_study_hours(question):
        return None
    english = _should_answer_in_english(question)
    label = "Self-study hours" if english else "Số giờ tự học"
    for idx, chunk in enumerate(chunks, start=1):
        text = _get_chunk_text(chunk["meta"])
        value = _extract_self_study_hours_value(text)
        if value:
            return f"{label}: {value} [{idx}]"
    doc_ids = _identify_target_doc_ids(question)
    for doc_id in doc_ids:
        value = _extract_self_study_hours_from_doc(doc_id)
        if value:
            citation = ""
            for idx, chunk in enumerate(chunks, start=1):
                if chunk["meta"].get("doc_id") == doc_id:
                    citation = f" [{idx}]"
                    break
            return f"{label}: {value}{citation}"
    return None


def _answer_instructors(question: str, chunks: List[Dict[str, Any]], context: str) -> Optional[str]:
    if not _question_targets_instructors(question):
        return None
    english = _should_answer_in_english(question)

    target_doc_ids: List[str] = []
    for chunk in chunks:
        meta = chunk.get("meta") or {}
        doc_id = meta.get("doc_id")
        if doc_id and doc_id not in target_doc_ids:
            target_doc_ids.append(doc_id)

    names: List[str] = []
    seen: set[str] = set()
    for doc_id in target_doc_ids:
        for name in _collect_instructors_from_doc(doc_id):
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)

    if not names:
        for chunk in chunks:
            for name in _extract_instructor_names_from_text(_get_chunk_text(chunk.get("meta") or {})):
                if not name or name in seen:
                    continue
                seen.add(name)
                names.append(name)

    if not names:
        extras = _search_instructor_chunks_from_index(
            question,
            target_doc_ids or None,
        )
        for extra in extras:
            for name in _extract_instructor_names_from_text(_get_chunk_text(extra["meta"])):
                if not name or name in seen:
                    continue
                seen.add(name)
                names.append(name)

    if not names:
        return None

    label = "Lecturers" if english else "Giảng viên"
    body = "\n".join(f"- {entry}" for entry in names)
    return f"{label}:\n{body}"


def _answer_section_focus(question: str, chunks: List[Dict[str, Any]], _: str) -> Optional[str]:
    focus_matches = _match_section_focus(question)
    if not focus_matches:
        return None

    def _normalize_order(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    doc_ids: List[str] = _identify_target_doc_ids(question)
    doc_first_idx: Dict[str, int] = {}
    unordered_doc_ids: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.get("meta") or {}
        doc_id = meta.get("doc_id")
        if doc_id and doc_id not in doc_first_idx:
            doc_first_idx[doc_id] = idx
        if doc_id and doc_id not in unordered_doc_ids:
            unordered_doc_ids.append(doc_id)

    if not doc_ids:
        doc_ids = unordered_doc_ids

    sections_map: Dict[str, Dict[str, List[Tuple[int, int, int, str]]]] = defaultdict(lambda: defaultdict(list))
    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.get("meta") or {}
        text = _get_chunk_text(meta)
        matched_key = _detect_section_key(meta, text, focus_matches)
        if matched_key is None:
            continue
        heading = meta.get("section_heading") or meta.get("primary_heading") or meta.get("breadcrumbs") or matched_key
        section_id = _slugify_identifier(f"{meta.get('doc_id', '')}-{heading}")
        order_val = _normalize_order(meta.get("chunk_order"))
        subindex_val = _normalize_order(meta.get("chunk_subindex"))
        sections_map[matched_key][section_id].append((idx, order_val, subindex_val, text.strip()))

    sections: List[str] = []
    for key, config in focus_matches.items():
        combined_parts: List[str] = []
        citations: List[str] = []
        for doc_id in doc_ids:
            texts = _collect_section_texts(doc_id, config)
            if texts:
                combined_parts.extend(texts)
                if doc_id in doc_first_idx:
                    citations.append(f"[{doc_first_idx[doc_id]}]")
        section_groups = sections_map.get(key)
        if not combined_parts and section_groups:
            section_id, segments = max(
                section_groups.items(),
                key=lambda item: sum(len(seg[3]) for seg in item[1]),
            )
            sorted_segments = sorted(segments, key=lambda seg: (seg[1], seg[2], seg[0]))
            combined_parts = [seg[3] for seg in sorted_segments if seg[3]]
            citations = [f"[{seg[0]}]" for seg in sorted_segments]
        combined_parts = _filter_section_segments(combined_parts, config)
        combined = "\n".join(part for part in combined_parts if part).strip()
        if not combined:
            continue
        label = config.label or key.title()
        citation_text = " ".join(citations)
        sections.append(f"{label}: {combined} {citation_text}".strip())

    if not sections:
        return None
    return "\n\n".join(sections)


def _collect_department_sections(doc_id: str) -> List[str]:
    data = _load_course_document(doc_id)
    if not data:
        return []
    sections: List[str] = []
    seen: set[str] = set()
    for chunk in data.get("chunks", []):
        heading = chunk.get("primary_heading") or chunk.get("breadcrumbs") or ""
        slug = _slugify_identifier(heading)
        text = (chunk.get("text") or "").strip()
        normalized = _normalize_query(text)
        if not (
            any(slug.startswith(target) for target in DEPARTMENT_SECTION_SLUGS)
            or "phu trach khoa" in normalized
            or "phụ trách khoa" in text.lower()
        ):
            continue
        if text and text not in seen:
            sections.append(text)
            seen.add(text)
    return sections


def _collect_section_texts(doc_id: str, selector: Union[SectionFocus, Sequence[str]]) -> List[str]:
    data = _load_course_document(doc_id)
    if not data:
        return []
    texts: List[str] = []
    seen: set[str] = set()
    config = selector if isinstance(selector, SectionFocus) else None
    slug_targets = tuple(selector) if not isinstance(selector, SectionFocus) else ()
    for chunk in data.get("chunks", []):
        heading = chunk.get("primary_heading") or chunk.get("breadcrumbs") or ""
        raw_text = _normalize_chunk_text(chunk.get("text") or "")
        if config:
            if not _section_focus_matches(heading, raw_text, config):
                continue
        else:
            slug = _slugify_identifier(heading)
            if not any(slug.startswith(target) for target in slug_targets):
                continue
        if raw_text and raw_text not in seen:
            texts.append(raw_text)
            seen.add(raw_text)
    return texts


def _answer_department(question: str, chunks: List[Dict[str, Any]], _: str) -> Optional[str]:
    if not _question_targets_department(question):
        return None
    doc_ids: List[str] = []
    for chunk in chunks:
        meta = chunk.get("meta") or {}
        doc_id = meta.get("doc_id")
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)

    sections: List[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        meta = chunk.get("meta") or {}
        heading_slug = _slugify_identifier(
            meta.get("section_heading") or meta.get("primary_heading") or meta.get("breadcrumbs") or ""
        )
        if not any(heading_slug.startswith(target) for target in DEPARTMENT_SECTION_SLUGS):
            continue
        cleaned = _get_chunk_text(meta).strip()
        if cleaned and cleaned not in seen:
            sections.append(cleaned)
            seen.add(cleaned)

    for doc_id in doc_ids:
        for section_text in _collect_department_sections(doc_id):
            if section_text not in seen:
                sections.append(section_text)
                seen.add(section_text)

    if not sections:
        return None
    body = _render_tables_in_text("\n\n".join(sections))
    return f"Khoa/Viện phụ trách:\n{body}"


# ----- [S5F] Answer assembly & fallback responses ---------------------------
# Chọn chiến lược trả lời (heuristic hoặc LLM), trình bày citations và xây
# fallback deterministic nếu thiếu dữ kiện. Khi cần sửa logic trả lời, tìm "[S5F]".
@dataclass
class AnswerStrategy:
    name: str
    matcher: Callable[[str], bool]
    builder: Callable[[str, List[Dict[str, Any]], str], Optional[str]]


ANSWER_STRATEGIES: List[AnswerStrategy] = [
    AnswerStrategy("instructors", _question_targets_instructors, _answer_instructors),
    AnswerStrategy("department", _question_targets_department, _answer_department),
    AnswerStrategy("section_focus", lambda q: bool(_match_section_focus(q)), _answer_section_focus),
    AnswerStrategy("course_credits", _question_targets_credits, _answer_course_credits),
    AnswerStrategy("class_hours", _question_targets_class_hours, _answer_class_hours),
    AnswerStrategy("self_study_hours", _question_targets_self_study_hours, _answer_self_study_hours),
]


def _apply_answer_strategies(question: str, chunks: List[Dict[str, Any]], context: str) -> Optional[str]:
    for strategy in ANSWER_STRATEGIES:
        if not strategy.matcher(question):
            continue
        try:
            answer = strategy.builder(question, chunks, context)
        except Exception:
            answer = None
        if answer:
            return answer
    return None


def _build_no_info_response(question: str, chunks: List[Dict[str, Any]]) -> str:
    english = _should_answer_in_english(question)
    if english:
        base = "No matching fact was found in the syllabus sections I retrieved."
        hint = "Please double-check the course name/code or add the document that contains this info."
    else:
        base = "Tôi chưa thấy thông tin này trong các đoạn đã trích."
        hint = "Bạn hãy kiểm tra lại tên/mã học phần hoặc bổ sung tài liệu chứa thông tin đó."

    courses: List[str] = []
    for chunk in chunks:
        meta = chunk["meta"]
        course_name = meta.get("course_name")
        course_code = meta.get("course_code")
        if course_name or course_code:
            label = course_name or ""
            if course_code:
                label = f"{label} ({course_code})" if label else course_code
            if label and label not in courses:
                courses.append(label)

    if courses:
        course_line = ", ".join(courses[:3])
        detail = f" (Ngữ cảnh hiện có: {course_line})" if not english else f" (Context: {course_line})"
    else:
        detail = ""
    return f"{base}{detail} {hint}"


def _build_context_summary_answer(question: str, chunks: List[Dict[str, Any]]) -> str:
    """Provide a deterministic fallback answer by echoing key context chunks."""
    pieces: List[str] = []
    english = _should_answer_in_english(question)
    header = "Extracted context:" if english else "Thông tin trích từ tài liệu:"

    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.get("meta") or {}
        snippet = (_get_chunk_text(meta) or "").strip()
        if not snippet:
            continue
        heading = (
            meta.get("section_heading")
            or meta.get("primary_heading")
            or meta.get("breadcrumbs")
            or meta.get("course_name")
            or meta.get("doc_id")
            or "Chi tiết"
        )
        snippet = _render_tables_in_text(snippet)
        snippet = re.sub(r"\s+", " ", snippet).strip()
        max_len = 900
        if len(snippet) > max_len:
            snippet = snippet[:max_len].rstrip() + "..."
        pieces.append(f"[{idx}] {heading}\n{snippet}")
        if len(pieces) >= 2:
            break

    if not pieces:
        return ""
    return f"{header}\n\n" + "\n\n".join(pieces)


def _generate_answer_from_chunks(question: str, chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]], str]:
    """Run answer strategies/LLM with lexical context fallback."""
    chunks = _dedupe_chunks(chunks)
    context = format_backend_context(chunks)
    if not context.strip():
        lexical_chunks = _keyword_search_chunks(
            question,
            st.session_state.retriever_k,
        )
        if lexical_chunks:
            chunks = _dedupe_chunks(lexical_chunks)
            context = format_backend_context(chunks)
    if not context.strip():
        return _build_no_info_response(question, chunks), chunks, context

    answer = _apply_answer_strategies(question, chunks, context)
    connection_note: Optional[str] = None
    if not answer:
        try:
            answer = call_llm(st.session_state.chat_model, question, context)
        except Exception as exc:
            if _is_connection_issue(exc):
                connection_note = str(exc)
                answer = ""
            else:
                raise
        if not (answer or "").strip():
            answer = _build_context_summary_answer(question, chunks)
            if not answer:
                answer = _build_no_info_response(question, chunks)

    if connection_note and answer:
        english = _should_answer_in_english(question)
        note = (
            "Could not reach the language model, so the response is composed directly from the retrieved context."
            if english
            else "Không thể kết nối tới mô hình trả lời, nên tôi trích xuất trực tiếp từ ngữ cảnh hiện có."
        )
        answer = f"{answer}\n\n_{note}_"

    return answer, chunks, context


# -----------------------------------------------------------------------------#
# [S6] Streamlit helpers
# -----------------------------------------------------------------------------#
# Phụ trách theme, bố cục, widget điều khiển runtime và trải nghiệm chat giúp
# kết nối tầng truy hồi với người dùng. Nhìn “[S6]” để lần ra UI nào điều khiển
# embedding/pipeline/run-mode.


# ----- [S6A] Theme & session bootstrap --------------------------------------
# apply_material_theme áp dụng giao diện, init_session thiết lập session_state
# mặc định, đảm bảo việc reload UI/pipeline luôn có cùng cấu hình nền.


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

    if "history" not in st.session_state:
        st.session_state.history = []
    if "retriever_k" not in st.session_state:
        st.session_state.retriever_k = 4
    if "chat_model" not in st.session_state:
        st.session_state.chat_model = DEFAULT_CHAT_MODEL
    if "chunk_mode" not in st.session_state:
        st.session_state.chunk_mode = DEFAULT_CHUNK_MODE
    if "embedding_backend" not in st.session_state:
        st.session_state.embedding_backend = "tei"
    if "embed_model" not in st.session_state:
        st.session_state.embed_model = DEFAULT_EMBED_MODEL or list(TEI_MODELS.keys())[0]
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
    if "active_view" not in st.session_state:
        st.session_state.active_view = USER_VIEW
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False
    if "admin_selected_file" not in st.session_state:
        st.session_state.admin_selected_file = None
    if "admin_text_editor_value" not in st.session_state:
        st.session_state.admin_text_editor_value = ""
    if "admin_file_feedback" not in st.session_state:
        st.session_state.admin_file_feedback = None
    if "admin_editor_loaded_file" not in st.session_state:
        st.session_state.admin_editor_loaded_file = None

    if st.session_state.embedding_backend == "tei" and st.session_state.embed_model not in TEI_MODELS:
        st.session_state.embed_model = list(TEI_MODELS.keys())[0]
    if "backend_index_cache" not in st.session_state:
        st.session_state.backend_index_cache = {}



# ----- [S6B] Settings & sidebar widgets --------------------------------------
# Các hàm render_* ở phần này quản lý trải nghiệm sidebar: lựa chọn backend,
# điều khiển runtime Docker, upload/pipeline và action nhanh cho người dùng.
def render_settings_body():
    st.title("Settings")
    st.button(
        "Trang quản trị",
        use_container_width=True,
        key="sidebar_admin_nav_button",
        on_click=_set_active_view,
        args=(ADMIN_VIEW,),
    )
    st.session_state.embedding_backend = "tei"
    st.caption("Run mode: Local TEI embeddings with LocalAI chat runtime.")

    st.divider()

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

    render_local_runtime_controls()

def _trigger_streamlit_rerun() -> None:
    """Call the available Streamlit rerun API across versions."""
    rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun_fn is None:
        raise RuntimeError("Streamlit rerun function not available.")
    rerun_fn()


def _set_active_view(target: str) -> None:
    st.session_state.active_view = target
    if target != ADMIN_VIEW:
        st.session_state.admin_authenticated = False


def _logout_admin() -> None:
    st.session_state.admin_authenticated = False
    st.session_state.active_view = USER_VIEW


def _get_admin_password() -> str:
    value = os.getenv(ADMIN_PASSWORD_ENV_VAR, "")
    return value.strip() if value else ""


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


def _list_raw_document_paths() -> List[Path]:
    if not DATA_DIR.exists():
        return []
    files: List[Path] = []
    for path in DATA_DIR.rglob("*"):
        if path.is_file():
            files.append(path)
    files.sort()
    return files


def _format_filesize(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ["KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        size /= 1024
        if size < 1024:
            return f"{size:.1f} {unit}"
    return f"{size:.1f} PB"


def _read_text_file_with_fallback(path: Path) -> Tuple[str, str]:
    encodings = ("utf-8", "utf-16", "latin-1")
    last_error: Optional[Exception] = None
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding), encoding
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    raise RuntimeError(f"Không thể đọc {path.name}: {last_error}")


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


# ----- [S6C] Docker runtime monitor -----------------------------------------
# render_local_runtime_controls hiển thị trạng thái + nút điều khiển TEI/LocalAI
# và kích hoạt pipeline ingest nên mọi vấn đề Docker -> tìm “[S6C]”.
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


# ----- [S6D] Quick actions & pipeline controls -------------------------------
# render_sidebar_quick_actions gom các nút rebuild index, tải tài liệu, clear
# history... giúp vận hành nhanh không cần cuộn cả sidebar.
def render_sidebar_quick_actions():
    pipeline_running = st.session_state.get("pipeline_running", False)
    active_view = st.session_state.get("active_view", USER_VIEW)

    if active_view != ADMIN_VIEW:
        st.subheader("Conversation")
        if st.button(
            "Clear chat history",
            use_container_width=True,
            disabled=pipeline_running,
            key="sidebar_clear_history_button",
        ):
            st.session_state.history = []
            st.rerun()
        st.caption("Upload tài liệu và rebuild index nằm trong trang quản trị.")
        return

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


# ----- [S6E] View switching & admin tools ------------------------------------
# Phần này gom điều khiển chuyển Trang người dùng/Trang quản trị, đăng nhập
# admin, quản lý file gốc và layout trang quản trị. Tìm "[S6E]" để sửa UI admin.
def ensure_admin_access() -> bool:
    admin_password = _get_admin_password()
    already_authenticated = st.session_state.get("admin_authenticated", False)

    if not admin_password:
        st.subheader("Đăng nhập trang quản trị")
        st.warning("ADMIN_PASSWORD chưa được cấu hình. Cho phép truy cập tạm thời vào trang quản trị.")
        st.session_state.admin_authenticated = True
        return True

    if already_authenticated:
        return True

    st.subheader("Đăng nhập trang quản trị")
    with st.form("admin_login_form"):
        entered_password = st.text_input("Mật khẩu", type="password")
        submitted = st.form_submit_button("Đăng nhập")
        if submitted:
            if entered_password == admin_password:
                st.session_state.admin_authenticated = True
                st.success("Đăng nhập admin thành công.")
                _trigger_streamlit_rerun()
            else:
                st.error("Mật khẩu không đúng.")
    st.info("Nhập đúng mật khẩu để mở khóa trang quản trị.")
    return False


def render_admin_file_manager() -> None:
    st.subheader("Quản lý tài liệu gốc")
    all_files = _list_raw_document_paths()
    if not all_files:
        st.info("Chưa có tệp nào trong `backend/data/raw`. Hãy upload tài liệu trước.")
        return

    relative_labels: List[str] = []
    for path in all_files:
        try:
            label = str(path.relative_to(BACKEND_ROOT))
        except ValueError:
            label = str(path.relative_to(PROJECT_ROOT.parent))
        relative_labels.append(label)

    selected_label = st.session_state.get("admin_selected_file")
    default_index = 0
    if selected_label and selected_label in relative_labels:
        default_index = relative_labels.index(selected_label)

    selected_option = st.selectbox(
        "Chọn tệp để xem/sửa",
        options=relative_labels,
        index=default_index,
        key="admin_file_selectbox",
    )
    selected_idx = relative_labels.index(selected_option)
    selected_path = all_files[selected_idx]
    st.session_state.admin_selected_file = selected_option

    stat = selected_path.stat()
    size_label = _format_filesize(stat.st_size)
    modified_label = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Kích thước: {size_label} | Cập nhật lần cuối: {modified_label}")
    st.caption(f"Toàn đường dẫn: `{selected_path}`")

    download_bytes = selected_path.read_bytes()
    st.download_button(
        "Tải tệp",
        data=download_bytes,
        file_name=selected_path.name,
        mime="application/octet-stream",
        use_container_width=True,
        key=f"admin_download_{selected_option}",
    )

    editable_ext = selected_path.suffix.lower().lstrip(".")
    can_edit_inline = editable_ext in TEXT_EDITABLE_EXTS

    if st.session_state.get("admin_editor_loaded_file") != selected_option:
        if can_edit_inline:
            try:
                text_value, encoding = _read_text_file_with_fallback(selected_path)
            except Exception as exc:  # noqa: BLE001
                can_edit_inline = False
                st.warning(f"Không thể đọc nội dung dạng text: {exc}")
            else:
                st.session_state.admin_text_editor_value = text_value
                st.session_state.admin_editor_loaded_file = selected_option
                st.session_state.admin_text_editor_encoding = encoding
        else:
            st.session_state.admin_text_editor_value = ""
            st.session_state.admin_editor_loaded_file = selected_option

    if can_edit_inline:
        st.text_area(
            "Chỉnh sửa nội dung (lưu ý chỉ hỗ trợ tệp văn bản thuần)",
            key="admin_text_editor_value",
            height=320,
        )
        if st.button(
            "Lưu nội dung",
            key="admin_save_text_button",
            use_container_width=True,
        ):
            try:
                selected_path.write_text(st.session_state.admin_text_editor_value, encoding="utf-8")
                st.session_state.admin_file_feedback = ("success", f"Đã lưu {selected_option}.")
            except Exception as exc:  # noqa: BLE001
                st.session_state.admin_file_feedback = ("error", f"Lỗi khi lưu {selected_option}: {exc}")
            _trigger_streamlit_rerun()
    else:
        st.info(
            "Đây là tệp nhị phân (PDF/DOCX/XLSX...). Hãy tải xuống, chỉnh sửa bằng công cụ chuyên dụng và upload lại để thay thế."
        )

    replacement = st.file_uploader(
        "Upload để thay thế tệp hiện tại",
        type=None,
        key=f"admin_replace_uploader_{selected_option}",
    )
    if replacement is not None and st.button(
        "Ghi đè tệp bằng bản tải lên",
        key=f"admin_replace_button_{selected_option}",
        use_container_width=True,
    ):
        try:
            with open(selected_path, "wb") as target:
                target.write(replacement.getbuffer())
            st.session_state.admin_file_feedback = ("success", f"Đã cập nhật {selected_option} từ bản tải lên.")
            st.session_state.admin_editor_loaded_file = None
        except Exception as exc:  # noqa: BLE001
            st.session_state.admin_file_feedback = ("error", f"Không thể ghi đè {selected_option}: {exc}")
        _trigger_streamlit_rerun()

    feedback = st.session_state.get("admin_file_feedback")
    if feedback:
        status, message = feedback
        if status == "success":
            st.success(message)
        elif status == "error":
            st.error(message)
        else:
            st.info(message)
        st.session_state.admin_file_feedback = None


def render_admin_page() -> None:
    st.header("Trang quản trị")
    st.caption("Quản lý tài liệu, pipeline và thiết lập runtime.")
    st.button(
        "Đăng xuất admin",
        key="admin_logout_button",
        on_click=_logout_admin,
    )
    st.divider()
    render_admin_file_manager()
    st.divider()
    render_sidebar_quick_actions()


# ----- [S6F] User chat view & app entrypoint ---------------------------------
# Các hàm dưới quản lý cảnh báo runtime, lịch sử chat, hộp thoại hỏi đáp và
# entrypoint Streamlit. Tìm "[S6F]" khi cần điều chỉnh trải nghiệm người dùng.
def _render_user_runtime_notices(current_index_dir: Path) -> None:
    runtime_mode = st.session_state.tei_runtime_mode
    model_key = st.session_state.embed_model
    if not tei_backend_is_active(model_key, runtime_mode):
        st.info("The Docker-based TEI service is not running. Start it from the sidebar before continuing.")

    if not index_exists(current_index_dir):
        st.info("No backend index detected. Run the pipeline from the sidebar to build it from DOCX files.")


def _render_chat_history() -> None:
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                with st.expander("Sources"):
                    for idx, source in enumerate(turn["sources"], start=1):
                        st.markdown(f"**{idx}.** `{source['source']}` (p.{source.get('page', '?')})")
                        if source.get("snippet"):
                            st.caption(source["snippet"])


def _collect_user_question() -> Optional[str]:
    pending_question = st.session_state.pop("_pending_question", None)
    return pending_question or st.chat_input("Ask something about your documents...")


def _handle_user_question(current_index_dir: Path, user_question: str) -> None:
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
            chunks = retrieve_relevant_chunks(user_question)
            if not chunks:
                st.session_state.history.append({
                    "role": "assistant",
                    "content": "No relevant chunks found in the current index.",
                })
                st.rerun()

            answer, used_chunks, _ = _generate_answer_from_chunks(user_question, chunks)
            if not answer:
                st.session_state.history.append({
                    "role": "assistant",
                    "content": _build_no_info_response(user_question, used_chunks),
                })
                st.rerun()

            sources = []
            for item in used_chunks:
                meta = item["meta"]
                source = meta.get("source_filename") or meta.get("filename") or meta.get("doc_id", "unknown")
                snippet = _get_chunk_text(meta)
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


def render_user_view() -> None:
    current_index_dir = resolve_index_dir(st.session_state.embed_model)
    _render_user_runtime_notices(current_index_dir)
    _render_chat_history()
    user_question = _collect_user_question()
    if user_question:
        _handle_user_question(current_index_dir, user_question)

def _load_project_envs() -> None:
    dotenv_files = [
        Path(".env"),
        PROJECT_ROOT.parent / ".env",
        PROJECT_ROOT / ".env",
        BACKEND_ROOT / ".env",
    ]
    for env_path in dotenv_files:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)


def main():
    _load_project_envs()
    ensure_dirs()
    init_session()

    st.set_page_config(page_title="RAG over PDFs", page_icon=":books:", layout="wide")
    apply_material_theme()

    active_view = st.session_state.get("active_view", USER_VIEW)

    with st.sidebar:
        render_sidebar()

    if active_view == ADMIN_VIEW:
        if ensure_admin_access():
            render_admin_page()
        return

    st.header("Trang người dùng")
    st.divider()
    render_user_view()


def render_sidebar():
    active_view = st.session_state.get("active_view", USER_VIEW)
    if active_view == ADMIN_VIEW:
        st.header("Trang quản trị")
        st.caption("Các công cụ upload và pipeline hiển thị ở phần nội dung chính.")
        if st.session_state.get("admin_authenticated"):
            st.button(
                "Đăng xuất admin",
                use_container_width=True,
                key="sidebar_admin_logout_button",
                on_click=_logout_admin,
            )
        else:
            st.button(
                "Về trang người dùng",
                use_container_width=True,
                key="sidebar_back_to_user_button",
                on_click=_set_active_view,
                args=(USER_VIEW,),
            )
        return

    render_settings_body()
    st.divider()
    render_sidebar_quick_actions()


if __name__ == "__main__":
    main()


