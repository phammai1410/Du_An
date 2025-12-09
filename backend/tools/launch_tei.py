#!/usr/bin/env python3
"""
Script tiện ích dùng để khởi chạy các server Hugging Face Text Embeddings Inference (TEI) 
bằng Docker images và các mô hình đã được tải về cục bộ
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    from shlex import join as shlex_join
except ImportError:  # pragma: no cover - Python < 3.8 fallback
    def shlex_join(parts: List[str]) -> str:
        return " ".join(parts)

# Định nghĩa các hằng số cấu hình và đường dẫn
BACKEND_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_ROOT = BACKEND_ROOT / "local-llm" / "Embedding"
CONFIG_PATH = EMBEDDING_ROOT / "models.json"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_RUNTIME_KEY = "cpu"
CONTAINER_MODEL_PATH = "/data"
CONTAINER_PORT = 80
CONTAINER_NAME_PREFIX = "tei-"
COMPOSE_PROJECT_NAME = os.getenv("TEI_COMPOSE_PROJECT", "khoa_luan")
COMPOSE_SERVICE_NAME = os.getenv("TEI_COMPOSE_SERVICE", "tei-runtime")
COMPOSE_VERSION = os.getenv("TEI_COMPOSE_VERSION", "1.29.2")

# Định nghĩa lớp dữ liệu để lưu trữ thông tin về các chế độ runtime TEI
@dataclass(frozen=True)
class RuntimeSpec:
    """Container metadata for a TEI runtime mode."""

    label: str
    image: str
    requires_gpu: bool

# Định nghĩa các chế độ runtime TEI có sẵn
# Các chế độ này xác định Docker image và yêu cầu GPU hay không
# Người dùng có thể chọn chế độ runtime khi khởi chạy container
RUNTIME_SPECS: Dict[str, RuntimeSpec] = {
    "cpu": RuntimeSpec(
        label="CPU",
        image="ghcr.io/huggingface/text-embeddings-inference:cpu-1.8",
        requires_gpu=False,
    ),
    "turing": RuntimeSpec(
        label="Turing (T4 / RTX 2000 series)",
        image="ghcr.io/huggingface/text-embeddings-inference:turing-1.8",
        requires_gpu=True,
    ),
    "ampere_80": RuntimeSpec(
        label="Ampere 80 (A100 / A30)",
        image="ghcr.io/huggingface/text-embeddings-inference:1.8",
        requires_gpu=True,
    ),
    "ampere_86": RuntimeSpec(
        label="Ampere 86 (A10 / A40)",
        image="ghcr.io/huggingface/text-embeddings-inference:86-1.8",
        requires_gpu=True,
    ),
    "ada_lovelace": RuntimeSpec(
        label="Ada Lovelace (RTX 4000 series)",
        image="ghcr.io/huggingface/text-embeddings-inference:89-1.8",
        requires_gpu=True,
    ),
    "hopper": RuntimeSpec(
        label="Hopper (H100, experimental)",
        image="ghcr.io/huggingface/text-embeddings-inference:hopper-1.8",
        requires_gpu=True,
    ),
}

# Hướng dẫn cài đặt Docker nếu chưa có
# Người dùng cần cài đặt Docker để chạy các container TEI
DOCKER_INSTALL_HELP = textwrap.dedent(
    """\
    Docker is required to run Text Embeddings Inference with this helper.

    Windows:
      - Installer: https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe
      - Microsoft Store: https://apps.microsoft.com/detail/xp8cbj40xlbwkx

    macOS:
      - Apple silicon: https://desktop.docker.com/mac/main/arm64/Docker.dmg
      - Intel: https://desktop.docker.com/mac/main/amd64/Docker.dmg

    Linux:
      - Docker Engine: https://docs.docker.com/engine/install/
      - Docker Desktop: https://docs.docker.com/desktop/setup/install/linux/

    Install Docker, restart your terminal, and then re-run this command.
    """
)
# Hướng dẫn thiết lập môi trường GPU trên Linux
# Người dùng cần cài đặt NVIDIA Container Toolkit để sử dụng GPU trong container
LINUX_GPU_HELP = textwrap.dedent(
    """\
    GPU runtimes on Linux require the NVIDIA Container Toolkit:
    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

    After installing, verify the setup with:
        sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
    """
)


class LaunchError(RuntimeError):
    """Raised when the launch configuration is invalid."""

# hàm để tải cấu hình mô hình từ file JSON
# trả về từ điển chứa cấu hình của các mô hình
# ném lỗi LaunchError nếu file không tồn tại hoặc không hợp lệ
def load_models_config() -> Dict[str, Dict[str, object]]:
    if not CONFIG_PATH.exists():
        raise LaunchError(f"Config file not found: {CONFIG_PATH}")

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:  # pragma: no cover - safeguard only
        raise LaunchError(f"Invalid JSON in {CONFIG_PATH}: {exc}") from exc

    if not isinstance(data, dict):
        raise LaunchError(f"Config root must be an object in {CONFIG_PATH}")

    return data

# hàm để xác định đường dẫn mô hình từ cấu hình
#nếu đường dẫn không phải là tuyệt đối thì chuyển thành đường dẫn tuyệt đối
def resolve_model_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (CONFIG_PATH.parent / path).resolve()
    return path

# hàm để đảm bảo Docker đã được cài đặt và có thể sử dụng
# trả về đường dẫn đến lệnh Docker
# ném lỗi LaunchError nếu Docker không khả dụng hoặc có lỗi
def ensure_docker_available() -> str:
    docker_cmd = shutil.which("docker")
    if not docker_cmd:
        raise LaunchError(f"Docker command not found.\n\n{DOCKER_INSTALL_HELP}")

    try:
        result = subprocess.run(
            [docker_cmd, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise LaunchError(f"Failed to execute Docker CLI: {exc}\n\n{DOCKER_INSTALL_HELP}") from exc

    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip() or "Unknown error."
        raise LaunchError(f"Docker CLI returned an error:\n{details}\n\n{DOCKER_INSTALL_HELP}")

    return docker_cmd

# hàm để đảm bảo môi trường runtime GPU đã sẵn sàng trên Linux
def ensure_gpu_runtime_ready(docker_cmd: str) -> None:
    if platform.system().lower() != "linux":
        return

    try:
        info = subprocess.run(
            [docker_cmd, "info", "--format", "{{json .Runtimes}}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise LaunchError(
            f"Failed to query Docker runtimes: {exc}\n\n{LINUX_GPU_HELP}"
        ) from exc

    if info.returncode != 0:
        details = info.stderr.strip() or info.stdout.strip() or "unknown error."
        raise LaunchError(
            "Docker is not accessible or returned an error while checking runtime support.\n"
            f"{details}\n\n{LINUX_GPU_HELP}"
        )

    if "nvidia" not in info.stdout.lower():
        raise LaunchError(
            "NVIDIA Container Toolkit not detected. Install it before running GPU runtimes on Linux.\n\n"
            f"{LINUX_GPU_HELP}"
        )

# hàm để tạo tên container Docker hợp lệ từ tên mô hình và chế độ runtime
# thay thế các ký tự không hợp lệ bằng dấu gạch ngang
# giới hạn độ dài tên container tối đa là 63 ký tự
def sanitize_container_name(model_name: str, runtime_key: str) -> str:
    base = f"{CONTAINER_NAME_PREFIX}{model_name}-{runtime_key}"
    slug_chars: List[str] = []
    for char in base.lower():
        if char.isalnum() or char == "-":
            slug_chars.append(char)
        else:
            slug_chars.append("-")
    slug = "".join(slug_chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:63] or "tei-runtime"

# hàm để liệt kê các container TEI đang chạy
# trả về danh sách tên container
def list_active_containers(docker_cmd: str) -> List[str]:
    result = subprocess.run(
        [
            docker_cmd,
            "ps",
            "--format",
            "{{.Names}}",
            "--filter",
            f"name={CONTAINER_NAME_PREFIX}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise LaunchError(f"Failed to query running TEI containers: {details}")

    return [line.strip() for line in result.stdout.splitlines() if line.strip()]

# hàm để dừng một container TEI cụ thể theo tên

def stop_container(docker_cmd: str, container_name: str) -> None:
    result = subprocess.run(
        [docker_cmd, "stop", container_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise LaunchError(f"Failed to stop container `{container_name}`: {details}")

# hàm để dừng tất cả các container TEI đang chạy
# trả về danh sách tên các container đã dừng
def stop_all_containers(docker_cmd: str) -> List[str]:
    running = list_active_containers(docker_cmd)
    stopped: List[str] = []
    for name in running:
        stop_container(docker_cmd, name)
        stopped.append(name)
    return stopped

# hàm để xây dựng lệnh Docker run với các tham số đã cho
# trả về danh sách các thành phần lệnh
# sử dụng các tham số cấu hình mô hình và runtime để tạo lệnh phù hợp
def build_docker_command(
    docker_cmd: str,
    runtime_spec: RuntimeSpec,
    model_cfg: Dict[str, object],
    model_path: Path,
    hostname: str,
    port_value: int,
    extra_router_args: List[str],
    container_name: str,
    detach: bool,
) -> List[str]:
    if port_value <= 0 or port_value > 65535:
        raise LaunchError(f"Invalid port value: {port_value}")

    mount_arg = f"{model_path.resolve().as_posix()}:{CONTAINER_MODEL_PATH}"

    command: List[str] = [
        docker_cmd,
        "run",
        "--rm",
        "--name",
        container_name,
        "-p",
        f"{port_value}:{CONTAINER_PORT}",
        "-v",
        mount_arg,
        "--pull",
        "always",
        "--label",
        f"com.docker.compose.project={COMPOSE_PROJECT_NAME}",
        "--label",
        f"com.docker.compose.service={COMPOSE_SERVICE_NAME}",
        "--label",
        f"com.docker.compose.version={COMPOSE_VERSION}",
    ]

    if detach:
        command.append("-d")

    if runtime_spec.requires_gpu:
        command.extend(["--gpus", "all"])

    command.append(runtime_spec.image)

    router_command = [
        "--model-id",
        CONTAINER_MODEL_PATH,
        "--hostname",
        hostname,
    ]

    optional_flag_map = {
        "max_client_batch_size": "--max-client-batch-size",
        "max_batch_tokens": "--max-batch-tokens",
        "dtype": "--dtype",
        "revision": "--revision",
        "pooling": "--pooling",
        "normalize": "--normalize",
        "trust_remote_code": "--trust-remote-code",
        "auto_truncate": "--auto-truncate",
    }

    for key, flag in optional_flag_map.items():
        if key not in model_cfg:
            continue
        value = model_cfg[key]
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                router_command.append(flag)
        else:
            router_command.extend([flag, str(value)])

    router_command.extend(extra_router_args)
    command.extend(router_command)
    return command

# hàm để tải một file từ URL dựa trên đường dẫn tương đối
# lưu file vào thư mục đích đã cho
# xử lý lỗi HTTP và lỗi kết nối
def list_models(models_cfg: Dict[str, Dict[str, object]]) -> None:
    print("Available TEI models:\n")
    for name, cfg in models_cfg.items():
        raw_path = str(cfg.get("path", ""))
        resolved_path = resolve_model_path(raw_path) if raw_path else Path("??")
        port = cfg.get("port", "auto (default 8800)")
        exists_mark = "OK" if resolved_path.exists() else "MISSING PATH"
        print(f"- {name}: port {port}, path {resolved_path} [{exists_mark}]")

    print("\nRuntime options:\n")
    for key, spec in RUNTIME_SPECS.items():
        accelerator = "GPU" if spec.requires_gpu else "CPU"
        print(f"- {key}: {spec.label} ({accelerator}) -> {spec.image}")

    print(
        "\nLaunch example:\n"
        "  python backend/tools/launch_tei.py --model sentence-transformers-all-MiniLM-L6-v2 --runtime cpu\n"
    )

# hàm để phân tích các đối số dòng lệnh
# trả về Namespace chứa các đối số đã phân tích
# sử dụng các tham số cấu hình mô hình và runtime để tạo lệnh phù hợp
def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a Hugging Face TEI container for local embedding models."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List configured models and runtimes, then exit.",
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Model name to launch (must match models.json).",
    )
    parser.add_argument(
        "--runtime",
        default=DEFAULT_RUNTIME_KEY,
        choices=sorted(RUNTIME_SPECS.keys()),
        help="TEI runtime mode / Docker image (default: cpu).",
    )
    parser.add_argument(
        "--hostname",
        default=DEFAULT_HOST,
        help=f"Address to bind inside the container (default: {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Override the port defined in models.json.",
    )
    parser.add_argument(
        "--container-name",
        help="Optional Docker container name (default derives from model and runtime).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the docker run command and exit without launching.",
    )
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument(
        "--stop",
        action="store_true",
        help="Stop the TEI container for the selected model/runtime and exit.",
    )
    action_group.add_argument(
        "--stop-all",
        action="store_true",
        help="Stop all running TEI containers and exit.",
    )
    action_group.add_argument(
        "--status",
        action="store_true",
        help="List running TEI containers and exit.",
    )
    parser.add_argument(
        "--detach",
        dest="detach",
        action="store_true",
        help="Run the Docker container in detached mode.",
    )
    parser.add_argument(
        "--no-detach",
        dest="detach",
        action="store_false",
        help="Run the Docker container in the foreground (default).",
    )
    parser.set_defaults(detach=False)
    parser.add_argument(
        "router_args",
        nargs=argparse.REMAINDER,
        help="Extra CLI arguments forwarded to text-embeddings-inference (prepend with `--`).",
    )
    return parser.parse_args(argv)

# hàm chính để xử lý các đối số dòng lệnh và khởi chạy container TEI
# sử dụng các hàm phụ để xây dựng và chạy lệnh Docker
# báo cáo tiến trình và kết quả cuối cùng
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_arguments(argv)
    models_cfg: Optional[Dict[str, Dict[str, object]]] = None

    needs_models = any(
        [
            args.list,
            bool(args.model),
            args.stop,
        ]
    )

    if needs_models:
        models_cfg = load_models_config()

    if args.list or not args.model:
        if models_cfg is None:
            models_cfg = load_models_config()
        list_models(models_cfg)
        if not args.model:
            return 0

    docker_cmd: Optional[str] = None

    if args.status or args.stop_all or args.stop:
        docker_cmd = ensure_docker_available()

    if args.status:
        running = list_active_containers(docker_cmd)
        if running:
            print("Running TEI containers:")
            for name in running:
                print(f"- {name}")
        else:
            print("No TEI containers are currently running.")
        return 0

    if args.stop_all:
        stopped = stop_all_containers(docker_cmd)
        if stopped:
            for name in stopped:
                print(f"Stopped {name}")
        else:
            print("No TEI containers to stop.")
        return 0

    model_key = args.model
    if models_cfg is None or model_key not in models_cfg:
        available = ", ".join(sorted(models_cfg or {}))
        raise LaunchError(f"Unknown model `{model_key}`. Available: {available}")

    runtime_key = args.runtime or DEFAULT_RUNTIME_KEY
    runtime_spec = RUNTIME_SPECS[runtime_key]
    model_cfg = models_cfg[model_key]

    if args.stop:
        container_name = args.container_name or sanitize_container_name(model_key, runtime_key)
        running = list_active_containers(docker_cmd)
        if container_name not in running:
            print(f"No running container found for `{container_name}`.")
            return 0
        stop_container(docker_cmd, container_name)
        print(f"Stopped {container_name}")
        return 0

    if "path" not in model_cfg:
        raise LaunchError(f"Missing `path` for model `{model_key}` in {CONFIG_PATH}")

    model_path = resolve_model_path(str(model_cfg["path"]))
    if not model_path.exists():
        raise LaunchError(
            f"Model directory for `{model_key}` does not exist: {model_path}\n"
            "Verify the `path` value in models.json."
        )

    port_value = args.port or int(model_cfg.get("port", 8800))

    docker_cmd = docker_cmd or ensure_docker_available()

    if runtime_spec.requires_gpu:
        ensure_gpu_runtime_ready(docker_cmd)

    container_name = args.container_name or sanitize_container_name(model_key, runtime_key)

    running = list_active_containers(docker_cmd)
    for name in running:
        if name == container_name:
            stop_container(docker_cmd, name)
            print(f"Stopped existing container {name} before relaunch.")
        else:
            stop_container(docker_cmd, name)
            print(f"Stopped concurrent TEI container {name}.")

    extra_router_args = args.router_args or []
    command = build_docker_command(
        docker_cmd=docker_cmd,
        runtime_spec=runtime_spec,
        model_cfg=model_cfg,
        model_path=model_path,
        hostname=args.hostname,
        port_value=port_value,
        extra_router_args=extra_router_args,
        container_name=container_name,
        detach=args.detach,
    )

    command_str = shlex_join(command)
    print(f"[TEI] Container : {container_name}")
    print(f"[TEI] Image     : {runtime_spec.image}")
    print(f"[TEI] Model dir : {model_path}")
    print(f"[TEI] Host port : {port_value}")
    print(f"[TEI] Command   : {command_str}")

    if runtime_spec.requires_gpu and platform.system().lower() == "linux":
        print("[TEI] Reminder  : GPU runtimes on Linux require NVIDIA Container Toolkit.")

    if args.dry_run:
        return 0

    try:
        subprocess.run(command, cwd=str(CONFIG_PATH.parent), check=True)
    except KeyboardInterrupt:
        print("\n[TEI] Interrupted, shutting down...")
    except subprocess.CalledProcessError as exc:
        raise LaunchError(f"TEI container exited with error code {exc.returncode}") from exc

    if args.detach:
        print(f"\n[TEI] Detached container `{container_name}` started successfully.")
    else:
        print("\nPress Ctrl+C to stop the container. It will be removed automatically thanks to --rm.\n")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except LaunchError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
