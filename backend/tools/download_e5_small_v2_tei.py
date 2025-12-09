from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import shutil

# URL cơ sở để tải các tài sản mô hình từ Hugging Face
BASE_URL = "https://huggingface.co/intfloat/e5-small-v2/resolve/main"

# Files required to run intfloat/e5-small-v2 with the TEI backend
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "modules.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
    "vocab.txt",
)

# mở rộng các file tùy chọn cho các định dạng khác nhau như ONNX và OpenVINO
OPTIONAL_FILES = (
    "model.onnx",
    "onnx/model_O4.onnx",
    "onnx/model_qint8_avx512_vnni.onnx",
    "openvino/openvino_model.bin",
    "openvino/openvino_model.xml",
    "openvino/openvino_model_qint8_quantized.bin",
    "openvino/openvino_model_qint8_quantized.xml",
    "pytorch_model.bin",
    "tf_model.h5",
    "README.md",
)

# hàm để xác định thư mục đích lưu trữ tài sản
# nếu không chỉ định thì sử dụng thư mục mặc định trong backend/local-llm/embedding
def resolve_target_dir(custom_target: str | None = None) -> Path:
    """
    Resolve the directory where the assets should be stored.

    Defaults to backend/local-llm/embedding/intfloat-e5-small-v2 relative to this file.
    """
    if custom_target:
        target_path = Path(custom_target).expanduser().resolve()
    else:
        target_path = (
            Path(__file__)
            .resolve()
            .parents[1]
            .joinpath("local-llm", "embedding", "intfloat-e5-small-v2")
        )
    target_path.mkdir(parents=True, exist_ok=True)
    return target_path

# hàm để tải một file từ URL và lưu vào thư mục đích
# xử lý lỗi HTTP và lỗi kết nối
def download_file(relative_path: str, destination_dir: Path) -> None:
    destination_path = destination_dir / relative_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"{BASE_URL}/{relative_path}?download=1"
    request = Request(url, headers={"User-Agent": "curl/7.79.1"})

    try:
        with urlopen(request) as response, open(destination_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP error {exc.code} while fetching {relative_path}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to reach server for {relative_path}: {exc}") from exc

# hàm chính để xử lý đối số dòng lệnh và tải các file cần thiết
# sử dụng các hàm phụ để kiểm tra và tải từng file
# báo cáo tiến trình và kết quả cuối cùng
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download required intfloat/e5-small-v2 files for TEI backend."
    )
    parser.add_argument(
        "--target",
        help=(
            "Custom directory to place the downloaded files. "
            "Defaults to backend/local-llm/embedding/intfloat-e5-small-v2."
        ),
    )
    args = parser.parse_args()

    target_dir = resolve_target_dir(args.target)
    print(f"Downloading files into {target_dir}")
# hàm phụ để đảm bảo tải các file trong danh sách
# bỏ qua các file đã tồn tại
    def _download_many(file_list: tuple[str, ...], is_optional: bool = False) -> None:
        for relative in file_list:
            destination_path = target_dir / relative
            if destination_path.exists():
                print(f"[skip] {relative} already exists")
                continue
            print(f"[download] {relative}")
            try:
                download_file(relative, target_dir)
            except RuntimeError as exc:
                if is_optional:
                    print(f"[warn] optional download failed for {relative}: {exc}")
                    continue
                raise

    _download_many(REQUIRED_FILES, is_optional=False)
    _download_many(OPTIONAL_FILES, is_optional=True)

    print("Download complete.")
    return 0

# chạy hàm chính nếu được gọi trực tiếp
if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
