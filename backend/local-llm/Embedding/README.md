# TEI backend (Text Embeddings Inference)

This folder stores the locally downloaded embedding models (`Embedding/…`),
their configuration (`models.json`), and documentation for running Hugging Face
Text Embeddings Inference (TEI) in Docker.

## 1. Install Docker

`backend/tools/launch_tei.py` checks that the Docker CLI is available before
starting a container. If Docker is missing, install it with one of the official
builds:

- **Windows**
  - Installer: <https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe>
  - Microsoft Store: <https://apps.microsoft.com/detail/xp8cbj40xlbwkx>
- **macOS**
  - Apple silicon: <https://desktop.docker.com/mac/main/arm64/Docker.dmg>
  - Intel: <https://desktop.docker.com/mac/main/amd64/Docker.dmg>
- **Linux**
  - Docker Engine: <https://docs.docker.com/engine/install/>
  - Docker Desktop: <https://docs.docker.com/desktop/setup/install/linux/>

If you plan to run a GPU-enabled image on Linux, install the NVIDIA Container
Toolkit as well: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>
and verify with:

```bash
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

## 2. Review or tweak the model config

`backend/local-llm/Embedding/models.json` maps a human-friendly model name to
its local directory and preferred port. Example:

```json
{
  "AITeamVN-Vietnamese_Embedding_v2": {
    "path": "../Embedding/AITeamVN-Vietnamese_Embedding_v2",
    "port": 8800
  }
}
```

Adjust the `path` or `port` if you move models around. Optional TEI flags (for
example `max_client_batch_size`, `dtype`, `normalize`, …) can still be added per
model and will be forwarded to the container entrypoint automatically.

## 3. Launch TEI with Docker

```powershell
python backend/tools/launch_tei.py --model <model_name> [--runtime cpu]
```

Key flags:

- `--model / -m` - required, must match a key in `models.json`.
- `--runtime` - selects the Docker image. Default is `cpu`. Run with `--list` to see all modes.
- `--port` - host port to expose (mapped to port 80 inside the container).
- `--container-name` - optional explicit Docker container name (defaults to `tei-<model>-<runtime>`).
- `--detach` - run the container in the background (recommended when starting from the UI).
- `--status`, `--stop`, `--stop-all` - inspect or terminate running TEI containers.
- `--dry-run` - print the exact `docker run ...` command without executing it.

### Runtime modes

| Key           | Label                                   | Docker image                                                       |
|---------------|-----------------------------------------|--------------------------------------------------------------------|
| `cpu`         | CPU                                     | `ghcr.io/huggingface/text-embeddings-inference:cpu-1.8`            |
| `turing`      | Turing (T4 / RTX 2000 series)           | `ghcr.io/huggingface/text-embeddings-inference:turing-1.8`         |
| `ampere_80`   | Ampere 80 (A100 / A30)                  | `ghcr.io/huggingface/text-embeddings-inference:1.8`                |
| `ampere_86`   | Ampere 86 (A10 / A40)                   | `ghcr.io/huggingface/text-embeddings-inference:86-1.8`             |
| `ada_lovelace`| Ada Lovelace (RTX 4000 series)          | `ghcr.io/huggingface/text-embeddings-inference:89-1.8`             |
| `hopper`      | Hopper (H100, experimental)             | `ghcr.io/huggingface/text-embeddings-inference:hopper-1.8`         |

GPU modes add `--gpus all` to the `docker run` command. On Linux they also
require the NVIDIA Container Toolkit (see step 1).

### Examples

```powershell
# List configured models and runtime choices
python backend/tools/launch_tei.py --list

# Launch the Vietnamese embedding model on CPU mode, exposing port 8800
python backend/tools/launch_tei.py --model AITeamVN-Vietnamese_Embedding_v2 --runtime cpu

# Launch the BGE M3 model with the Ampere 86 build on port 9000
python backend/tools/launch_tei.py --model BAAI-bge-m3 --runtime ampere_86 --port 9000
```

Press `Ctrl+C` to stop the container when running in the foreground, or call `python backend/tools/launch_tei.py --model <model_name> --runtime <mode> --stop` to stop a detached run. The helper automatically stops other TEI containers before launching a new one.

## 4. Smoke test the endpoint

With the server running on port `8800`:

```powershell
curl -X POST http://localhost:8800/embed `
     -H "Content-Type: application/json" `
     -d '{"inputs": ["Hello from TEI!"]}'
```

Python check:

```python
import requests

payload = {"inputs": ["Hello from TEI!"]}
resp = requests.post("http://localhost:8800/embed", json=payload, timeout=30)
print(resp.json())
```

To stop the container manually use `docker stop <container-name>` (the helper
prints the chosen name). A new run automatically reuses the latest downloaded
image and mounts your local model directory read-only inside the container.
