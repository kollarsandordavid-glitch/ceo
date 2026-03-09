from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import modal


def _find_project_dir() -> Path:
    candidates = [
        Path(__file__).resolve().parent / "efla-trainer",
        Path.cwd() / "efla-trainer",
        Path(__file__).resolve().parent,
        Path.cwd(),
    ]
    for candidate in candidates:
        if (candidate / "build.zig").is_file():
            return candidate.resolve()
    searched = "\n".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(f"Could not locate the efla-trainer Zig project. Checked:\n{searched}")


def _entry_path(entry: Any) -> str:
    if hasattr(entry, "path"):
        value = getattr(entry, "path")
        return str(value)
    if isinstance(entry, dict) and "path" in entry:
        return str(entry["path"])
    if isinstance(entry, (tuple, list)) and entry:
        return str(entry[0])
    return str(entry)


def _entry_is_dir(entry: Any, path_text: str) -> bool:
    if hasattr(entry, "is_dir"):
        value = getattr(entry, "is_dir")
        if callable(value):
            return bool(value())
        return bool(value)
    if isinstance(entry, dict) and "is_dir" in entry:
        return bool(entry["is_dir"])
    entry_type = None
    if hasattr(entry, "type"):
        entry_type = getattr(entry, "type")
    elif isinstance(entry, dict) and "type" in entry:
        entry_type = entry["type"]
    if entry_type is not None:
        return str(entry_type).lower() in {"dir", "directory"}
    return path_text.endswith("/")


PROJECT_DIR = _find_project_dir()
APP_DIR = "/app/efla-trainer"
BINARY_PATH = f"{APP_DIR}/zig-out/bin/efla-train"
DEFAULT_TRAIN_CONFIG = "configs/train.yaml"
DEFAULT_SMOKE_CONFIG = "configs/smoke.yaml"
DEFAULT_TRAIN_DATA = "/data/train.bin"
DEFAULT_EVAL_DATA = "/data/eval.bin"
DEFAULT_CHECKPOINT_DIR = "/checkpoints"

app = modal.App("efla-trainer")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "git",
        "curl",
        "build-essential",
        "libnccl2",
        "libnccl-dev",
        "xz-utils",
        "ca-certificates",
    )
    .run_commands(
        "curl -fsSL https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz | tar -xJ -C /tmp",
        "mv /tmp/zig-linux-x86_64-0.13.0 /opt/zig",
        "ln -sf /opt/zig/zig /usr/local/bin/zig",
    )
    .pip_install(
        "modal",
        "pyyaml",
    )
    .copy_local_dir(str(PROJECT_DIR), APP_DIR)
    .run_commands(
        f"cd {APP_DIR} && zig build -Doptimize=ReleaseFast",
        f"test -x {BINARY_PATH}",
    )
)

data_volume = modal.Volume.from_name("efla-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("efla-checkpoints", create_if_missing=True)


@app.cls(
    image=image,
    gpu="B200:8",
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume,
    },
    timeout=86400,
    retries=0,
)
class EflaTrainer:
    @modal.enter()
    def setup(self) -> None:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
            check=True,
            capture_output=True,
            text=True,
        )
        env = os.environ
        env["NCCL_DEBUG"] = "INFO"
        env["PYTHONUNBUFFERED"] = "1"
        visible_devices = [str(index) for index, _ in enumerate(result.stdout.splitlines()[1:])]
        if visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)
        if not Path(BINARY_PATH).is_file():
            raise FileNotFoundError(BINARY_PATH)

    def _run_command(
        self,
        cmd: list[str],
        *,
        reload_data: bool = False,
        reload_checkpoints: bool = False,
        capture_output: bool = False,
    ) -> Any:
        import subprocess

        if reload_data:
            data_volume.reload()
        if reload_checkpoints:
            checkpoint_volume.reload()
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=capture_output,
            text=True,
            env=os.environ.copy(),
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip() if capture_output else ""
            stdout = (result.stdout or "").strip() if capture_output else ""
            details = "\n".join(part for part in [stdout, stderr] if part)
            if details:
                raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}\n{details}")
            raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
        return result

    @modal.method()
    def train(
        self,
        config_path: str = DEFAULT_TRAIN_CONFIG,
        resume_from: str | None = None,
        data_path: str = DEFAULT_TRAIN_DATA,
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    ) -> None:
        cmd = [
            BINARY_PATH,
            "train",
            "--config",
            f"{APP_DIR}/{config_path}",
            "--data",
            data_path,
            "--checkpoint-dir",
            checkpoint_dir,
        ]
        if resume_from:
            cmd.extend(["--resume", resume_from])
        self._run_command(cmd, reload_data=True, reload_checkpoints=True, capture_output=False)
        checkpoint_volume.commit()

    @modal.method()
    def evaluate(self, checkpoint_path: str, data_path: str = DEFAULT_EVAL_DATA) -> None:
        cmd = [
            BINARY_PATH,
            "evaluate",
            "--checkpoint",
            checkpoint_path,
            "--data",
            data_path,
        ]
        self._run_command(cmd, reload_data=True, reload_checkpoints=True, capture_output=False)

    @modal.method()
    def generate(self, checkpoint_path: str, prompt: str, max_tokens: int = 256) -> str:
        cmd = [
            BINARY_PATH,
            "generate",
            "--checkpoint",
            checkpoint_path,
            "--prompt",
            prompt,
            "--max-tokens",
            str(max_tokens),
        ]
        result = self._run_command(cmd, reload_checkpoints=True, capture_output=True)
        return result.stdout

    @modal.method()
    def status(self) -> dict[str, Any]:
        import subprocess

        checkpoint_volume.reload()
        gpu_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv"],
            check=True,
            capture_output=True,
            text=True,
        )
        checkpoint_root = Path(DEFAULT_CHECKPOINT_DIR)
        checkpoints: list[tuple[int, str]] = []
        if checkpoint_root.exists():
            for entry in checkpoint_root.iterdir():
                if not entry.is_dir() or not entry.name.startswith("step_"):
                    continue
                suffix = entry.name.removeprefix("step_")
                if suffix.isdigit():
                    checkpoints.append((int(suffix), entry.name))
        checkpoints.sort(key=lambda item: item[0])
        latest_checkpoint = checkpoints[-1][1] if checkpoints else None
        return {
            "gpu_status": gpu_result.stdout,
            "latest_checkpoint": latest_checkpoint,
            "checkpoint_count": len(checkpoints),
        }


@app.function(
    image=image,
    gpu="B200",
    timeout=3600,
)
def smoke_test(config_path: str = DEFAULT_SMOKE_CONFIG) -> bool:
    import subprocess

    result = subprocess.run(
        [BINARY_PATH, "smoke-test", "--config", f"{APP_DIR}/{config_path}"],
        check=False,
        capture_output=False,
        text=True,
    )
    return result.returncode == 0


def _upload_to_volume(volume: modal.Volume, local_source: str, remote_path: str, force: bool = False) -> None:
    source = Path(local_source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(str(source))
    normalized_remote_path = remote_path.strip("/")
    if not normalized_remote_path:
        raise ValueError("remote_path must not be empty")
    with volume.batch_upload(force=force) as batch:
        if source.is_dir():
            batch.put_directory(str(source), normalized_remote_path)
        else:
            batch.put_file(str(source), normalized_remote_path)


def _download_file_from_volume(volume: modal.Volume, remote_path: str, local_path: str) -> None:
    source_path = remote_path.strip("/")
    if not source_path:
        raise ValueError("remote_path must not be empty")
    target = Path(local_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        for chunk in volume.read_file(source_path):
            handle.write(chunk)


def _download_directory_from_volume(volume: modal.Volume, remote_path: str, local_path: str) -> None:
    source_root = remote_path.strip("/")
    if not source_root:
        raise ValueError("remote_path must not be empty")
    destination_root = Path(local_path).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    entries = volume.listdir(source_root, recursive=True)
    wrote_anything = False
    for entry in entries:
        entry_path = _entry_path(entry).strip("/")
        if not entry_path:
            continue
        if not entry_path.startswith(source_root):
            continue
        relative = Path(entry_path).relative_to(source_root)
        destination = destination_root / relative
        if _entry_is_dir(entry, entry_path):
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            for chunk in volume.read_file(entry_path):
                handle.write(chunk)
        wrote_anything = True
    if not wrote_anything:
        raise FileNotFoundError(source_root)


@app.local_entrypoint()
def main(
    action: str = "train",
    config: str = DEFAULT_TRAIN_CONFIG,
    resume: str = "",
    checkpoint: str = "latest",
    data_path: str = "",
    prompt: str = "Hello, world!",
    max_tokens: int = 256,
    local_path: str = "",
    remote_path: str = "",
    force: bool = False,
) -> None:
    trainer = EflaTrainer()
    normalized_action = action.strip().lower()

    if normalized_action == "train":
        trainer.train.remote(
            config,
            resume or None,
            data_path or DEFAULT_TRAIN_DATA,
            DEFAULT_CHECKPOINT_DIR,
        )
        return

    if normalized_action == "evaluate":
        checkpoint_path = checkpoint if checkpoint.startswith("/") else f"{DEFAULT_CHECKPOINT_DIR}/{checkpoint}"
        trainer.evaluate.remote(checkpoint_path, data_path or DEFAULT_EVAL_DATA)
        return

    if normalized_action == "generate":
        checkpoint_path = checkpoint if checkpoint.startswith("/") else f"{DEFAULT_CHECKPOINT_DIR}/{checkpoint}"
        output = trainer.generate.remote(checkpoint_path, prompt, max_tokens)
        print(output, end="" if output.endswith("\n") else "\n")
        return

    if normalized_action == "status":
        print(trainer.status.remote())
        return

    if normalized_action == "smoke-test":
        print(smoke_test.remote())
        return

    if normalized_action == "upload-data":
        if not local_path or not remote_path:
            raise ValueError("upload-data requires --local-path and --remote-path")
        _upload_to_volume(data_volume, local_path, remote_path, force)
        return

    if normalized_action == "upload-checkpoint":
        if not local_path or not remote_path:
            raise ValueError("upload-checkpoint requires --local-path and --remote-path")
        _upload_to_volume(checkpoint_volume, local_path, remote_path, force)
        return

    if normalized_action == "download-data-file":
        if not local_path or not remote_path:
            raise ValueError("download-data-file requires --local-path and --remote-path")
        _download_file_from_volume(data_volume, remote_path, local_path)
        return

    if normalized_action == "download-checkpoint":
        if not local_path:
            raise ValueError("download-checkpoint requires --local-path")
        checkpoint_path = checkpoint.strip().strip("/")
        if not checkpoint_path:
            raise ValueError("download-checkpoint requires --checkpoint")
        _download_directory_from_volume(checkpoint_volume, checkpoint_path, local_path)
        return

    raise ValueError(
        "action must be one of: train, evaluate, generate, status, smoke-test, upload-data, upload-checkpoint, download-data-file, download-checkpoint"
    )
