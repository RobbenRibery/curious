import argparse
import os
import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = "curious-training"
ARTIFACTS_DIR = "/modal-artifacts"
ARTIFACTS_VOLUME_NAME = "curious-training-artifacts"
DEFAULT_GPU = "H100"
DEFAULT_TIMEOUT_SECONDS = 24 * 60 * 60

LOCAL_SECRET_ENV_KEYS = (
    "WANDB_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)

app = modal.App(APP_NAME)
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)

training_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "build-essential",
        "git",
    )
    .uv_sync(uv_version="0.6.12")
    .workdir("/root")
    .env(
        {
            "TOKENIZERS_PARALLELISM": "true",
            "TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True",
            "HF_HOME": f"{ARTIFACTS_DIR}/hf",
            "WANDB_DIR": f"{ARTIFACTS_DIR}/wandb",
            "WANDB_CACHE_DIR": f"{ARTIFACTS_DIR}/wandb_cache",
        }
    )
    .add_local_python_source("curious")
)


def _has_cli_option(args: list[str], option: str) -> bool:
    option_prefix = f"{option}="
    return any(arg == option or arg.startswith(option_prefix) for arg in args)


def _with_default_artifact_paths(args: list[str]) -> list[str]:
    defaults = {
        "--base-config.train-log-dir": f"{ARTIFACTS_DIR}/train_logs",
        "--base-config.eval-log-dir": f"{ARTIFACTS_DIR}/eval_logs",
        "--base-config.checkpoint-dir": f"{ARTIFACTS_DIR}/checkpoints",
    }
    resolved_args = list(args)
    for option, value in defaults.items():
        if not _has_cli_option(resolved_args, option):
            resolved_args.extend([option, value])
    return resolved_args


def _dotenv_values(dotenv_path: str) -> dict[str, str]:
    path = Path(dotenv_path)
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] in {"'", '"'} and value[-1:] == value[0]:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _local_secrets(dotenv_path: str | None, include_env_secret: bool, secret_names: list[str]) -> list[modal.Secret]:
    secrets: list[modal.Secret] = []
    if dotenv_path is not None:
        dotenv_secret = _dotenv_values(dotenv_path)
        if dotenv_secret:
            secrets.append(modal.Secret.from_dict(dotenv_secret))

    if include_env_secret:
        env_secret = {
            key: value
            for key in LOCAL_SECRET_ENV_KEYS
            if (value := os.environ.get(key))
        }
        if env_secret:
            secrets.append(modal.Secret.from_dict(env_secret))

    secrets.extend(modal.Secret.from_name(name) for name in secret_names)
    return secrets


@app.function(
    image=training_image,
    gpu=DEFAULT_GPU,
    volumes={ARTIFACTS_DIR: artifacts_volume},
    cpu=16,
    memory=96 * 1024,
    timeout=DEFAULT_TIMEOUT_SECONDS,
)
def run_training(training_args: list[str]) -> int:
    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "curious.training",
        *_with_default_artifact_paths(training_args),
    ]
    print("Launching training command:", " ".join(command), flush=True)
    try:
        completed = subprocess.run(command, check=False)
        return completed.returncode
    finally:
        artifacts_volume.commit()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch curious training on Modal and forward trailing args to curious.training.",
    )
    parser.add_argument("--gpu", default=DEFAULT_GPU, help="Modal GPU request, e.g. H100, A100-80GB, H100:2.")
    parser.add_argument("--cloud", default=None, help="Optional Modal cloud selector, e.g. aws, gcp, oci, auto.")
    parser.add_argument("--region", default=None, help="Optional Modal region selector.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="Function timeout in seconds.")
    parser.add_argument("--secret", action="append", default=[], help="Modal Secret name to inject. Repeatable.")
    parser.add_argument("--dotenv", default=".env", help="Local dotenv path to send as a Modal Secret.")
    parser.add_argument("--no-dotenv", action="store_true", help="Do not read a local dotenv file.")
    parser.add_argument(
        "--no-local-env-secret",
        action="store_true",
        help="Do not forward WANDB_API_KEY/HF_TOKEN/HUGGING_FACE_HUB_TOKEN from the local environment.",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Spawn the remote function and return immediately after Modal accepts the call.",
    )
    return parser


def _parse_local_args(arglist: tuple[str, ...]) -> tuple[argparse.Namespace, list[str]]:
    raw_args = list(arglist)
    if "--" in raw_args:
        separator_index = raw_args.index("--")
        launcher_args = raw_args[:separator_index]
        training_args = raw_args[separator_index + 1 :]
    else:
        launcher_args = raw_args
        training_args = []

    parser = _build_arg_parser()
    parsed_args, unknown_args = parser.parse_known_args(launcher_args)
    return parsed_args, [*unknown_args, *training_args]


@app.local_entrypoint()
def main(*arglist: str) -> None:
    launcher_args, training_args = _parse_local_args(arglist)
    dotenv_path = None if launcher_args.no_dotenv else launcher_args.dotenv
    function = run_training.with_options(
        gpu=launcher_args.gpu,
        cloud=launcher_args.cloud,
        region=launcher_args.region,
        timeout=launcher_args.timeout,
        secrets=_local_secrets(
            dotenv_path=dotenv_path,
            include_env_secret=not launcher_args.no_local_env_secret,
            secret_names=launcher_args.secret,
        ),
    )

    call = function.spawn(training_args)
    print(f"Started Modal training call {call.object_id}")
    print(f"Artifacts volume: {ARTIFACTS_VOLUME_NAME}:{ARTIFACTS_DIR}")
    if launcher_args.background:
        return

    exit_code = call.get()
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    _build_arg_parser().print_help()
