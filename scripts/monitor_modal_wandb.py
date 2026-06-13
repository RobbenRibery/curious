#!/usr/bin/env -S uv run python
import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import wandb
from dotenv import load_dotenv


DEFAULT_METRICS = (
    "num_batches_visited",
    "train/mean_batch_returns",
    "train/mean_batch_solved_rate",
    "train/mean_batch_outcome_returns",
    "train/mean_num_words_in_completions",
    "train/mean_action_entropy",
    "train/loss",
    "train/actor_loss",
    "train/mean_kl",
    "train/kl_loss",
    "train/kl_weight",
    "train/kl_weight_used",
    "train/kl_weight_next",
    "train/grad_norm",
    "train/lr",
)


def default_modal_bin() -> str:
    uv_tool_modal = Path.home() / ".local" / "bin" / "modal"
    if uv_tool_modal.exists():
        return str(uv_tool_modal)
    return "modal"


def normalized_lookup(item: dict[str, Any], *candidate_keys: str) -> Any:
    normalized = {
        key.lower().replace(" ", "_").replace("-", "_"): value
        for key, value in item.items()
    }
    for key in candidate_keys:
        value = normalized.get(key.lower().replace(" ", "_").replace("-", "_"))
        if value is not None:
            return value
    return None


def find_wandb_run(entity: str, project: str, run_name: str) -> Any | None:
    load_dotenv(Path.cwd() / ".env")
    api = wandb.Api()
    path = f"{entity}/{project}"

    for filters in ({"display_name": run_name}, {"name": run_name}):
        runs = list(api.runs(path, filters=filters, per_page=10))
        if runs:
            return runs[0]

    recent_runs = list(api.runs(path, order="-created_at", per_page=50))
    for run in recent_runs:
        if run.name == run_name or run.id == run_name:
            return run
    return None


def latest_wandb_metrics(run: Any, metric_keys: tuple[str, ...]) -> dict[str, Any]:
    summary = dict(run.summary)
    return {
        key: summary.get(key)
        for key in metric_keys
        if summary.get(key) is not None
    }


def modal_app_list(modal_bin: str) -> list[dict[str, Any]]:
    completed = subprocess.run(
        [modal_bin, "app", "list", "--json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    if not completed.stdout.strip():
        return []
    return json.loads(completed.stdout)


def matching_modal_apps(apps: list[dict[str, Any]], app_name: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for app in apps:
        app_id = normalized_lookup(app, "app_id", "id")
        description = normalized_lookup(app, "description", "name")
        if app_name in {app_id, description}:
            matches.append(app)
    return matches


def print_modal_state(modal_bin: str, app_name: str, show_logs: bool, log_tail: int) -> None:
    apps = modal_app_list(modal_bin)
    matches = matching_modal_apps(apps, app_name)
    if not matches:
        print(f"Modal: no app named or identified as {app_name!r} is currently listed.")
        return

    for app in matches:
        app_id = normalized_lookup(app, "app_id", "id")
        description = normalized_lookup(app, "description", "name")
        state = normalized_lookup(app, "state")
        tasks = normalized_lookup(app, "tasks")
        created_at = normalized_lookup(app, "created_at")
        print(f"Modal: app_id={app_id} description={description} state={state} tasks={tasks} created_at={created_at}")

        if show_logs and app_id:
            logs = subprocess.run(
                [modal_bin, "app", "logs", str(app_id), "--tail", str(log_tail), "--timestamps"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if logs.returncode == 0 and logs.stdout.strip():
                print("Modal logs:")
                print(logs.stdout.rstrip())
            elif logs.stderr.strip():
                print(f"Modal logs unavailable: {logs.stderr.strip()}")


def print_wandb_state(entity: str, project: str, run_name: str, metric_keys: tuple[str, ...]) -> None:
    try:
        run = find_wandb_run(entity=entity, project=project, run_name=run_name)
    except Exception as exc:
        print(f"W&B: project or run is not available yet: {exc}")
        return
    if run is None:
        print(f"W&B: run {entity}/{project}/{run_name!r} not found yet.")
        return

    run.load(force=True)
    print(f"W&B: run={run.name} id={run.id} state={run.state} url={run.url}")
    metrics = latest_wandb_metrics(run, metric_keys)
    if not metrics:
        print("W&B: no requested metrics have reached run.summary yet.")
        return
    for key in metric_keys:
        if key in metrics:
            print(f"  {key}: {metrics[key]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll W&B metrics and Modal app state for a Curious training run.")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "autocurriculum"))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "curious"))
    parser.add_argument("--run-name", default=os.environ.get("RUN_NAME", "baseline-gsm8k-grpo-qwen3-1p7b-sglang-fa3-h100-seed42-v1"))
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3-1.7B"))
    parser.add_argument("--generation-backend", default=os.environ.get("GENERATION_BACKEND", "sglang"))
    parser.add_argument("--sglang-attention-backend", default=os.environ.get("SGLANG_ATTENTION_BACKEND", "fa3"))
    parser.add_argument("--modal-app", default=os.environ.get("MODAL_APP", "curious-training"))
    parser.add_argument("--modal-bin", default=os.environ.get("MODAL_BIN", default_modal_bin()))
    parser.add_argument("--poll-seconds", type=float, default=float(os.environ.get("POLL_SECONDS", "60")))
    parser.add_argument("--max-polls", type=int, default=None)
    parser.add_argument("--metric", action="append", default=None, help="Metric key to print. Repeatable.")
    parser.add_argument("--show-modal-logs", action="store_true")
    parser.add_argument("--modal-log-tail", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric_keys = tuple(args.metric) if args.metric else DEFAULT_METRICS
    poll_idx = 0

    while args.max_polls is None or poll_idx < args.max_polls:
        poll_idx += 1
        print("=" * 80)
        print(f"poll={poll_idx} time={datetime.now(timezone.utc).isoformat()}")
        print(
            "target="
            f"run={args.run_name} model={args.model_name} "
            f"backend={args.generation_backend} attention={args.sglang_attention_backend}"
        )
        try:
            print_wandb_state(
                entity=args.entity,
                project=args.project,
                run_name=args.run_name,
                metric_keys=metric_keys,
            )
        except Exception as exc:
            print(f"W&B polling failed: {exc}")

        try:
            print_modal_state(
                modal_bin=args.modal_bin,
                app_name=args.modal_app,
                show_logs=args.show_modal_logs,
                log_tail=args.modal_log_tail,
            )
        except Exception as exc:
            print(f"Modal polling failed: {exc}")

        if args.max_polls is not None and poll_idx >= args.max_polls:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
