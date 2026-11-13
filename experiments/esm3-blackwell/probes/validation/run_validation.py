#!/usr/bin/env python3
"""Durable, bounded validation of the recorded PR head; does not install GPU drivers."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HEAD = "d285ddeead83631bbe0bc6aa7c879eb5a8946ccb"
MODEL_HASH = "913e6fad41315e24b5575982c3c54d1ebed8dc6fe10fb8442bc0a6ac3c5ab5ab"
ACTIVE = None


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stop_active():
    if ACTIVE is not None and ACTIVE.poll() is None:
        os.killpg(ACTIVE.pid, signal.SIGTERM)
        try:
            ACTIVE.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(ACTIVE.pid, signal.SIGKILL)
            ACTIVE.wait()


def interrupted(signum, _frame):
    stop_active()
    raise SystemExit(128 + signum)


def main():
    global ACTIVE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--pixi", required=True)
    parser.add_argument("--perf-python", type=Path)
    parser.add_argument(
        "--phase", choices=("locked", "performance", "all"), default="all"
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cpu-workers", default="8")
    parser.add_argument("--budget-seconds", type=int, default=5400)
    parser.add_argument("--extended-gpu", action="store_true")
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.results = args.results.resolve()
    args.results.mkdir(parents=True, exist_ok=True)
    if (args.results / "summary.json").exists():
        raise ValueError(
            "Use a fresh results directory; existing evidence must not be overwritten"
        )
    harness = Path(__file__).resolve().parent
    started = time.time()
    deadline = started + args.budget_seconds
    records = []
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    actual_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.repo, text=True
    ).strip()
    actual_pixi = subprocess.check_output([args.pixi, "--version"], text=True).strip()
    if actual_pixi != "pixi 0.82.1":
        raise RuntimeError(f"Expected CI pixi0.82.1; got {actual_pixi}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=args.repo,
        text=True,
    )
    source_hash = hashlib.sha256(
        (args.repo / "esm/models/esm3.py").read_bytes()
    ).hexdigest()
    if actual_head != HEAD or dirty or source_hash != MODEL_HASH:
        raise RuntimeError(
            f"Expected clean {HEAD} and recorded ESM3 source; got head={actual_head}, dirty={dirty!r}, source={source_hash}"
        )
    atomic_json(
        args.results / "source.json",
        {
            "head": actual_head,
            "esm3_source_sha256": source_hash,
            "pixi_lock_sha256": hashlib.sha256(
                (args.repo / "pixi.lock").read_bytes()
            ).hexdigest(),
            "started_epoch": started,
            "deadline_epoch": deadline,
            "phase": args.phase,
            "gpu": args.gpu,
            "cpu_workers": args.cpu_workers,
            "scope": "Locked CI lint/CPU tests plus explicitly selected GPU tests; performance recipes use a separate Torch2.15/cu132 environment. No Forge API, Docker SDK API integration, notebook suite, or multi-GPU training.",
        },
    )
    env = os.environ.copy()
    # Force local example branches without ever logging or persisting API credentials.
    env.pop("ESM_API_KEY", None)
    env.pop("INFRA_PROVIDER", None)
    env.update(
        OMP_NUM_THREADS="4",
        PYTHONPATH=str(args.repo),
        PYTHONUNBUFFERED="1",
        CUDA_VISIBLE_DEVICES=args.gpu,
        PYTEST_XDIST_AUTO_NUM_WORKERS=args.cpu_workers,
    )
    pixi = [
        args.pixi,
        "run",
        "--manifest-path",
        str(args.repo / "pyproject.toml"),
        "--locked",
        "-e",
        "dev",
    ]

    def run(label, command, timeout=600, updates=None, cwd=None, required=True):
        nonlocal records
        global ACTIVE
        remaining = deadline - time.time()
        if remaining < 20:
            record = {"label": label, "status": "not_run_budget", "required": required}
            records.append(record)
            atomic_json(args.results / "summary.json", records)
            return False
        job_env = env.copy()
        if updates:
            job_env.update(updates)
        record = {
            "label": label,
            "command": [str(x) for x in command],
            "required": required,
            "started_epoch": time.time(),
            "status": "running",
        }
        records.append(record)
        atomic_json(args.results / "summary.json", records)
        print(f"START {label}", flush=True)
        with (args.results / f"{label}.log").open("w") as log:
            ACTIVE = subprocess.Popen(
                command,
                cwd=cwd or args.repo,
                env=job_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = ACTIVE.wait(timeout=min(timeout, remaining - 5))
                record.update(
                    exit_code=code, status="passed" if code == 0 else "failed"
                )
            except subprocess.TimeoutExpired:
                stop_active()
                record.update(exit_code=124, status="timeout")
            finally:
                ACTIVE = None
        record["finished_epoch"] = time.time()
        (args.results / f"{label}.exit").write_text(str(record["exit_code"]) + "\n")
        atomic_json(args.results / "summary.json", records)
        print(f"END {label}: {record['status']}", flush=True)
        return record["status"] == "passed"

    def python_job(label, python, script, extra=(), **kwargs):
        return run(label, [*python, str(harness / script), *map(str, extra)], **kwargs)

    if args.phase in ("locked", "all"):
        ready = run(
            "locked-install",
            [
                args.pixi,
                "install",
                "--manifest-path",
                str(args.repo / "pyproject.toml"),
                "--locked",
                "-e",
                "dev",
            ],
            timeout=1200,
        )
        if ready:
            python_job(
                "locked-environment",
                [*pixi, "python"],
                "environment_probe.py",
                [
                    "--expected-torch",
                    "2.11.0",
                    "--output",
                    args.results / "locked-environment.json",
                ],
            )
            run(
                "locked-lint",
                [*pixi, "lint-all"],
                timeout=900,
                updates={"CUDA_VISIBLE_DEVICES": ""},
            )
            run(
                "locked-cpu",
                [*pixi, "cov-test"],
                timeout=900,
                updates={"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"},
            )
            # Move CI's generated reports to the durable evidence directory.
            for name in ("pytest.xml", "coverage.xml"):
                report = args.repo / name
                if report.exists():
                    (args.results / f"locked-{name}").write_bytes(report.read_bytes())
            run(
                "locked-esm3-regression",
                [
                    *pixi,
                    "python",
                    "-m",
                    "pytest",
                    "-o",
                    "addopts=",
                    "-n",
                    "0",
                    "-ra",
                    "--junitxml",
                    str(args.results / "locked-esm3-regression.xml"),
                    "tests/models/esm3_cuda_graph_test.py",
                    "tests/models/esm3_logits_selection_test.py",
                ],
                updates={"HF_HUB_OFFLINE": "1"},
            )
            if args.extended_gpu:
                run(
                    "locked-small-gpu",
                    [
                        *pixi,
                        "python",
                        "-m",
                        "pytest",
                        "-o",
                        "addopts=",
                        "-n",
                        "0",
                        "-m",
                        "gpu and not manual",
                        "-p",
                        "small_gpu_selection",
                        "-ra",
                        "--junitxml",
                        str(args.results / "locked-small-gpu.xml"),
                        "tests/models",
                    ],
                    timeout=1200,
                    updates={
                        "HF_HUB_OFFLINE": "1",
                        "PYTHONPATH": str(harness) + os.pathsep + str(args.repo),
                        "ESM_SELECTION_REPORT": str(
                            args.results / "small-gpu-selection.json"
                        ),
                    },
                    required=False,
                )

    if args.phase in ("performance", "all"):
        if args.perf_python is None:
            raise ValueError("--perf-python is required for the performance phase")
        python = [str(args.perf_python.absolute())]
        ready = python_job(
            "performance-environment",
            python,
            "environment_probe.py",
            [
                "--expected-torch",
                "2.15.0",
                "--expected-cuda",
                "13.2",
                "--require-strict-numerics",
                "--output",
                args.results / "performance-environment.json",
            ],
        )
        if ready:
            staged = python_job(
                "stage-checkpoints",
                python,
                "stage_checkpoints.py",
                ["--output", args.results / "checkpoints.json"],
                timeout=900,
                updates={"HF_HUB_OFFLINE": "0"},
            )
            if staged:
                run(
                    "performance-esm3-regression",
                    [
                        *python,
                        "-m",
                        "pytest",
                        "-o",
                        "addopts=",
                        "-n",
                        "0",
                        "-ra",
                        "--junitxml",
                        str(args.results / "performance-esm3-regression.xml"),
                        "tests/models/esm3_cuda_graph_test.py",
                        "tests/models/esm3_logits_selection_test.py",
                    ],
                    updates={"HF_HUB_OFFLINE": "1"},
                )
                recipe = args.repo / "cookbook/local/esm3_cuda_graph.py"
                configurations = [
                    ("default-b1", []),
                    ("sequence-b1", ["--outputs", "sequence"]),
                    ("embeddings-b1", ["--outputs", "embeddings"]),
                    ("swiglu-b1", ["--compile-swiglu"]),
                    ("swiglu-b8", ["--compile-swiglu", "--batch-size", "8"]),
                    ("swiglu-l512", ["--compile-swiglu", "--length", "512"]),
                ]
                for label, flags in configurations:
                    python_job(
                        label,
                        python,
                        "run_local_example.py",
                        [
                            "--checkpoints",
                            args.results / "checkpoints.json",
                            "--workdir",
                            args.results / "work" / label,
                            "--",
                            recipe,
                            *flags,
                            "--warmup",
                            "20",
                            "--repeats",
                            "100",
                            "--output",
                            args.results / f"{label}.json",
                        ],
                        updates={"HF_HUB_OFFLINE": "1"},
                    )
                for label, script in [
                    ("smoke-raw-forwards", "cookbook/local/raw_forwards.py"),
                    ("smoke-esm3", "cookbook/snippets/esm3.py"),
                    ("smoke-esmc", "cookbook/snippets/esmc.py"),
                ]:
                    python_job(
                        label,
                        python,
                        "run_local_example.py",
                        [
                            "--checkpoints",
                            args.results / "checkpoints.json",
                            "--workdir",
                            args.results / "work" / label,
                            "--",
                            args.repo / script,
                        ],
                        timeout=900,
                        updates={"HF_HUB_OFFLINE": "1"},
                    )
    dirty_after = subprocess.check_output(
        ["git", "diff", "--stat"], cwd=args.repo, text=True
    )
    (args.results / "tracked-diff-after.txt").write_text(dirty_after)
    if dirty_after:
        print(
            "Tracked files changed during checks; inspect tracked-diff-after.txt",
            file=sys.stderr,
        )
        raise SystemExit(1)
    raise SystemExit(
        0 if records and all(x["status"] == "passed" for x in records) else 1
    )


if __name__ == "__main__":
    main()
