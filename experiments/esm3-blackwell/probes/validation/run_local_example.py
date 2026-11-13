#!/usr/bin/env python3
"""Run a checkout script unchanged, with checkpoint data and outputs in separate locations."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoints", type=Path, required=True)
parser.add_argument("--workdir", type=Path, required=True)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
if not command:
    parser.error("A script path must follow --")
script = Path(command[0]).resolve()
records = json.loads(args.checkpoints.read_text())
data = Path(records["esm3"]["directory"]) / "data"
assert data.is_dir(), data
work = args.workdir.resolve()
work.mkdir(parents=True, exist_ok=True)
link = work / "data"
if not link.exists():
    link.symlink_to(data, target_is_directory=True)
assert link.resolve() == data.resolve()
(work / "invocation.json").write_text(
    json.dumps(
        {
            "script": str(script),
            "sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "arguments": command[1:],
            "esm3_revision": records["esm3"]["resolved_revision"],
            "esmc_revision": records["esmc"]["resolved_revision"],
        },
        indent=2,
    )
    + "\n"
)
os.environ.pop("ESM_API_KEY", None)
os.environ["INFRA_PROVIDER"] = "local-pinned-validation"
os.chdir(work)
sys.argv = [str(script), *command[1:]]
runpy.run_path(str(script), run_name="__main__")
