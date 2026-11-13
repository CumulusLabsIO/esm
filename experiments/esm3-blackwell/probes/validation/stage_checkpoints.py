#!/usr/bin/env python3
"""Stage exact ESM3 weights and record the ESMC revision for local examples."""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
records = {}
for name, repo, revision in (
    ("esm3", "biohub/esm3-sm-open-v1", "47f0545b2b6daf26a93439a3cd610f4f7f3d5478"),
    ("esmc", "biohub/ESMC-300M", os.environ.get("ESMC_REVISION", "main")),
):
    directory = Path(snapshot_download(repo_id=repo, revision=revision)).resolve()
    records[name] = {
        "repo": repo,
        "requested_revision": revision,
        "resolved_revision": directory.name,
        "directory": str(directory),
        "files": [
            {"path": str(p.relative_to(directory)), "bytes": p.stat().st_size}
            for p in sorted(directory.rglob("*"))
            if p.is_file()
        ],
    }
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, indent=2) + "\n")
    temporary.replace(args.output)
    print(f"Staged {name}: {directory}", flush=True)
assert (
    Path(records["esm3"]["directory"]) / "data/weights/esm3_sm_open_v1.pth"
).is_file()
