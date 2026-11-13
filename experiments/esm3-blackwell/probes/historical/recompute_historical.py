"""Recompute the historical headline figures from the preserved local JSON.

This script does not import torch, contact a service, or execute a GPU workload.
"""

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "data" / "historical" / "blackwell-profile"


def load(relative):
    return json.loads((ROOT / relative).read_text())


def inference_triplet(names):
    records = [load(f"perf-results/{name}.json") for name in names]
    medians = []
    for record in records:
        assert not record["profile_capture"]
        samples = record["wall_ms"]["samples"]
        assert len(samples) == record["repeats"]
        median = statistics.median(samples)
        assert math.isclose(median, record["wall_ms"]["median"], abs_tol=1e-10)
        assert record["output_finite"]
        medians.append(median)
    before, candidate, after = medians
    return {
        "control_before_ms": before,
        "candidate_ms": candidate,
        "control_after_ms": after,
        "speedup_vs_mean_control_medians": statistics.mean([before, after]) / candidate,
        "control_drift_percent": (after / before - 1) * 100,
        "candidate_comparison": records[1]["comparison"],
    }


def optimizer_record(name):
    record = load(f"train-profile/adamw-{name}/summary.json")
    ranks = record["ranks"]
    assert record["passed"] and len(ranks) == 8
    assert all(rank["status"] == "pass" for rank in ranks)
    assert all(rank["result"]["warmup_steps"] == 3 for rank in ranks)
    samples = [rank["result"]["measured_step_seconds"] for rank in ranks]
    assert all(len(values) == 5 for values in samples)
    assert all(math.isfinite(value) for values in samples for value in values)
    hashes = {rank["result"]["parameter_sha256"] for rank in ranks}
    assert len(hashes) == 1
    assert all(rank["result"]["replicas_identical"] for rank in ranks)
    return {
        "median_slowest_rank_step_ms": statistics.median(
            max(step) for step in zip(*samples)
        ) * 1000,
        "peak_allocated_gib": max(rank["peak_allocated_gib"] for rank in ranks),
        "peak_reserved_gib": max(rank["peak_reserved_gib"] for rank in ranks),
        "parameter_sha256": next(iter(hashes)),
        "global_mean_loss": ranks[0]["result"]["global_mean_loss"],
    }


def main():
    optimizer = {name: optimizer_record(name) for name in ("before", "fused", "after")}
    control = statistics.mean(
        optimizer[name]["median_slowest_rank_step_ms"] for name in ("before", "after")
    )
    optimizer["fused_latency_reduction_percent"] = (
        1 - optimizer["fused"]["median_slowest_rank_step_ms"] / control
    ) * 100
    optimizer["fused_allocation_reduction_percent"] = (
        1 - optimizer["fused"]["peak_allocated_gib"]
        / optimizer["before"]["peak_allocated_gib"]
    ) * 100
    result = {
        "logits_batching": inference_triplet(
            ["logits-eight-serial-before", "logits-eight-batched", "logits-eight-serial-after"]
        ),
        "structure_batching": inference_triplet(
            ["structure-eight-serial-before", "structure-eight-batch", "structure-eight-serial-after"]
        ),
        "adamw": optimizer,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
