"""Shared, explicit measurement protocol for independent B200 experiments."""

import argparse
import hashlib
import inspect
import json
import os
import statistics
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

MODEL_REPOSITORY = "biohub/esm3-sm-open-v1"
MODEL_REVISION = "47f0545b2b6daf26a93439a3cd610f4f7f3d5478"
UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
TRACKS = ("sequence", "structure", "secondary_structure", "sasa", "function")


def parser(description):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--batch", default=1, type=int)
    result.add_argument("--length", default=76, type=int)
    result.add_argument("--warmup", default=5, type=int)
    result.add_argument("--repeats", default=30, type=int)
    result.add_argument("--skip-profile", action="store_true")
    return result


def save(args, report):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, default=str) + "\n")
    temporary.replace(args.output)


def source_record(obj):
    path = Path(inspect.getfile(obj))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def setup(args):
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if min(args.batch, args.length, args.repeats) < 1 or args.warmup < 3:
        raise ValueError("Positive dimensions/repeats and warmup >=3 required")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    torch.cuda.set_device(0)
    model_path = Path(snapshot_download(MODEL_REPOSITORY, revision=MODEL_REVISION, local_files_only=True))
    os.chdir(model_path)
    os.environ["INFRA_PROVIDER"] = "local-pinned-test"
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, LogitsConfig
    from esm.utils.sampling import _BatchedESMProteinTensor

    model = ESM3.from_pretrained("esm3-open", device=torch.device("cuda:0")).eval()
    config = LogitsConfig(sequence=True, structure=True, secondary_structure=True,
                          sasa=True, function=True, residue_annotations=True, return_embeddings=True)
    sequence = (UBIQUITIN * ((args.length + 75) // 76))[:args.length]
    prompts = [sequence] + ["".join("_" if i % 7 == off else aa for i, aa in enumerate(sequence)) for off in range(3)]
    encoded = [model.encode(ESMProtein(sequence=p)).sequence for p in prompts]
    inputs = [_BatchedESMProteinTensor(sequence=torch.stack([encoded[(j+i) % len(encoded)] for j in range(args.batch)])) for i in range(len(encoded))]
    hosts = [item.sequence.cpu().pin_memory() for item in inputs]
    packages = {}
    for name in ("torch", "esm", "torchao", "flash-attn-4", "nvidia-cutlass-dsl", "triton", "nvidia-cudnn-cu13"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    report = {"status": "running", "gpu": torch.cuda.get_device_name(),
              "compute_capability": torch.cuda.get_device_capability(), "cuda": torch.version.cuda,
              "cudnn": torch.backends.cudnn.version(), "packages": packages,
              "model_revision": MODEL_REVISION, "esm_source": source_record(ESM3),
              "shape": list(inputs[0].sequence.shape), "batch": args.batch,
              "residue_length": args.length, "fixture": "ubiquitin" if args.length == 76 else "repeated ubiquitin synthetic workload",
              "input_bytes": hosts[0].numel() * hosts[0].element_size(),
              "protocol": "Four changing preencoded equal-shape inputs; pinned H2D copy included; all output heads+embeddings; individual synchronization; tokenization/staging/capture/output-D2H excluded",
              "warmup": args.warmup, "repeats": args.repeats}
    save(args, report)
    return model, config, inputs, hosts, report


def tensors(output):
    result = {name: getattr(output.logits, name) for name in TRACKS}
    result.update(residue_annotations=output.residue_annotation_logits, embeddings=output.embeddings)
    if any(value is None for value in result.values()):
        raise AssertionError("Requested head missing")
    return result


def snapshot(output):
    return {name: value.detach().float().cpu().clone() for name, value in tensors(output).items()}


def quality(output, expected, strict=False, sequence_mask=None):
    current = snapshot(output)
    result = {}
    for name, reference in expected.items():
        actual = current[name]
        if not torch.isfinite(actual).all():
            raise AssertionError(f"Nonfinite {name}")
        delta = actual - reference
        stats = {"exact": torch.equal(actual, reference), "max_absolute_error": delta.abs().max().item(),
                 "rmse": delta.square().mean().sqrt().item(),
                 "normalized_rmse": (delta.square().mean() / reference.square().mean().clamp_min(1e-20)).sqrt().item()}
        if name != "embeddings":
            stats["argmax_agreement"] = (actual.argmax(-1) == reference.argmax(-1)).float().mean().item()
        if name == "sequence":
            ref_logp = reference.log_softmax(-1)
            stats["mean_kl_reference_to_candidate"] = (ref_logp.exp() * (ref_logp - actual.log_softmax(-1))).sum(-1).mean().item()
            if sequence_mask is not None and sequence_mask.any():
                mask = sequence_mask.cpu()
                stats["masked_argmax_agreement"] = (actual.argmax(-1)[mask] == reference.argmax(-1)[mask]).float().mean().item()
                stats["masked_mean_kl_reference_to_candidate"] = (ref_logp.exp() * (ref_logp - actual.log_softmax(-1))).sum(-1)[mask].mean().item()
                stats["masked_positions"] = mask.sum().item()
        if strict:
            torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-3)
        result[name] = stats
    return result


def summarize(values):
    ordered = sorted(values)
    return {"median": statistics.median(ordered), "min": ordered[0], "max": ordered[-1],
            "p90": ordered[(9 * len(ordered) - 1) // 10], "samples": values}


def measure(run, args):
    for i in range(args.warmup):
        run(i % 4)
    torch.cuda.synchronize()
    wall = []
    for i in range(args.repeats):
        start = time.perf_counter_ns()
        run(i % 4)
        torch.cuda.synchronize()
        wall.append((time.perf_counter_ns() - start) / 1e6)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    events = []
    for i in range(args.repeats):
        start.record()
        run(i % 4)
        end.record()
        end.synchronize()
        events.append(start.elapsed_time(end))
    return {"wall_ms": summarize(wall), "cuda_ms": summarize(events),
            "requests_per_second": args.batch * 1000 / statistics.median(wall)}


def runners(model, config, hosts, args):
    from esm.utils.sampling import _BatchedESMProteinTensor
    static = _BatchedESMProteinTensor(sequence=hosts[0].to("cuda"))
    def eager(index):
        static.sequence.copy_(hosts[index], non_blocking=True)
        return model.logits(static, config)
    for i in range(args.warmup):
        eager(i % 4)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            model.logits(static, config)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = model.logits(static, config)
    def replay(index):
        static.sequence.copy_(hosts[index], non_blocking=True)
        graph.replay()
        return output
    return eager, replay, graph


def profile(run, args, label):
    for i in range(3):
        run(i % 4)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as trace:
        for i in range(3):
            run(i)
        torch.cuda.synchronize()
    trace_path = args.output.with_name(args.output.stem + f"-{label}-trace.json")
    trace.export_chrome_trace(str(trace_path))
    ops = [{"name": event.key, "count": event.count,
            "self_device_time_us": event.self_device_time_total} for event in trace.key_averages()]
    kernels = sorted({event.name for event in trace.events() if event.device_type == torch.autograd.DeviceType.CUDA})
    return {"trace": str(trace_path), "ops": ops, "kernels": kernels}


def evaluate(model, config, inputs, hosts, args, reference=None, label="baseline"):
    own = [snapshot(model.logits(item, config)) for item in inputs]
    if torch.equal(own[0]["sequence"], own[1]["sequence"]):
        raise AssertionError("Fresh inputs unexpectedly produced identical outputs")
    result = {}
    if reference is not None:
        result["quality_vs_bf16"] = [quality(model.logits(item, config), target, sequence_mask=(item.sequence == model.tokenizers.sequence.mask_token_id)) for item, target in zip(inputs, reference)]
    eager, replay, graph = runners(model, config, hosts, args)
    result["graph_correctness"] = [quality(replay(i % 4), own[i % 4], strict=True) for i in range(8)]
    result["eager_h2d"] = measure(eager, args)
    result["graph_h2d"] = measure(replay, args)
    return result, own, (eager, replay, graph)
