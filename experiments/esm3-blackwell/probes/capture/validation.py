"""Validate public ESM3 logits graph replay with changing, equal-shape inputs."""

import argparse
import hashlib
import inspect
import json
import os
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import torch
from huggingface_hub import snapshot_download


MODEL_REPOSITORY = "biohub/esm3-sm-open-v1"
MODEL_REVISION = "47f0545b2b6daf26a93439a3cd610f4f7f3d5478"
UBIQUITIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)
TRACKS = ("sequence", "structure", "secondary_structure", "sasa", "function")


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("eager", "capture-probe", "graph"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 3 or args.repeats < 1:
        parser.error("warmup must be at least three and repeats must be positive")
    args.output = args.output.resolve()
    if args.reference:
        args.reference = args.reference.resolve()
    return args


def tensors(output):
    result = {name: getattr(output.logits, name) for name in TRACKS}
    result["residue_annotations"] = output.residue_annotation_logits
    result["embeddings"] = output.embeddings
    if any(value is None for value in result.values()):
        raise AssertionError("A requested output head is missing")
    return result


def snapshot(output):
    return {
        name: value.detach().cpu().clone() for name, value in tensors(output).items()
    }


def compare(output, reference):
    current = snapshot(output)
    differences = {}
    for name, expected in reference.items():
        actual = current[name]
        if not torch.isfinite(actual).all():
            raise AssertionError(f"Nonfinite output in {name}")
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
        differences[name] = {
            "exact": torch.equal(actual, expected),
            "max_absolute_error": (actual - expected).abs().max().item(),
        }
    return differences


def summarize(samples):
    ordered = sorted(samples)
    return {
        "median": statistics.median(ordered),
        "p90": ordered[(9 * len(ordered) - 1) // 10],
        "min": ordered[0],
        "max": ordered[-1],
        "samples": samples,
    }


def measure(run, args, count):
    for index in range(args.warmup):
        run(index % count)
    torch.cuda.synchronize()
    wall = []
    for index in range(args.repeats):
        started = time.perf_counter_ns()
        output = run(index % count)
        torch.cuda.synchronize()
        wall.append((time.perf_counter_ns() - started) / 1_000_000)
        del output
    # Measure CUDA events separately so instrumentation does not inflate wall latency.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    device = []
    for index in range(args.repeats):
        start.record()
        output = run(index % count)
        end.record()
        end.synchronize()
        device.append(start.elapsed_time(end))
        del output
    return {"wall_ms": summarize(wall), "cuda_event_ms": summarize(device)}


def capture(run):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    return graph, output


@torch.inference_mode()
def main():
    args = options()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    torch.cuda.set_device(0)
    model_path = Path(
        snapshot_download(
            MODEL_REPOSITORY, revision=MODEL_REVISION, local_files_only=True
        )
    )
    os.chdir(model_path)
    os.environ["INFRA_PROVIDER"] = "local-pinned-test"
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, LogitsConfig
    from esm.utils.sampling import _BatchedESMProteinTensor

    model = ESM3.from_pretrained("esm3-open", device=torch.device("cuda:0")).eval()
    config = LogitsConfig(
        sequence=True,
        structure=True,
        secondary_structure=True,
        sasa=True,
        function=True,
        residue_annotations=True,
        return_embeddings=True,
    )
    # Four distinct masks of a benign fixture exercise changing inputs without generation.
    prompts = [UBIQUITIN]
    for offset in (0, 1, 2):
        prompts.append(
            "".join(
                "_" if index % 7 == offset else residue
                for index, residue in enumerate(UBIQUITIN)
            )
        )
    inputs = [
        _BatchedESMProteinTensor.from_protein_tensor(
            model.encode(ESMProtein(sequence=prompt))
        )
        for prompt in prompts
    ]
    # Sequence-only encoding is deliberate. Every input has the same metadata and shape.
    for item in inputs:
        if any(
            getattr(item, name) is not None
            for name in (*TRACKS[1:], "residue_annotations", "coordinates")
        ):
            raise AssertionError("Unexpected non-sequence input track")
    host_inputs = [item.sequence.cpu().pin_memory() for item in inputs]
    static = _BatchedESMProteinTensor(sequence=inputs[0].sequence.clone())
    eager_static = _BatchedESMProteinTensor(sequence=inputs[0].sequence.clone())
    source = Path(inspect.getfile(ESM3))
    report = {
        "mode": args.mode,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "esm": version("esm"),
        "source_path": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model_revision": MODEL_REVISION,
        "input_shape": list(static.sequence.shape),
        "input_bytes": static.sequence.numel() * static.sequence.element_size(),
        "input_count": len(inputs),
        "output_heads": [*TRACKS, "residue_annotations", "embeddings"],
        "warmup": args.warmup,
        "repeats": args.repeats,
        "scope": [
            "Public logits API with all six output heads and embeddings requested.",
            "Inputs are preencoded, equally shaped masks of the 76-residue ubiquitin fixture.",
            "Copy paths include input transfer, replay and completion synchronization.",
            "Host tensors are pinned and filled before timing. Tokenization, host staging, graph capture and output D2H are excluded.",
            "Every measured request is synchronized individually. These are latency measurements.",
        ],
    }
    if args.mode == "capture-probe":
        try:
            capture(lambda: model.logits(static, config))
        except RuntimeError as error:
            report["capture_succeeded"] = False
            report["capture_error"] = str(error)
        else:
            report["capture_succeeded"] = True
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return

    references = [snapshot(model.logits(item, config)) for item in inputs]
    if not all(
        not torch.equal(references[0]["sequence"], reference["sequence"])
        for reference in references[1:]
    ):
        raise AssertionError("Changing inputs failed to produce distinct outputs")
    torch.save(references, args.output.with_suffix(".pt"))
    if args.reference:
        baseline = torch.load(args.reference, map_location="cpu", weights_only=True)
        if len(baseline) != len(inputs):
            raise AssertionError("Reference input count changed")
        report["comparison_to_stock"] = [
            compare(model.logits(item, config), reference)
            for item, reference in zip(inputs, baseline)
        ]

    def eager_device(index):
        return model.logits(inputs[index], config)

    def eager_host(index):
        eager_static.sequence.copy_(host_inputs[index], non_blocking=True)
        return model.logits(eager_static, config)

    paths = [("eager_device_before", eager_device), ("eager_host_before", eager_host)]
    if args.mode == "graph":
        graph, static_output = capture(lambda: model.logits(static, config))

        def graph_device(index):
            static.sequence.copy_(inputs[index].sequence, non_blocking=True)
            graph.replay()
            return static_output

        def graph_host(index):
            static.sequence.copy_(host_inputs[index], non_blocking=True)
            graph.replay()
            return static_output

        def graph_fixed(_index):
            graph.replay()
            return static_output

        # Repeat beyond one input cycle to detect stale buffers and incorrect reuse.
        report["graph_correctness"] = {}
        for name, run in (
            ("device_copy", graph_device),
            ("pinned_host_copy", graph_host),
        ):
            report["graph_correctness"][name] = [
                compare(run(index % len(inputs)), references[index % len(inputs)])
                for index in range(3 * len(inputs))
            ]
        paths += [
            ("graph_fixed_before", graph_fixed),
            ("graph_device_copy", graph_device),
            ("graph_pinned_host_copy", graph_host),
            ("graph_fixed_after", graph_fixed),
        ]
    paths += [("eager_host_after", eager_host), ("eager_device_after", eager_device)]
    report["timings"] = {}
    for name, run in paths:
        if name.startswith("graph_fixed"):
            static.sequence.copy_(inputs[0].sequence)
            compare(run(0), references[0])
        report["timings"][name] = measure(run, args, len(inputs))
        print(name, report["timings"][name]["wall_ms"]["median"], "ms", flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["completed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
