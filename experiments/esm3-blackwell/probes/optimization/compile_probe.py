#!/usr/bin/env python3
"""Measure extra compiler benefit beyond manual CUDA Graph replay on ESM3.

Use the pinned local Hugging Face snapshot and source checkout on PYTHONPATH.
No downloads, model/source edits, training, or sampling are performed. The four
inputs are fixed-size sequence tensors. Graph timings include the input copy.
"""

import argparse
import hashlib
import inspect
import json
import os
import statistics
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path

import torch
import torch._dynamo
import torch._inductor
from huggingface_hub import snapshot_download

REPOSITORY = "biohub/esm3-sm-open-v1"
REVISION = "47f0545b2b6daf26a93439a3cd610f4f7f3d5478"
UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
HEADS = ("sequence", "structure", "secondary_structure", "sasa", "function", "residue_annotations", "embeddings")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--nsys-range", choices=("eager_graph", "compiled_graph", "both"))
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--compile-mode", default="max-autotune-no-cudagraphs")
    parser.add_argument("--compile-target", choices=("full", "transformer"), default="full")
    parser.add_argument("--force-cudnn", action="store_true")
    return parser.parse_args()


def summary(values):
    ordered = sorted(values)
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "p90": ordered[int(0.9 * (len(ordered) - 1))], "samples": values}


def timing(run, count, args):
    for index in range(args.warmup):
        run(index % count)
    torch.cuda.synchronize()
    wall, device = [], []
    for index in range(args.repeats):
        start = time.perf_counter_ns()
        run(index % count)
        torch.cuda.synchronize()
        wall.append((time.perf_counter_ns() - start) / 1e6)
    for index in range(args.repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        run(index % count)
        end.record()
        end.synchronize()
        device.append(start.elapsed_time(end))
    return {"wall_ms": summary(wall), "cuda_event_ms": summary(device)}


def capture(run):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run()
    return graph, outputs


def cpu_copy(outputs):
    return tuple(value.detach().cpu().clone() for value in outputs)


def validate(outputs, references, rtol, atol):
    result = {}
    for name, actual, expected in zip(HEADS, cpu_copy(outputs), references, strict=True):
        error = (actual - expected).abs()
        close = torch.isclose(actual, expected, rtol=rtol, atol=atol)
        finite = bool(torch.isfinite(actual).all())
        result[name] = {
            "shape": list(actual.shape), "finite": finite,
            "exact": bool(torch.equal(actual, expected)),
            "within_tolerance": finite and bool(close.all()),
            "max_absolute_error": float(error.max()),
            "mean_absolute_error": float(error.mean()),
            "rms_error": float(error.square().mean().sqrt()),
            "outside_tolerance_fraction": float((~close).float().mean()),
            "argmax_agreement": float((actual.argmax(-1) == expected.argmax(-1)).float().mean()),
        }
    return result


def profile(run, destination):
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
        for index in range(4):
            with torch.profiler.record_function("esm3_graph_with_input_copy"):
                run(index % 4)
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(destination))
    return profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=25)


@torch.inference_mode()
def main():
    args = arguments()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "starting", "scope": "fixed-shape all-head inference; one CUDA-visible GPU; graph timings include input copy", "tolerance": {"rtol": args.rtol, "atol": args.atol}, "compile_mode": args.compile_mode, "compile_target": args.compile_target, "force_cudnn": args.force_cudnn, "inductor_numerics": torch._inductor.config.numerics, "emulate_precision_casts": torch._inductor.config.emulate_precision_casts, "max_autotune_gemm_backends": torch._inductor.config.max_autotune_gemm_backends, "fullgraph": True, "dynamic": False, "repeats": args.repeats, "warmup": args.warmup}

    def save(stage):
        report["stage"] = stage
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"stage": stage, "output": str(args.output)}), flush=True)

    try:
        torch.cuda.set_device(0)
        checkpoint = snapshot_download(REPOSITORY, revision=REVISION, local_files_only=True)
        os.chdir(checkpoint)
        os.environ["INFRA_PROVIDER"] = "local-pinned-test"
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.sampling import _BatchedESMProteinTensor

        source = Path(inspect.getfile(ESM3))
        report["environment"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "esm": version("esm"), "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)), "model_repository": REPOSITORY, "model_revision": REVISION, "model_source": str(source), "model_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "inductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"), "compile_threads": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"), "mode_options": torch._inductor.list_mode_options().get(args.compile_mode)}
        save("loading_model")
        model = ESM3.from_pretrained("esm3-open", device=torch.device("cuda:0")).eval()
        config = LogitsConfig(sequence=True, structure=True, secondary_structure=True, sasa=True, function=True, residue_annotations=True, return_embeddings=True)
        prompts = [UBIQUITIN] + ["".join("_" if i % 7 == offset else residue for i, residue in enumerate(UBIQUITIN)) for offset in range(3)]
        inputs = [_BatchedESMProteinTensor.from_protein_tensor(model.encode(ESMProtein(sequence=prompt))) for prompt in prompts]
        for item in inputs:
            assert all(getattr(item, name) is None for name in ("structure", "secondary_structure", "sasa", "function", "residue_annotations", "coordinates")), "This benchmark only supports sequence-only inputs."
        tokens = [item.sequence for item in inputs]
        host_tokens = [value.cpu().pin_memory() for value in tokens]
        static = tokens[0].clone()
        confidence = torch.ones((), device=static.device)
        report["input"] = {"shape": list(static.shape), "bytes": static.numel() * static.element_size(), "changing_inputs": len(tokens), "sequence_sha256": [hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts]}

        def public(sequence):
            output = model.logits(_BatchedESMProteinTensor(sequence=sequence), config)
            return (output.logits.sequence, output.logits.structure, output.logits.secondary_structure, output.logits.sasa, output.logits.function, output.residue_annotation_logits, output.embeddings)

        def forward(sequence):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(sequence_tokens=sequence, average_plddt=confidence)
                return (output.sequence_logits.float(), output.structure_logits.float(), output.secondary_structure_logits.float(), output.sasa_logits.float(), output.function_logits.float(), output.residue_logits.float(), output.embeddings.float())

        references = [cpu_copy(public(value)) for value in tokens]
        assert not torch.equal(references[0][0], references[1][0]), "Fixture did not change the output."
        report["wrapper_correctness"] = [validate(forward(value), reference, 0, 0) for value, reference in zip(tokens, references, strict=True)]
        assert all(head["exact"] for item in report["wrapper_correctness"] for head in item.values()), "Raw wrapper must exactly match the public logits API."
        # This records the existing raw-forward dtype issue separately; benchmark
        # execution uses the same BF16 autocast as public logits throughout.
        try:
            model(sequence_tokens=tokens[0], average_plddt=confidence)
            torch.cuda.synchronize()
            report["raw_forward_without_autocast"] = {"status": "passed"}
        except RuntimeError as error:
            report["raw_forward_without_autocast"] = {"status": "error", "message": str(error)}

        save("eager_baselines")
        report["eager_public"] = timing(lambda index: public(tokens[index]), len(tokens), args)
        report["eager_wrapper"] = timing(lambda index: forward(tokens[index]), len(tokens), args)
        eager_graph, eager_output = capture(lambda: forward(static))

        def eager_graph_run(index, host=True):
            static.copy_(host_tokens[index] if host else tokens[index], non_blocking=True)
            eager_graph.replay()
            return eager_output

        report["eager_graph_correctness"] = [validate(eager_graph_run(index), reference, 0, 0) for index, reference in enumerate(references)]
        assert all(head["exact"] for item in report["eager_graph_correctness"] for head in item.values())
        report["eager_graph_host_before"] = timing(eager_graph_run, len(tokens), args)
        report["eager_graph_device_before"] = timing(lambda index: eager_graph_run(index, False), len(tokens), args)
        public_graph, public_graph_output = capture(lambda: public(static))

        def public_graph_run(index):
            static.copy_(host_tokens[index], non_blocking=True)
            public_graph.replay()
            return public_graph_output

        report["public_graph_host"] = timing(public_graph_run, len(tokens), args)
        save("compiling")
        torch._dynamo.reset()
        if args.compile_target == "transformer":
            model.transformer = torch.compile(model.transformer, mode=args.compile_mode, fullgraph=True, dynamic=False)
            compiled = forward
        else:
            compiled = torch.compile(forward, mode=args.compile_mode, fullgraph=True, dynamic=False)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        compiled(static)
        torch.cuda.synchronize()
        report["first_compiled_call_seconds"] = time.perf_counter() - started
        report["compilation_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        save("validating_compiled")
        report["compiled_correctness"] = [validate(compiled(value), reference, args.rtol, args.atol) for value, reference in zip(tokens, references, strict=True)]
        report["compiled_eager"] = timing(lambda index: compiled(tokens[index]), len(tokens), args)
        compiled_graph, compiled_output = capture(lambda: compiled(static))

        def compiled_graph_run(index, host=True):
            static.copy_(host_tokens[index] if host else tokens[index], non_blocking=True)
            compiled_graph.replay()
            return compiled_output

        report["compiled_graph_correctness"] = [validate(compiled_graph_run(index), reference, args.rtol, args.atol) for index, reference in enumerate(references)]
        report["compiled_graph_matches_compiled"] = [validate(compiled_graph_run(index), cpu_copy(compiled(tokens[index])), 0, 0) for index in range(len(tokens))]
        report["compiled_graph_host"] = timing(compiled_graph_run, len(tokens), args)
        report["compiled_graph_device"] = timing(lambda index: compiled_graph_run(index, False), len(tokens), args)
        report["eager_graph_host_after"] = timing(eager_graph_run, len(tokens), args)
        report["correctness_passed"] = all(head["within_tolerance"] for key in ("compiled_correctness", "compiled_graph_correctness", "compiled_graph_matches_compiled") for item in report[key] for head in item.values())
        report["dynamo_counters"] = {name: dict(values) for name, values in torch._dynamo.utils.counters.items() if values}
        baseline = statistics.mean(report[name]["wall_ms"]["median"] for name in ("eager_graph_host_before", "eager_graph_host_after"))
        optimized = report["compiled_graph_host"]["wall_ms"]["median"]
        report["gain"] = {"baseline_mean_of_medians_ms": baseline, "compiled_graph_ms": optimized, "speedup": baseline / optimized, "latency_reduction_percent": 100 * (1 - optimized / baseline), "cold_call_amortization_requests": report["first_compiled_call_seconds"] * 1000 / (baseline - optimized) if baseline > optimized else None}
        save("profiling" if args.profile else "complete")
        if args.profile:
            report["profiles"] = {}
            for label, run in (("eager_graph", eager_graph_run), ("compiled_graph", compiled_graph_run)):
                path = args.output.with_name(args.output.stem + "." + label + ".trace.json")
                report["profiles"][label] = {"trace": str(path), "summary": profile(run, path)}
        if args.nsys_range:
            for label, run in (("eager_graph", eager_graph_run), ("compiled_graph", compiled_graph_run)):
                if args.nsys_range not in (label, "both"):
                    continue
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                with torch.cuda.nvtx.range(label):
                    for index in range(3):
                        run(index % len(tokens))
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
        report["status"] = "passed" if report["correctness_passed"] else "numerical_mismatch"
        save("complete")
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        save(report.get("stage", "unknown"))
        raise


if __name__ == "__main__":
    if "--force-cudnn" in sys.argv:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            main()
    else:
        main()
