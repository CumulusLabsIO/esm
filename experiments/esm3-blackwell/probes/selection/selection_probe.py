"""Matched-config A-B-A probe for skipping unrequested ESM3 output conversions."""

import ast
import hashlib
import importlib
import traceback
from pathlib import Path
from types import MethodType

import torch

from probe_common import TRACKS, measure, parser, profile, save, setup


def selected(output):
    values = {name: getattr(output.logits, name) for name in TRACKS}
    values.update(embeddings=output.embeddings, residue_annotations=output.residue_annotation_logits)
    return {key: value for key, value in values.items() if isinstance(value, torch.Tensor)}


def snapshot(output):
    return {key: value.detach().cpu().clone() for key, value in selected(output).items()}


def verify(output, expected, exact):
    values = snapshot(output)
    assert values.keys() == expected.keys()
    result = {}
    for key, reference in expected.items():
        value = values[key]
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, reference, rtol=0 if exact else 1e-3, atol=0 if exact else 1e-3)
        result[key] = {"exact": torch.equal(value, reference), "max_absolute_error": (value - reference).abs().max().item()}
    return result


def runners(method, config, hosts, args):
    from esm.utils.sampling import _BatchedESMProteinTensor
    static = _BatchedESMProteinTensor(sequence=hosts[0].to("cuda"))

    def eager(index):
        static.sequence.copy_(hosts[index], non_blocking=True)
        return method(static, config)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(args.warmup):
            method(static, config)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = method(static, config)

    def replay(index):
        static.sequence.copy_(hosts[index], non_blocking=True)
        graph.replay()
        return output

    return eager, replay, graph


def measurements(run, args):
    eager, graph, _ = run
    return {"eager_h2d": measure(eager, args), "graph_h2d": measure(graph, args)}


@torch.inference_mode()
def main():
    options = parser(__doc__)
    options.add_argument("--baseline-source", required=True, type=Path)
    args = options.parse_args()
    args.baseline_source = args.baseline_source.resolve()
    report = {"status": "failed"}
    try:
        model, all_config, inputs, hosts, report = setup(args)
        report["protocol"] = "Matched output configuration A-B-A: changing equal-shape inputs, pinned H2D included, separate wall/CUDA timings; tokenization/capture/output-D2H excluded. All heads still compute; only output conversions are selected."
        source = args.baseline_source.read_text()
        class_node = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "ESM3")
        function = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "logits")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(args.baseline_source), "exec"), vars(importlib.import_module("esm.models.esm3")), namespace)
        baseline = MethodType(namespace["logits"], model)
        candidate = model.logits
        report["baseline_source"] = {"path": str(args.baseline_source), "sha256": hashlib.sha256(source.encode()).hexdigest()}
        from esm.sdk.api import LogitsConfig
        configs = {"sequence_only": LogitsConfig(sequence=True), "embeddings_only": LogitsConfig(return_embeddings=True), "all_outputs": all_config}
        report["cases"] = {}
        for name, config in configs.items():
            references = [snapshot(baseline(item, config)) for item in inputs]
            result = {"eager_correctness": [verify(candidate(item, config), reference, exact=True) for item, reference in zip(inputs, references)]}
            baseline_runners = runners(baseline, config, hosts, args)
            candidate_runners = runners(candidate, config, hosts, args)
            result["graph_correctness"] = [verify(candidate_runners[1](i % 4), references[i % 4], exact=False) for i in range(8)]
            result["baseline_before"] = measurements(baseline_runners, args)
            result["candidate"] = measurements(candidate_runners, args)
            result["baseline_after"] = measurements(baseline_runners, args)
            if not args.skip_profile:
                result["baseline_profile"] = profile(baseline_runners[0], args, name + "-baseline")
                result["candidate_profile"] = profile(candidate_runners[0], args, name + "-candidate")
            report["cases"][name] = result
            save(args, report)
            del baseline_runners, candidate_runners
        empty_config = LogitsConfig()
        report["empty_config_correctness"] = [verify(candidate(item, empty_config), snapshot(baseline(item, empty_config)), exact=True) for item in inputs]
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        save(args, report)


if __name__ == "__main__":
    main()
