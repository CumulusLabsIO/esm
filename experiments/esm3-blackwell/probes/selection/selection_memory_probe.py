"""Exact output-byte accounting and measured temporary allocations for selection."""

import ast
import gc
import importlib
from pathlib import Path
from types import MethodType

import torch

from probe_common import parser, save, setup
from selection_probe import selected


@torch.inference_mode()
def main():
    options = parser(__doc__)
    options.add_argument("--baseline-source", required=True, type=Path)
    args = options.parse_args()
    source = args.baseline_source.resolve().read_text()
    model, all_config, inputs, hosts, report = setup(args)
    cls = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "ESM3")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "logits")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "baseline-logits.py", "exec"), vars(importlib.import_module("esm.models.esm3")), namespace)
    baseline = MethodType(namespace["logits"], model)
    output = selected(baseline(inputs[0], all_config))
    report["float32_output_bytes"] = {key: value.numel() * value.element_size() for key, value in output.items()}
    report["output_shapes"] = {key: list(value.shape) for key, value in output.items()}
    del output
    from esm.sdk.api import LogitsConfig
    report["cases"] = {}
    for name, config in {"sequence_only": LogitsConfig(sequence=True), "embeddings_only": LogitsConfig(return_embeddings=True), "all_outputs": all_config}.items():
        result = {}
        for label, function in (("baseline_before", baseline), ("candidate", model.logits), ("baseline_after", baseline)):
            for _ in range(args.warmup):
                function(inputs[0], config)
            torch.cuda.synchronize()
            gc.collect()
            samples = []
            for _ in range(5):
                torch.cuda.reset_peak_memory_stats()
                before = torch.cuda.memory_allocated()
                output = function(inputs[0], config)
                torch.cuda.synchronize()
                samples.append(torch.cuda.max_memory_allocated() - before)
                del output
            result[label] = samples
        report["cases"][name] = result
    report["status"] = "completed"
    save(args, report)


if __name__ == "__main__":
    main()
