"""Selective FFN MXFP8: performance and approximate output-quality experiment."""

import copy
import inspect
import traceback

import torch

from probe_common import evaluate, measure, parser, profile, save, setup, source_record


@torch.inference_mode()
def main():
    options = parser(__doc__)
    options.add_argument("--compile-ffn", action="store_true", help="Compile FFNs in both BF16 control and MXFP8 candidate, disabling Inductor CUDA graphs because this harness captures explicitly")
    options.add_argument("--allow-small-m", action="store_true", help="Explicitly test below the published M>=1024 profitability heuristic")
    args = options.parse_args()
    report = {"status": "failed"}
    try:
        model, config, inputs, hosts, report = setup(args)
        if torch.cuda.get_device_capability()[0] != 10:
            raise RuntimeError("This experiment targets Blackwell SM10x")
        m = args.batch * (args.length + 2)
        report["M"] = m
        report["small_shape_heuristic"] = "Diffusion study heuristic, not an ESM3 correctness or performance guarantee"
        if m < 1024 and not args.allow_small_m:
            raise RuntimeError("M<1024: pass --allow-small-m to explicitly test this likely unprofitable regime")
        quantized = copy.deepcopy(model)
        report["compile_ffn"] = args.compile_ffn
        if args.compile_ffn:
            for block in model.transformer.blocks:
                block.ffn = torch.compile(block.ffn, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
        baseline, references, baseline_runners = evaluate(model, config, inputs, hosts, args)
        report["baseline_before"] = baseline
        save(args, report)
        from torchao.prototype.mx_formats.inference_workflow import MXDynamicActivationMXWeightConfig
        from torchao.quantization import quantize_
        from torchao.quantization.quant_api import KernelPreference
        report["torchao_source"] = source_record(MXDynamicActivationMXWeightConfig)
        report["config_signature"] = str(inspect.signature(MXDynamicActivationMXWeightConfig))
        selected = []
        def select(module, fqn):
            chosen = isinstance(module, torch.nn.Linear) and ".ffn." in fqn and fqn.startswith("transformer.")
            if chosen:
                selected.append({"name": fqn, "weight_shape": list(module.weight.shape), "weight_dtype": str(module.weight.dtype)})
            return chosen
        quantize_(quantized, MXDynamicActivationMXWeightConfig(activation_dtype=torch.float8_e4m3fn,
                  weight_dtype=torch.float8_e4m3fn, kernel_preference=KernelPreference.AUTO), filter_fn=select)
        if not selected:
            raise RuntimeError("No FFN linears selected; refuse a no-op quantization result")
        report["selected_linears"] = selected
        transformed = {name: type(module.weight).__name__ for name, module in quantized.named_modules() if any(entry["name"] == name for entry in selected)}
        report["transformed_weight_types"] = transformed
        if any(name == "Parameter" for name in transformed.values()):
            raise RuntimeError("Selected weight retained ordinary Parameter type")
        if args.compile_ffn:
            for block in quantized.transformer.blocks:
                block.ffn = torch.compile(block.ffn, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
        candidate, _, _candidate_runners = evaluate(quantized, config, inputs, hosts, args, references, label="mxfp8")
        report["candidate"] = candidate
        report["quality_scope"] = "All-head numeric errors, argmax agreement and sequence KL; no downstream biological-quality or training validation"
        report["baseline_after"] = {"eager_h2d": measure(baseline_runners[0], args), "graph_h2d": measure(baseline_runners[1], args)}
        if not args.skip_profile:
            report["baseline_before"]["profile"] = profile(baseline_runners[0], args, "baseline-eager-after-timing")
            candidate["profile"] = profile(_candidate_runners[0], args, "mxfp8-eager-after-timing")
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        save(args, report)


if __name__ == "__main__":
    main()
