"""Compare strict-numerics FFN compilation against uncompiled ESM3 outputs."""

import traceback

import torch

from probe_common import evaluate, measure, parser, save, setup, snapshot


@torch.inference_mode()
def main():
    options = parser(__doc__)
    options.add_argument("--scope", choices=("ffn", "swiglu"), default="ffn")
    args = options.parse_args()
    report = {"status": "failed"}
    try:
        model, config, inputs, hosts, report = setup(args)
        baseline, references, baseline_runners = evaluate(model, config, inputs, hosts, args)
        report["baseline_before"] = baseline
        save(args, report)
        originals = [block.ffn if args.scope == "ffn" else block.ffn[2] for block in model.transformer.blocks]
        report["compile_scope"] = args.scope
        compile_options = {"triton.cudagraphs": False, "numerics": "strict"}
        report["compile_options"] = compile_options
        for block, original in zip(model.transformer.blocks, originals):
            compiled = torch.compile(original, fullgraph=True, dynamic=False, options=compile_options)
            if args.scope == "ffn":
                block.ffn = compiled
            else:
                block.ffn[2] = compiled
        candidate, _, _runners = evaluate(model, config, inputs, hosts, args, references, label="strict-ffn")
        report["candidate"] = candidate
        report["raw_bf16_gate"] = {"rtol": 1e-2, "atol": 1e-2, "inputs": []}
        all_pass = True
        for item, reference in zip(inputs, references):
            current = snapshot(model.logits(item, config))
            checks = {}
            for name, expected in reference.items():
                try:
                    torch.testing.assert_close(current[name], expected, rtol=1e-2, atol=1e-2)
                    checks[name] = {"passed": True}
                except AssertionError as error:
                    checks[name] = {"passed": False, "error": str(error)}
                    all_pass = False
            report["raw_bf16_gate"]["inputs"].append(checks)
        report["raw_bf16_gate"]["all_passed"] = all_pass
        for block, original in zip(model.transformer.blocks, originals):
            if args.scope == "ffn":
                block.ffn = original
            else:
                block.ffn[2] = original
        report["baseline_after"] = {"eager_h2d": measure(baseline_runners[0], args), "graph_h2d": measure(baseline_runners[1], args)}
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        save(args, report)


if __name__ == "__main__":
    main()
