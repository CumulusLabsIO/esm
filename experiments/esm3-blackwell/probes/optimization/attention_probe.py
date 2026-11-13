"""A-B-A all-head ESM3 logits experiment for FA4 or cuDNN SDPA."""

import importlib
import inspect
import traceback
from contextlib import nullcontext

import torch

from probe_common import evaluate, measure, parser, profile, save, setup, source_record


@torch.inference_mode()
def main():
    options = parser(__doc__)
    options.add_argument("--backend", choices=("default", "cudnn", "fa4"), required=True)
    args = options.parse_args()
    report = {"status": "failed", "backend": args.backend}
    try:
        model, config, inputs, hosts, report = setup(args)
        report["backend"] = args.backend
        baseline, references, baseline_runners = evaluate(model, config, inputs, hosts, args)
        report["baseline_before"] = baseline
        save(args, report)
        import torch.nn.attention as attention
        active = getattr(attention, "current_flash_attention_impl", lambda: None)()
        if active is not None:
            raise RuntimeError(f"Unexpected preactivated Flash implementation {active}")
        if args.backend == "fa4":
            required = ("activate_flash_attention_impl", "restore_flash_attention_impl", "current_flash_attention_impl")
            if not all(hasattr(attention, name) for name in required):
                raise RuntimeError("Installed torch lacks the public FA4 registry APIs")
            implementation = importlib.import_module("flash_attn.cute.interface")
            report["fa4_source"] = source_record(implementation)
            report["torch_fa4_source"] = source_record(importlib.import_module("torch.nn.attention._fa4"))
            report["fa4_signature"] = str(inspect.signature(implementation._flash_attn_fwd))
            attention.activate_flash_attention_impl("FA4")
            if attention.current_flash_attention_impl() != "FA4":
                raise RuntimeError("FA4 was not activated")
            context = attention.sdpa_kernel(attention.SDPBackend.FLASH_ATTENTION)
        elif args.backend == "cudnn":
            context = attention.sdpa_kernel(attention.SDPBackend.CUDNN_ATTENTION)
        else:
            context = nullcontext()
        try:
            with context:
                report["active_flash_impl"] = getattr(attention, "current_flash_attention_impl", lambda: None)()
                candidate, _, _candidate_runners = evaluate(model, config, inputs, hosts, args, references, label=args.backend)
                report["candidate"] = candidate
        finally:
            if args.backend == "fa4":
                attention.restore_flash_attention_impl()
        report["baseline_after"] = {"eager_h2d": measure(baseline_runners[0], args), "graph_h2d": measure(baseline_runners[1], args)}
        if not args.skip_profile:
            report["baseline_before"]["profile"] = profile(baseline_runners[0], args, "baseline-eager-after-timing")
            if args.backend == "fa4":
                attention.activate_flash_attention_impl("FA4")
                profile_context = attention.sdpa_kernel(attention.SDPBackend.FLASH_ATTENTION)
            elif args.backend == "cudnn":
                profile_context = attention.sdpa_kernel(attention.SDPBackend.CUDNN_ATTENTION)
            else:
                profile_context = nullcontext()
            try:
                with profile_context:
                    candidate["profile"] = profile(_candidate_runners[0], args, f"{args.backend}-eager-after-timing")
            finally:
                if args.backend == "fa4":
                    attention.restore_flash_attention_impl()
            ops = [item["name"] for item in candidate["profile"]["ops"]]
            expected = "cudnn" if args.backend == "cudnn" else "flash"
            report["backend_operator_confirmed"] = any(expected in name.lower() and "attention" in name.lower() for name in ops)
            if args.backend != "default" and not report["backend_operator_confirmed"]:
                raise RuntimeError("Requested attention operator not observed in profiler")
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        save(args, report)


if __name__ == "__main__":
    main()
