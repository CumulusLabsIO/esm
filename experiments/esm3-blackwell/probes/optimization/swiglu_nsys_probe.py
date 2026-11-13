"""Profile already validated strict SwiGLU graph fusion; no latency benchmark."""

import json

import torch

from probe_common import parser, runners, save, setup, snapshot


@torch.inference_mode()
def main():
    args = parser(__doc__).parse_args()
    model, config, inputs, hosts, report = setup(args)
    report["scope"] = "Nsight capture of five baseline and five strict-SwiGLU CUDA graph replays; no latency benchmark"
    baseline_refs = [snapshot(model.logits(item, config)) for item in inputs]
    _eager, baseline, _baseline_graph = runners(model, config, hosts, args)
    for block in model.transformer.blocks:
        block.ffn[2] = torch.compile(block.ffn[2], fullgraph=True, dynamic=False,
                                    options={"triton.cudagraphs": False, "numerics": "strict"})
    for item, reference in zip(inputs, baseline_refs):
        for name, actual in snapshot(model.logits(item, config)).items():
            torch.testing.assert_close(actual, reference[name], rtol=0, atol=0)
    _eager, candidate, _candidate_graph = runners(model, config, hosts, args)
    for run in (baseline, candidate):
        for i, reference in enumerate(baseline_refs):
            for name, actual in snapshot(run(i)).items():
                torch.testing.assert_close(actual, reference[name], rtol=0, atol=0)
    report["all_head_bitwise_gate"] = True
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for name, run in (("BASELINE_GRAPH_REPLAYS_5", baseline), ("STRICT_SWIGLU_GRAPH_REPLAYS_5", candidate)):
        torch.cuda.nvtx.range_push(name)
        for i in range(5):
            run(i % 4)
            torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    report["status"] = "completed"
    save(args, report)
    print(json.dumps({"status": "completed", "all_head_bitwise_gate": True}), flush=True)


if __name__ == "__main__":
    main()
