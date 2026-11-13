"""Profile warmed CUDA graph nodes, excluding model loading and capture."""
import torch
from probe_common import parser, runners, setup, save, measure, quality, snapshot

@torch.inference_mode()
def main():
    args = parser(__doc__).parse_args()
    model, config, inputs, hosts, report = setup(args)
    references = [snapshot(model.logits(item, config)) for item in inputs]
    eager, replay, graph = runners(model, config, hosts, args)
    report['correctness'] = [quality(replay(i), ref, strict=True) for i, ref in enumerate(references)]
    report['graph_h2d'] = measure(replay, args)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for index in range(5):
        torch.cuda.nvtx.range_push('esm3_graph_replay')
        replay(index % 4)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    report['status'] = 'completed'
    save(args, report)

if __name__ == '__main__':
    main()
