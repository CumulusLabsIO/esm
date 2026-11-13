"""Categorize Nsight Systems kernel CSVs by name; preserve raw traces."""
import csv
import io
import json
import sys
from pathlib import Path

def main():
    source, replays, destination = sys.argv[1:]
    replays = int(replays)
    content = Path(source).read_text()
    rows = list(csv.DictReader(io.StringIO(content[content.index('Time (%),'):])))
    groups = {}
    for row in rows:
        name = row['Name'].lower()
        if any(key in name for key in ('nvjet', 'gemm', 'gemv', 'cutlass')):
            category = 'matmul'
        elif 'flash' in name or 'fmha' in name:
            category = 'attention'
        elif 'layer_norm' in name or 'layernorm' in name:
            category = 'normalization'
        elif 'converting_copy' in name:
            category = 'casts'
        elif any(key in name for key in ('copy', 'catarray')):
            category = 'copy_concat'
        elif 'triton_' in name:
            category = 'triton_fused_or_other'
        else:
            category = 'elementwise_and_other'
        item = groups.setdefault(category, {'ns': 0, 'kernels': 0})
        item['ns'] += int(row['Total Time (ns)'])
        item['kernels'] += int(row['Instances'])
    total = sum(item['ns'] for item in groups.values())
    for item in groups.values():
        item.update(share=item['ns'] / total, per_replay_ms=item['ns'] / replays / 1e6,
                    kernels_per_replay=item['kernels'] / replays)
    report = {'replays': replays, 'total_kernel_ms_per_replay': total / replays / 1e6,
              'kernels_per_replay': sum(item['kernels'] for item in groups.values()) / replays,
              'groups': groups,
              'classification': 'Kernel-name heuristic; summed traced durations, not uninstrumented wall latency'}
    Path(destination).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
