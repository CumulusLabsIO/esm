"""Real single-GPU ESM3 inference checks, including the reported dtype failure."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import time
import traceback

import torch
from huggingface_hub import snapshot_download

OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)
REVISION = '47f0545b2b6daf26a93439a3cd610f4f7f3d5478'
report = {'model_revision': REVISION, 'cases': [], 'environment': {
    'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
    'esm': metadata.version('esm'), 'python': __import__('sys').version,
    'visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
    'device_count': torch.cuda.device_count(), 'gpu': torch.cuda.get_device_name(0),
    'capability': list(torch.cuda.get_device_capability(0)),
    'flash_attn_installed': __import__('importlib.util', fromlist=['find_spec']).find_spec('flash_attn') is not None,
}}

def save():
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))

def case(name, fn, expected_failure=False):
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        result = fn()
        torch.cuda.synchronize()
        entry = {'name': name, 'status': 'pass', 'result': result}
    except Exception as e:
        entry = {'name': name, 'status': 'expected_failure' if expected_failure and 'dtype' in str(e) else 'fail',
                 'error': str(e), 'traceback': traceback.format_exc()}
    entry.update(seconds=time.perf_counter()-started,
                 allocated_gib=torch.cuda.memory_allocated()/2**30,
                 peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    report['cases'].append(entry)
    save()
    print(json.dumps(entry), flush=True)
    return entry

assert torch.cuda.device_count() == 1
assert 'B200' in torch.cuda.get_device_name(0)
save()
print(json.dumps(report['environment']), flush=True)
snapshot = snapshot_download('biohub/esm3-sm-open-v1', revision=REVISION)
# ESM's own local-asset switch resolves all data/ paths under this pinned snapshot.
os.chdir(snapshot)
os.environ['INFRA_PROVIDER'] = 'local-pinned-test'
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, ESMProteinError, GenerationConfig, LogitsConfig

model = ESM3.from_pretrained('esm3-open', device=torch.device('cuda')).eval()
report['environment']['parameter_dtype'] = str(next(model.parameters()).dtype)
report['environment']['attention_class'] = type(model.transformer.blocks[0].attn).__name__
save()
UBIQUITIN = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'
protein = ESMProtein(sequence=UBIQUITIN)
encoded = model.encode(protein)

def finite(tensor):
    assert torch.isfinite(tensor).all(), 'nonfinite output'
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'finite': True}

def direct(autocast):
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
        output = model(sequence_tokens=encoded.sequence.unsqueeze(0))
    return finite(output.sequence_logits)

case('direct_forward_without_autocast', lambda: direct(False), expected_failure=True)
case('direct_forward_with_bf16_autocast', lambda: direct(True))

def logits(length):
    sequence = (UBIQUITIN * ((length+len(UBIQUITIN)-1)//len(UBIQUITIN)))[:length]
    tensor = model.encode(ESMProtein(sequence=sequence))
    with torch.inference_mode():
        output = model.logits(tensor, LogitsConfig(sequence=True))
    return finite(output.logits.sequence)

for length in [76, 256, 512, 1024]:
    case(f'public_logits_length_{length}', lambda length=length: logits(length))

def generate_sequence():
    torch.manual_seed(2026)
    prompt = UBIQUITIN[:20] + '_'*8 + UBIQUITIN[28:]
    result = model.generate(ESMProtein(sequence=prompt), GenerationConfig(
        track='sequence', num_steps=8, temperature=0.7))
    assert not isinstance(result, ESMProteinError), str(result)
    assert result.sequence and len(result.sequence) == len(prompt)
    assert '_' not in result.sequence
    assert result.sequence[:20] == prompt[:20] and result.sequence[28:] == prompt[28:]
    (OUT/'generated-sequence.txt').write_text(result.sequence+'\n')
    return {'length': len(result.sequence), 'masked_residues_completed': 8, 'fixed_residues_preserved': True}

case('sequence_generation_8_steps', generate_sequence)

def generate_structure():
    torch.manual_seed(2026)
    result = model.generate(ESMProtein(sequence=UBIQUITIN), GenerationConfig(
        track='structure', num_steps=8))
    assert not isinstance(result, ESMProteinError), str(result)
    assert result.coordinates is not None
    assert result.coordinates.shape[0] == len(UBIQUITIN)
    backbone = result.coordinates[:, :3, :]
    info = finite(backbone)
    result.to_pdb(str(OUT/'ubiquitin-generated.pdb'))
    # This exercises the structure encoder as well as the decoder.
    encoded_structure = model.encode(result)
    assert encoded_structure.structure is not None
    info['reencoded_structure_tokens'] = len(encoded_structure.structure)
    info['pdb_bytes'] = (OUT/'ubiquitin-generated.pdb').stat().st_size
    return info

case('structure_generation_decode_and_reencode', generate_structure)

def batch_sequence():
    prompts = [ESMProtein(sequence=UBIQUITIN[:20]+'_'*8+UBIQUITIN[28:]) for _ in range(2)]
    outputs = model.batch_generate(prompts, [GenerationConfig(track='sequence', num_steps=8) for _ in prompts])
    assert len(outputs) == 2
    for output in outputs:
        assert not isinstance(output, ESMProteinError), str(output)
        assert output.sequence and len(output.sequence) == len(UBIQUITIN) and '_' not in output.sequence
    return {'batch_size': 2, 'length': len(UBIQUITIN), 'complete_sequences': 2}

case('batch_sequence_generation_2', batch_sequence)
for i in range(3):
    case(f'repeated_logits_512_{i+1}', lambda: logits(512))
report['passed'] = all(c['status'] != 'fail' for c in report['cases'])
save()
subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv'], check=False)
print('ALL_CHECKS_PASS=' + str(report['passed']), flush=True)
raise SystemExit(0 if report['passed'] else 1)
