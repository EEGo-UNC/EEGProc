"""Build a read-only, memory-mappable concatenation of unchanged EEG shards."""
import json
from pathlib import Path
import numpy as np

root=Path('/work/users/p/m/pmucharl/sic-v15-seed-iv/processed')
paths=[root/f'subject_{s:02}/X.npy' for s in range(1,16)]
shapes=[np.load(p,mmap_mode='r').shape for p in paths]
total=sum(s[0] for s in shapes)
assert all(s[1:]==(800,310) for s in shapes)
target=root/'all_windows.npy'
temporary=root/'all_windows.building.npy'
out=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=(total,800,310))
offset=0
for p in paths:
    data=np.load(p,mmap_mode='r')
    for start in range(0,len(data),32):
        end=min(start+32,len(data))
        out[offset+start:offset+end]=data[start:end]
    offset+=len(data)
    print('COPIED',p,'windows=',offset,flush=True)
out.flush()
del out
temporary.replace(target)
check=np.load(target,mmap_mode='r')
offset=0
for p in paths:
    original=np.load(p,mmap_mode='r')
    for start in range(0,len(original),32):
        end=min(start+32,len(original))
        assert np.array_equal(check[offset+start:offset+end],original[start:end])
    offset+=len(original)
(root/'all_windows_verified.json').write_text(json.dumps(dict(shape=list(check.shape),exact_shard_comparison=True)))
print('CACHE_VERIFIED',check.shape,flush=True)
