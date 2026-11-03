# 27.10: чем отличаются два .rwkvq — манифест (верхний уровень и записи тензоров) и буферы.
import os, sys, json, numpy as np
sys.path.insert(0, os.environ.get('RWKV_QUANT', '../rwkv-quant'))
from rwkv_quant.formats.codec import open_rwkvq
a, b = sys.argv[1], sys.argv[2]
ma, xa = open_rwkvq(a); mb, xb = open_rwkvq(b)
top = sorted(set(ma) | set(mb) - {'tensors'})
for k in top:
    if k == 'tensors': continue
    if ma.get(k) != mb.get(k): print('манифест', k, ':', json.dumps(ma.get(k))[:200], '->', json.dumps(mb.get(k))[:200])
ta, tb = ma['tensors'], mb['tensors']
print('тензоров', len(ta), len(tb), 'только в A', sorted(set(ta)-set(tb))[:5], 'только в B', sorted(set(tb)-set(ta))[:5])
ent = [k for k in ta if k in tb and ta[k] != tb[k]]
print('записей манифеста отличается', len(ent), ent[:5])
for k in ent[:3]:
    d = {f: (ta[k].get(f), tb[k].get(f)) for f in set(ta[k]) | set(tb[k]) if ta[k].get(f) != tb[k].get(f)}
    print('  ', k, json.dumps(d)[:300])
diff = []
for k in sorted(set(xa) & set(xb)):
    u, v = np.asarray(xa[k]), np.asarray(xb[k])
    if u.shape != v.shape or u.dtype != v.dtype or not np.array_equal(u.view(np.uint8) if u.dtype.kind == 'V' else u, v.view(np.uint8) if v.dtype.kind == 'V' else v):
        n = u.size if u.shape == v.shape else -1
        frac = float(np.mean(u.reshape(-1).view(np.uint8) != v.reshape(-1).view(np.uint8))) if u.shape == v.shape and u.dtype == v.dtype else -1
        diff.append((k, str(u.dtype), u.shape, frac))
print('буферов', len(xa), len(xb), 'отличается', len(diff))
groups = {}
for k, *_ in diff:
    g = k.split('::')[-1]; groups[g] = groups.get(g, 0) + 1
print('по полям', groups)
for r in diff[:8]: print('  ', r)
