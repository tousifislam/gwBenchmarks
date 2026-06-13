"""Validate the tau-representation + SVD reconstruction floor.

Two error sources we must quantify before any modelling:
  (a) round-trip error: wave -> tau-grid -> back to original t grid.
  (b) SVD truncation error at various ranks (no parameter fit yet).
If (a) is already comparable to the NR floor (1.4e-3) the representation
is unusable; if it is well below, we are good to model.
"""
import time
import numpy as np
import common as C

t0 = time.time()
print("loading training...")
P, waves = C.load_split("training")
print(f"  {len(waves)} waves in {time.time()-t0:.1f}s")

logA, phi, ends = C.build_tau_matrices("training")
print("tau matrices", logA.shape)

# (a) round-trip floor on 20 waveforms
idx = np.arange(0, 250, 12)[:20]
rt_pred, rt_true = [], []
for i in idx:
    t, h = waves[i]
    la, ph, a, b = C.wave_to_tau(t, h)
    hp = C.tau_to_wave(la, ph, t, a, b)
    rt_pred.append((t, hp)); rt_true.append((t, h))
rt = C.score_waveforms(rt_pred, rt_true)
print(f"(a) round-trip mismatch: median={np.median(rt):.2e} max={rt.max():.2e}")

# (b) SVD truncation reconstruction at several ranks
muA = logA.mean(0); muP = phi.mean(0)
UA, sA, VA = np.linalg.svd(logA - muA, full_matrices=False)
UP, sP, VP = np.linalg.svd(phi - muP, full_matrices=False)
print("singular value decay (logA):", np.round(sA[:8], 2))
print("singular value decay (phi): ", np.round(sP[:8], 2))

for k in [5, 10, 20, 30]:
    la_r = muA + (UA[:, :k] * sA[:k]) @ VA[:k]
    ph_r = muP + (UP[:, :k] * sP[:k]) @ VP[:k]
    pred, true = [], []
    for j, i in enumerate(idx):
        t, h = waves[i]
        hp = C.tau_to_wave(la_r[i], ph_r[i], t, ends[i, 0], ends[i, 1])
        pred.append((t, hp)); true.append((t, h))
    mm = C.score_waveforms(pred, true)
    print(f"(b) rank {k:2d}: median mismatch={np.median(mm):.2e} max={mm.max():.2e}")
print(f"done in {time.time()-t0:.1f}s")
