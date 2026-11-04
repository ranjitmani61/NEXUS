"""Research baseline: block-wise attention skipping with a certified error bound.
This is NOT claimed as an original algorithm or evidence of hardware speedup.
Requires numpy. Run: python processor_baseline.py
"""
import numpy as np

SEED, DIM, BLOCKS, BLOCK_LEN, TOL = 20261103, 64, 128, 32, 0.02
rng = np.random.default_rng(SEED)
q = rng.standard_normal(DIM)
V = rng.standard_normal((BLOCKS, BLOCK_LEN, DIM))

def test(kind):
    if kind == 'clustered':
        centers = rng.standard_normal((BLOCKS, DIM))
        centers[11] = 1.5*q + rng.normal(0, .15, DIM)
        K = centers[:, None, :] + rng.normal(0, .04, (BLOCKS, BLOCK_LEN, DIM))
    else:
        K = rng.standard_normal((BLOCKS, BLOCK_LEN, DIM))
    centroids = K.mean(axis=1)
    radii = np.linalg.norm(K - centroids[:, None, :], axis=2).max(axis=1)
    upper = (centroids @ q + np.linalg.norm(q)*radii) / np.sqrt(DIM)
    score = np.einsum('bkd,d->bk', K, q) / np.sqrt(DIM)
    # All scored weights are scaled by a safe global upper bound.
    shift = float(upper.max())
    exp_full = np.exp(score - shift)
    exact = np.einsum('bk,bkd->d', exp_full, V) / exp_full.sum()
    max_vnorm = np.linalg.norm(V, axis=-1).max()
    block_upper_mass = BLOCK_LEN * np.exp(upper - shift)
    unvisited_upper = block_upper_mass.sum()
    selected_mass = 0.0
    selected_weighted_values = np.zeros(DIM)
    selected = 0
    for b in np.argsort(-upper):
        unvisited_upper = max(0, unvisited_upper - block_upper_mass[b])
        selected_mass += exp_full[b].sum()
        selected_weighted_values += np.einsum('k,kd->d', exp_full[b], V[b])
        selected += 1
        relative_bound = 2*unvisited_upper/(selected_mass+unvisited_upper)
        if relative_bound <= TOL:
            break
    approximation = selected_weighted_values/selected_mass
    actual_normed_err = np.linalg.norm(approximation-exact)/max_vnorm
    baseline_bytes = 2*BLOCKS*BLOCK_LEN*DIM*4
    # Metadata: one d-float centroid and one float radius per block; values norms
    # assumed already known via a separate max register, not charged to each query.
    metadata_bytes = BLOCKS*(DIM+1)*4
    read_bytes = metadata_bytes+2*selected*BLOCK_LEN*DIM*4
    return {
        'input': kind,
        'blocks_read': selected,
        'blocks_total': BLOCKS,
        'fraction_KV_read': selected/BLOCKS,
        'relative_memory_traffic': round(read_bytes/baseline_bytes, 5),
        'certified_relative_error_bound': round(float(relative_bound), 8),
        'actual_relative_error': round(float(actual_normed_err), 8),
        'certificate_holds': bool(actual_normed_err <= relative_bound+1e-10),
    }

if __name__ == '__main__':
    for case in ('clustered','unstructured'):
        print(test(case))